#ifndef CPU_CANDIDATE_EXPORT_HPP
#define CPU_CANDIDATE_EXPORT_HPP

#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/beaufort_explicit_key_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/totient_offset_candidate_generator.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/search/gpu_candidate_export.hpp"
#include "parcae/search/search_job.hpp"
#include "parcae/search/search_prior.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/generate_candidates.hpp"
#include "parcae/tool/rank_candidates.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/tool/transform_envelope.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// CPU export: expand via `GeneratorRegistry`, apply `SearchPrior`, rank top-k
/// (`search-loop.md`). Wire shape matches `GpuCandidateExport::Result`.
class CpuCandidateExport {
public:
    using Row = GpuCandidateExport::Row;
    using Result = GpuCandidateExport::Result;

    [[nodiscard]] static StatusOr<Result> from_job(
        std::span<const Index29> cipher,
        const SearchJob& job,
        const parcae::tool::Context& ctx,
        const SearchPrior* prior = nullptr) {
        std::optional<SearchPrior> owned_prior;
        const SearchPrior* prior_ptr = prior;
        if (prior_ptr == nullptr && job.prior().has_value() && job.prior()->is_object()) {
            StatusOr<SearchPrior> parsed = SearchPrior::from_json(job.prior().value());
            if (!parsed.ok()) {
                return parsed.status();
            }
            owned_prior = std::move(parsed.value());
            prior_ptr = &owned_prior.value();
        }

        return run(
            cipher,
            job.family(),
            job.score_id(),
            job.k(),
            ctx,
            job.direction(),
            job.param_grid(),
            prior_ptr,
            job.score_version(),
            job.max_candidates());
    }

    [[nodiscard]] static StatusOr<Result> run(
        std::span<const Index29> cipher,
        std::string_view family,
        std::string_view score_id,
        std::size_t k,
        const parcae::tool::Context& ctx,
        TransformDirection direction = TransformDirection::Decrypt,
        const nlohmann::json& param_grid = nlohmann::json::object(),
        const SearchPrior* prior = nullptr,
        std::string_view score_version = "v0",
        std::size_t max_candidates = SIZE_MAX) {
        if (cipher.empty()) {
            return Status::error("CpuCandidateExport: ciphertext must be non-empty");
        }
        if (k == 0) {
            return Status::error("CpuCandidateExport: k must be >= 1");
        }

        StatusOr<std::string_view> generator_id = generator_id_for_family(family);
        if (!generator_id.ok()) {
            return generator_id.status();
        }

        StatusOr<nlohmann::json> gen_params = generator_params_for_family(family, param_grid);
        if (!gen_params.ok()) {
            return gen_params.status();
        }

        StatusOr<std::vector<TransformCandidate>> generated = GenerateCandidates::from_indices(
            generator_id.value(), cipher, direction, gen_params.value());
        if (!generated.ok()) {
            return generated.status();
        }

        std::vector<TransformCandidate> candidates = std::move(generated.value());
        if (prior != nullptr) {
            Status filter = filter_exclusions(candidates, *prior);
            if (!filter.ok()) {
                return filter;
            }
            StatusOr<std::vector<TransformCandidate>> merged =
                merge_seeds(cipher, direction, candidates, *prior);
            if (!merged.ok()) {
                return merged.status();
            }
            candidates = std::move(merged.value());
        }

        if (candidates.empty()) {
            return Status::error("CpuCandidateExport: no candidates after prior filters");
        }
        if (candidates.size() > max_candidates) {
            return Status::error(
                "CpuCandidateExport: candidate expansion exceeds max_candidates");
        }

        const std::size_t effective_k = std::min(k, candidates.size());
        StatusOr<BatchResult> ranked = RankCandidates::run(
            candidates,
            score_id,
            effective_k,
            &ctx,
            {},
            nlohmann::json::object(),
            score_version);
        if (!ranked.ok()) {
            return ranked.status();
        }

        return rows_from_batch(ranked.value(), candidates, parcae::tool::Backend::Cpu);
    }

    [[nodiscard]] static StatusOr<std::string_view> generator_id_for_family(
        std::string_view family) {
        if (family == "caesar") {
            return CaesarCandidateGenerator::generator_id;
        }
        if (family == "atbash") {
            return AtbashCandidateGenerator::generator_id;
        }
        if (family == "atbash_caesar") {
            return AtbashCaesarCandidateGenerator::generator_id;
        }
        if (family == "affine") {
            return AffineCandidateGenerator::generator_id;
        }
        if (family == "vigenere") {
            return VigenereExplicitKeyCandidateGenerator::generator_id;
        }
        if (family == "beaufort") {
            return BeaufortExplicitKeyCandidateGenerator::generator_id;
        }
        if (family == "totient") {
            return TotientOffsetCandidateGenerator::generator_id;
        }
        return Status::error("CpuCandidateExport: unsupported family");
    }

    /// Vigenère / Beaufort: explicit keys or bounded synthetic grid.
    /// Totient: `prime_start_indices` or contiguous `0..prime_start_count-1`.
    /// Other families ignore `param_grid` (family default enumeration).
    [[nodiscard]] static StatusOr<nlohmann::json> generator_params_for_family(
        std::string_view family,
        const nlohmann::json& param_grid) {
        if (family == "vigenere" || family == "beaufort") {
            return keyed_family_params(param_grid);
        }
        if (family == "totient") {
            return totient_family_params(param_grid);
        }
        return nlohmann::json::object();
    }

private:
    CpuCandidateExport() = delete;

    [[nodiscard]] static StatusOr<nlohmann::json> keyed_family_params(
        const nlohmann::json& param_grid) {
        if (param_grid.is_object() && !param_grid.empty()) {
            if (param_grid.contains("key_indices_list") || param_grid.contains("keys")) {
                return param_grid;
            }
            std::size_t max_len = GpuCandidateExport::default_vigenere_max_key_length;
            if (param_grid.contains("max_key_length")) {
                if (!param_grid.at("max_key_length").is_number_integer()) {
                    return Status::error(
                        "CpuCandidateExport: param_grid.max_key_length must be an integer");
                }
                const std::int64_t v = param_grid.at("max_key_length").get<std::int64_t>();
                if (v < 1) {
                    return Status::error(
                        "CpuCandidateExport: param_grid.max_key_length must be >= 1");
                }
                max_len = static_cast<std::size_t>(v);
            }
            StatusOr<std::vector<std::vector<Index29>>> keys =
                GpuCandidateExport::default_bounded_key_grid(max_len);
            if (!keys.ok()) {
                return keys.status();
            }
            return key_indices_list_json(keys.value());
        }

        StatusOr<std::vector<std::vector<Index29>>> keys =
            GpuCandidateExport::default_bounded_key_grid(
                GpuCandidateExport::default_vigenere_max_key_length);
        if (!keys.ok()) {
            return keys.status();
        }
        return key_indices_list_json(keys.value());
    }

    [[nodiscard]] static StatusOr<nlohmann::json> totient_family_params(
        const nlohmann::json& param_grid) {
        if (param_grid.is_object() && param_grid.contains("prime_start_indices")) {
            if (!param_grid.at("prime_start_indices").is_array()) {
                return Status::error(
                    "CpuCandidateExport: param_grid.prime_start_indices must be an array");
            }
            return nlohmann::json{
                {"prime_start_indices", param_grid.at("prime_start_indices")}};
        }
        std::size_t count = GpuCandidateExport::default_totient_start_count;
        if (param_grid.is_object() && param_grid.contains("prime_start_count")) {
            if (!param_grid.at("prime_start_count").is_number_integer()) {
                return Status::error(
                    "CpuCandidateExport: param_grid.prime_start_count must be an integer");
            }
            const std::int64_t v = param_grid.at("prime_start_count").get<std::int64_t>();
            if (v < 1) {
                return Status::error(
                    "CpuCandidateExport: param_grid.prime_start_count must be >= 1");
            }
            count = static_cast<std::size_t>(v);
        }
        StatusOr<std::vector<std::size_t>> starts =
            GpuCandidateExport::default_totient_starts(count);
        if (!starts.ok()) {
            return starts.status();
        }
        nlohmann::json list = nlohmann::json::array();
        for (const std::size_t s : starts.value()) {
            list.push_back(s);
        }
        return nlohmann::json{{"prime_start_indices", std::move(list)}};
    }

    [[nodiscard]] static nlohmann::json key_indices_list_json(
        const std::vector<std::vector<Index29>>& keys) {
        nlohmann::json list = nlohmann::json::array();
        for (const std::vector<Index29>& key : keys) {
            nlohmann::json row = nlohmann::json::array();
            for (const Index29 idx : key) {
                row.push_back(idx.value());
            }
            list.push_back(std::move(row));
        }
        return nlohmann::json{{"key_indices_list", std::move(list)}};
    }

    [[nodiscard]] static Status filter_exclusions(
        std::vector<TransformCandidate>& candidates,
        const SearchPrior& prior) {
        const auto end = std::remove_if(
            candidates.begin(),
            candidates.end(),
            [&](const TransformCandidate& c) { return prior.excludes_envelope(c.envelope()); });
        candidates.erase(end, candidates.end());
        return Status::success();
    }

    [[nodiscard]] static bool has_matching_params(
        const std::vector<TransformCandidate>& candidates,
        const nlohmann::json& params) {
        for (const TransformCandidate& c : candidates) {
            if (c.params() == params) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> merge_seeds(
        std::span<const Index29> cipher,
        TransformDirection job_direction,
        std::vector<TransformCandidate> candidates,
        const SearchPrior& prior) {
        for (const SearchPrior::Seed& seed : prior.seeds()) {
            StatusOr<TransformCandidate> materialized =
                candidate_from_seed(cipher, seed, job_direction);
            if (!materialized.ok()) {
                return materialized.status();
            }
            if (has_matching_params(candidates, materialized.value().params())) {
                continue;
            }
            candidates.push_back(std::move(materialized.value()));
        }
        return candidates;
    }

    [[nodiscard]] static StatusOr<TransformCandidate> candidate_from_seed(
        std::span<const Index29> cipher,
        const SearchPrior::Seed& seed,
        TransformDirection job_direction) {
        if (!seed.envelope().is_object()) {
            return Status::error("CpuCandidateExport: seed envelope must be an object");
        }
        StatusOr<parcae::tool::TransformEnvelope> parsed =
            parcae::tool::TransformEnvelope::from_json(seed.envelope());
        if (!parsed.ok()) {
            return parsed.status();
        }

        const parcae::tool::TransformEnvelope& envelope = parsed.value();

        const parcae::tool::TransformEnvelope call(
            envelope.transform_id(),
            job_direction,
            envelope.params(),
            envelope.interrupt());
        StatusOr<std::vector<Index29>> plain =
            parcae::tool::apply_to_indices(cipher, call, parcae::tool::Backend::Cpu);
        if (!plain.ok()) {
            return plain.status();
        }

        std::optional<nlohmann::json> interrupt;
        if (!envelope.interrupt().skip_indices().empty()) {
            interrupt = envelope.interrupt().to_json();
        }

        const std::string candidate_id = "prior-seed:" + seed.hypothesis_id();
        return TransformCandidate(
            candidate_id,
            envelope.transform_id(),
            job_direction,
            envelope.params(),
            std::move(plain.value()),
            std::move(interrupt));
    }

    [[nodiscard]] static StatusOr<Result> rows_from_batch(
        const BatchResult& batch,
        std::span<const TransformCandidate> candidates,
        parcae::tool::Backend backend) {
        std::vector<Row> rows;
        rows.reserve(batch.top().size());
        for (std::size_t rank = 0; rank < batch.top().size(); ++rank) {
            const BatchHit& hit = batch.top()[rank];
            if (hit.source_index() >= candidates.size()) {
                return Status::error("CpuCandidateExport: batch source_index out of range");
            }
            const TransformCandidate& candidate = candidates[hit.source_index()];
            if (hit.candidate_id() != candidate.candidate_id()) {
                return Status::error(
                    "CpuCandidateExport: batch candidate_id mismatch (ordering drift)");
            }
            rows.emplace_back(candidate, hit.score(), rank, hit.source_index());
        }
        return Result{std::move(rows), backend};
    }
};

#endif  // CPU_CANDIDATE_EXPORT_HPP
