#ifndef CPU_CANDIDATE_EXPORT_HPP
#define CPU_CANDIDATE_EXPORT_HPP

#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/batch/batch_runner.hpp"
#include "parcae/cli/console_progress_sink.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_dispatch.hpp"
#include "parcae/dsl/theory_envelope_bridge.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/beaufort_explicit_key_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/ciphertext_autokey_explicit_primer_candidate_generator.hpp"
#include "parcae/generate/compose_recipe_candidate_generator.hpp"
#include "parcae/generate/hill2_candidate_generator.hpp"
#include "parcae/generate/hill3_candidate_generator.hpp"
#include "parcae/generate/plaintext_autokey_explicit_primer_candidate_generator.hpp"
#include "parcae/generate/theory_explicit_params_candidate_generator.hpp"
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
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// CPU export: expand via `GeneratorRegistry`, apply `SearchPrior`, rank top-k
/// (`search-loop.md`). Wire shape matches `GpuCandidateExport::Result`.
///
/// Optional `BatchRunner::Progress` emits stage ticks (`expand` / `filter` /
/// then `score` via `RankCandidates`). Observe-only — results are identical
/// with or without a sink.
class CpuCandidateExport {
public:
    using Row = GpuCandidateExport::Row;
    using Result = GpuCandidateExport::Result;

    [[nodiscard]] static StatusOr<Result>
    from_job(std::span<const Index29> cipher, const SearchJob& job, const Context& ctx,
             const SearchPrior* prior = nullptr,
             BatchRunner::Progress progress = BatchRunner::Progress{}) {
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

        return run(cipher, job.family(), job.score_id(), job.k(), ctx, job.direction(),
                   job.param_grid(), prior_ptr, job.score_version(), job.max_candidates(),
                   progress);
    }

    [[nodiscard]] static StatusOr<Result>
    run(std::span<const Index29> cipher, std::string_view family, std::string_view score_id,
        std::size_t k, const Context& ctx,
        TransformDirection direction = TransformDirection::Decrypt,
        const nlohmann::json& param_grid = nlohmann::json::object(),
        const SearchPrior* prior = nullptr, std::string_view score_version = "v0",
        std::size_t max_candidates = SIZE_MAX,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (cipher.empty()) {
            return Status::error("CpuCandidateExport: ciphertext must be non-empty");
        }
        if (k == 0) {
            return Status::error("CpuCandidateExport: k must be >= 1");
        }

        if (progress.rune_count == 0) {
            progress.rune_count = cipher.size();
        }

        StatusOr<std::vector<TransformCandidate>> generated =
            expand_family(cipher, family, direction, param_grid, ctx);
        if (!generated.ok()) {
            return generated.status();
        }

        std::vector<TransformCandidate> candidates = std::move(generated.value());
        emit_stage(progress.sink, "expand", candidates.size(), candidates.size(),
                   progress.rune_count);

        if (prior != nullptr) {
            Status filter = filter_exclusions(candidates, *prior);
            if (!filter.ok()) {
                return filter;
            }
            StatusOr<std::vector<TransformCandidate>> merged =
                merge_seeds(cipher, direction, candidates, *prior, ctx.data_root() / "theories");
            if (!merged.ok()) {
                return merged.status();
            }
            candidates = std::move(merged.value());
            emit_stage(progress.sink, "filter", candidates.size(), candidates.size(),
                       progress.rune_count);
        }

        if (candidates.empty()) {
            return Status::error("CpuCandidateExport: no candidates after prior filters");
        }
        if (candidates.size() > max_candidates) {
            return Status::error("CpuCandidateExport: candidate expansion exceeds max_candidates");
        }

        const std::size_t effective_k = std::min(k, candidates.size());
        StatusOr<BatchResult> ranked = RankCandidates::run(
            candidates, score_id, effective_k, &ctx, {}, nlohmann::json::object(), score_version,
            BatchExecution::Serial, Backend::Cpu, progress);
        if (!ranked.ok()) {
            return ranked.status();
        }

        return rows_from_batch(ranked.value(), candidates, Backend::Cpu);
    }

    [[nodiscard]] static StatusOr<std::string_view>
    generator_id_for_family(std::string_view family) {
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
        if (family == "hill_2") {
            return Hill2CandidateGenerator::generator_id;
        }
        if (family == "hill_3") {
            return Hill3CandidateGenerator::generator_id;
        }
        if (family == "ciphertext_autokey") {
            return CiphertextAutokeyExplicitPrimerCandidateGenerator::generator_id;
        }
        if (family == "plaintext_autokey") {
            return PlaintextAutokeyExplicitPrimerCandidateGenerator::generator_id;
        }
        if (family == "totient") {
            return TotientOffsetCandidateGenerator::generator_id;
        }
        if (family == "compose") {
            return ComposeRecipeCandidateGenerator::generator_id;
        }
        if (family == "theory") {
            return TheoryExplicitParamsCandidateGenerator::generator_id;
        }
        return Status::error("CpuCandidateExport: unsupported family");
    }

    /// Vigenère / Beaufort / CTAK / PTAK: explicit keys/primers or bounded synthetic grid.
    /// Hill: `matrices` or sample (`max_candidates` / `seed`); empty → generator defaults.
    /// Totient: `prime_start_indices` or contiguous `0..prime_start_count-1`.
    /// Compose: empty → Atbash∘Caesar 29; or recipes / stages / template (passed through).
    /// Theory: `theory_uri` + `params_list` (passed through).
    /// Other families ignore `param_grid` (family default enumeration).
    [[nodiscard]] static StatusOr<nlohmann::json>
    generator_params_for_family(std::string_view family, const nlohmann::json& param_grid) {
        if (family == "vigenere" || family == "beaufort" || family == "ciphertext_autokey" ||
            family == "plaintext_autokey") {
            return keyed_family_params(param_grid);
        }
        if (family == "hill_2" || family == "hill_3") {
            return hill_family_params(param_grid);
        }
        if (family == "totient") {
            return totient_family_params(param_grid);
        }
        if (family == "compose") {
            return compose_family_params(param_grid);
        }
        if (family == "theory") {
            return theory_family_params(param_grid);
        }
        return nlohmann::json::object();
    }

private:
    CpuCandidateExport() = delete;

    static void emit_stage(ConsoleProgressSink* sink, std::string_view stage, std::size_t done,
                           std::size_t total, std::size_t rune_count) {
        if (sink == nullptr) {
            return;
        }
        ConsoleProgressSnapshot snap;
        snap.set_stage(std::string(stage));
        snap.set_candidates_done(done);
        snap.set_candidates_total(total);
        snap.set_rune_count(rune_count);
        sink->on_stage(stage, snap);
    }

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    expand_family(std::span<const Index29> cipher, std::string_view family,
                  TransformDirection direction, const nlohmann::json& param_grid,
                  const Context& ctx) {
        if (family == "theory") {
            StatusOr<nlohmann::json> gen_params = theory_family_params(param_grid);
            if (!gen_params.ok()) {
                return gen_params.status();
            }
            const std::string theory_uri = gen_params.value().at("theory_uri").get<std::string>();
            std::vector<nlohmann::json> params_list;
            params_list.reserve(gen_params.value().at("params_list").size());
            for (const auto& item : gen_params.value().at("params_list")) {
                params_list.push_back(item);
            }
            return TheoryExplicitParamsCandidateGenerator::generate(
                cipher, ctx.data_root() / "theories", theory_uri, params_list, direction);
        }

        StatusOr<std::string_view> generator_id = generator_id_for_family(family);
        if (!generator_id.ok()) {
            return generator_id.status();
        }

        StatusOr<nlohmann::json> gen_params = generator_params_for_family(family, param_grid);
        if (!gen_params.ok()) {
            return gen_params.status();
        }

        return GenerateCandidates::from_indices(generator_id.value(), cipher, direction,
                                                gen_params.value());
    }

    [[nodiscard]] static StatusOr<nlohmann::json>
    compose_family_params(const nlohmann::json& param_grid) {
        // Validate resolve; pass grid through so GeneratorRegistry can expand.
        StatusOr<std::vector<nlohmann::json>> recipes =
            ComposeRecipeCandidateGenerator::recipes_from_param_grid(
                param_grid.is_null() ? nlohmann::json::object() : param_grid);
        if (!recipes.ok()) {
            return Status::error(std::string("CpuCandidateExport: ") + recipes.status().message());
        }
        if (param_grid.is_null() || (param_grid.is_object() && param_grid.empty())) {
            return nlohmann::json::object();
        }
        return param_grid;
    }

    [[nodiscard]] static StatusOr<nlohmann::json>
    theory_family_params(const nlohmann::json& param_grid) {
        Status ok = SearchJob::validate_theory_param_grid(
            param_grid.is_null() ? nlohmann::json::object() : param_grid);
        if (!ok.ok()) {
            return Status::error(std::string("CpuCandidateExport: ") + ok.message());
        }
        return nlohmann::json{{"theory_uri", param_grid.at("theory_uri")},
                              {"params_list", param_grid.at("params_list")}};
    }

    [[nodiscard]] static StatusOr<nlohmann::json>
    hill_family_params(const nlohmann::json& param_grid) {
        if (param_grid.is_null() || (param_grid.is_object() && param_grid.empty())) {
            return nlohmann::json::object();
        }
        if (!param_grid.is_object()) {
            return Status::error("CpuCandidateExport: hill param_grid must be an object");
        }
        // Pass through: explicit `matrices` and/or sample `max_candidates` / `seed`.
        return param_grid;
    }

    [[nodiscard]] static StatusOr<nlohmann::json>
    keyed_family_params(const nlohmann::json& param_grid) {
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

    [[nodiscard]] static StatusOr<nlohmann::json>
    totient_family_params(const nlohmann::json& param_grid) {
        if (param_grid.is_object() && param_grid.contains("prime_start_indices")) {
            if (!param_grid.at("prime_start_indices").is_array()) {
                return Status::error(
                    "CpuCandidateExport: param_grid.prime_start_indices must be an array");
            }
            return nlohmann::json{{"prime_start_indices", param_grid.at("prime_start_indices")}};
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

    [[nodiscard]] static nlohmann::json
    key_indices_list_json(const std::vector<std::vector<Index29>>& keys) {
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

    [[nodiscard]] static Status filter_exclusions(std::vector<TransformCandidate>& candidates,
                                                  const SearchPrior& prior) {
        const auto end =
            std::remove_if(candidates.begin(), candidates.end(), [&](const TransformCandidate& c) {
                return prior.excludes_envelope(c.envelope());
            });
        candidates.erase(end, candidates.end());
        return Status::success();
    }

    [[nodiscard]] static bool has_matching_params(const std::vector<TransformCandidate>& candidates,
                                                  const nlohmann::json& params) {
        for (const TransformCandidate& c : candidates) {
            if (c.params() == params) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    merge_seeds(std::span<const Index29> cipher, TransformDirection job_direction,
                std::vector<TransformCandidate> candidates, const SearchPrior& prior,
                const std::filesystem::path& theories_root) {
        for (const SearchPrior::Seed& seed : prior.seeds()) {
            StatusOr<TransformCandidate> materialized =
                candidate_from_seed(cipher, seed, job_direction, theories_root);
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

    [[nodiscard]] static StatusOr<TransformCandidate>
    candidate_from_seed(std::span<const Index29> cipher, const SearchPrior::Seed& seed,
                        TransformDirection job_direction,
                        const std::filesystem::path& theories_root) {
        if (!seed.envelope().is_object()) {
            return Status::error("CpuCandidateExport: seed envelope must be an object");
        }
        StatusOr<TransformEnvelope> parsed = TransformEnvelope::from_json(seed.envelope());
        if (!parsed.ok()) {
            return parsed.status();
        }

        const TransformEnvelope& envelope = parsed.value();

        std::optional<nlohmann::json> interrupt;
        if (!envelope.interrupt().skip_indices().empty()) {
            interrupt = envelope.interrupt().to_json();
        }

        const std::string candidate_id = "prior-seed:" + seed.hypothesis_id();

        StatusOr<TheoryUri> theory_uri = TheoryUri::parse(envelope.transform_id().str());
        if (theory_uri.ok()) {
            TheoryEnvelopeBridge::Envelope theory_env{TheoryEnvelopeBridge::Kind::Theory,
                                                      theory_uri.value().to_string(),
                                                      job_direction,
                                                      envelope.params(),
                                                      envelope.interrupt(),
                                                      theory_uri.value()};
            StatusOr<std::vector<Index29>> plain =
                TheoryDispatch::apply(theories_root, theory_env, cipher);
            if (!plain.ok()) {
                return plain.status();
            }
            return TransformCandidate(
                candidate_id, TransformId::unchecked(theory_uri.value().to_string()), job_direction,
                envelope.params(), std::move(plain.value()), std::move(interrupt));
        }

        const TransformEnvelope call(envelope.transform_id(), job_direction, envelope.params(),
                                     envelope.interrupt());
        StatusOr<std::vector<Index29>> plain =
            ToolApi::apply_to_indices(cipher, call, Backend::Cpu);
        if (!plain.ok()) {
            return plain.status();
        }

        return TransformCandidate(candidate_id, envelope.transform_id(), job_direction,
                                  envelope.params(), std::move(plain.value()),
                                  std::move(interrupt));
    }

    [[nodiscard]] static StatusOr<Result>
    rows_from_batch(const BatchResult& batch, std::span<const TransformCandidate> candidates,
                    Backend backend) {
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

#endif // CPU_CANDIDATE_EXPORT_HPP
