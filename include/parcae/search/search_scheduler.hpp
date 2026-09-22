#ifndef SEARCH_SCHEDULER_HPP
#define SEARCH_SCHEDULER_HPP

#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/search/batch_artifact.hpp"
#include "parcae/search/cpu_candidate_export.hpp"
#include "parcae/search/gpu_candidate_export.hpp"
#include "parcae/search/hypothesis_bridge.hpp"
#include "parcae/search/search_job.hpp"
#include "parcae/search/search_prior.hpp"
#include "parcae/search/workspace_cipher.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// One closed-loop search cycle (or later: multi-iteration loop).
/// Normative: `docs/spec/search-loop.md` (`SearchScheduler` / `parcae.search_cycle_result.v0`).
class SearchScheduler {
public:
    static constexpr std::string_view result_schema_id = "parcae.search_cycle_result.v0";

    /// Options for `run_once` (deterministic when `created_utc` / `batch_id` are fixed).
    class Options {
    public:
        /// RFC 3339 UTC used for batch + prior build timestamps (required, non-empty).
        std::string created_utc;
        /// When set, used as `BatchArtifact.batch_id`; otherwise derived from digests.
        std::optional<std::string> batch_id;
        /// When true, omit optional `report.json` (agent-safe).
        bool omit_timing = true;
    };

    /// One batch summary line inside the cycle result.
    class BatchSummary {
    public:
        BatchSummary() = default;

        BatchSummary(
            std::string batch_id,
            std::size_t candidate_count,
            std::string job_digest_sha256)
            : batch_id_(std::move(batch_id)),
              candidate_count_(candidate_count),
              job_digest_sha256_(std::move(job_digest_sha256)) {}

        [[nodiscard]] const std::string& batch_id() const noexcept {
            return batch_id_;
        }

        [[nodiscard]] std::size_t candidate_count() const noexcept {
            return candidate_count_;
        }

        [[nodiscard]] const std::string& job_digest_sha256() const noexcept {
            return job_digest_sha256_;
        }

        [[nodiscard]] nlohmann::json to_json() const {
            return nlohmann::json{
                {"batch_id", batch_id_},
                {"candidate_count", candidate_count_},
                {"job_digest_sha256", job_digest_sha256_},
            };
        }

    private:
        std::string batch_id_;
        std::size_t candidate_count_ = 0;
        std::string job_digest_sha256_;
    };

    /// `parcae.search_cycle_result.v0` payload (also `to_json()` for tool envelopes).
    class CycleResult {
    public:
        CycleResult() = default;

        [[nodiscard]] const std::string& workspace_id() const noexcept {
            return workspace_id_;
        }

        [[nodiscard]] std::size_t iterations() const noexcept {
            return iterations_;
        }

        [[nodiscard]] const std::string& stop_reason() const noexcept {
            return stop_reason_;
        }

        [[nodiscard]] const std::vector<BatchSummary>& batches() const noexcept {
            return batches_;
        }

        [[nodiscard]] std::size_t hypotheses_written() const noexcept {
            return hypotheses_written_;
        }

        [[nodiscard]] bool omit_timing() const noexcept {
            return omit_timing_;
        }

        [[nodiscard]] nlohmann::json to_json() const {
            nlohmann::json batches = nlohmann::json::array();
            for (const BatchSummary& b : batches_) {
                batches.push_back(b.to_json());
            }
            return nlohmann::json{
                {"schema", std::string(result_schema_id)},
                {"workspace_id", workspace_id_},
                {"iterations", iterations_},
                {"stop_reason", stop_reason_},
                {"batches", std::move(batches)},
                {"hypotheses_written", hypotheses_written_},
                {"omit_timing", omit_timing_},
            };
        }

    private:
        friend class SearchScheduler;
        std::string workspace_id_;
        std::size_t iterations_ = 0;
        std::string stop_reason_;
        std::vector<BatchSummary> batches_;
        std::size_t hypotheses_written_ = 0;
        bool omit_timing_ = true;
    };

    /// Run one cycle: cipher → prior → export → `BatchArtifact` → `HypothesisBridge`.
    [[nodiscard]] static StatusOr<CycleResult> run_once(
        const std::filesystem::path& data_root,
        const parcae::tool::Context& ctx,
        const SearchJob& job,
        const Options& options) {
        if (options.created_utc.empty()) {
            return Status::error("SearchScheduler::run_once requires Options.created_utc");
        }
        if (job.workspace_id().empty()) {
            return Status::error("SearchScheduler::run_once: job.workspace_id is required");
        }

        Status usable = parcae::tool::BackendUtil::ensure_usable(job.backend());
        if (!usable.ok()) {
            return usable;
        }

        StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::load(data_root, job.workspace_id());
        if (!cipher.ok()) {
            return cipher.status();
        }

        StatusOr<SearchPrior> prior = SearchPrior::from_workspace(
            data_root, job.workspace_id(), options.created_utc);
        if (!prior.ok()) {
            return prior.status();
        }

        StatusOr<CpuCandidateExport::Result> exported =
            export_candidates(cipher.value().indices(), job, ctx, prior.value());
        if (!exported.ok()) {
            return exported.status();
        }

        CycleResult result;
        result.workspace_id_ = job.workspace_id();
        result.iterations_ = 1;
        result.omit_timing_ = options.omit_timing;

        if (exported.value().size() == 0) {
            result.stop_reason_ = "no_new_candidates";
            result.hypotheses_written_ = 0;
            return result;
        }

        const std::string job_digest = job.job_digest_sha256();
        const std::string prior_digest = prior.value().prior_digest_sha256();

        std::string batch_id;
        if (options.batch_id.has_value()) {
            StatusOr<std::string> bid = WorkspacePaths::validate_id(*options.batch_id);
            if (!bid.ok()) {
                return bid.status();
            }
            batch_id = std::move(bid.value());
        } else {
            batch_id = make_batch_id(
                job.workspace_id(), job.family(), job_digest, prior_digest, options.created_utc);
        }

        std::vector<nlohmann::json> lines;
        lines.reserve(exported.value().size());
        for (const CpuCandidateExport::Row& row : exported.value().rows()) {
            lines.push_back(BatchArtifact::candidate_wire(
                row.candidate(),
                job.score_id(),
                job.score_version(),
                row.score(),
                exported.value().backend(),
                row.rank()));
        }

        std::optional<nlohmann::json> report;
        if (!options.omit_timing) {
            report = nlohmann::json{
                {"candidate_count", lines.size()},
                {"backend",
                 std::string(parcae::tool::BackendUtil::to_string(exported.value().backend()))},
            };
        }

        StatusOr<BatchArtifact> artifact = BatchArtifact::make(
            job.workspace_id(),
            batch_id,
            options.created_utc,
            job_digest,
            prior_digest,
            job.family(),
            job.score_id(),
            job.score_version(),
            exported.value().backend(),
            job.k(),
            job.seed(),
            std::move(lines),
            std::move(report));
        if (!artifact.ok()) {
            return artifact.status();
        }

        Status stored = artifact.value().store(data_root);
        if (!stored.ok()) {
            return stored;
        }

        StatusOr<HypothesisBridge::Result> ingested =
            HypothesisBridge::ingest(data_root, artifact.value());
        if (!ingested.ok()) {
            return ingested.status();
        }

        result.batches_.emplace_back(
            artifact.value().batch_id(),
            artifact.value().candidate_count(),
            job_digest);
        result.hypotheses_written_ = ingested.value().written_count();
        result.stop_reason_ = "completed_iterations";
        return result;
    }

    /// Deterministic batch id: `b` + 32 hex of SHA-256(preimage).
    [[nodiscard]] static std::string make_batch_id(
        std::string_view workspace_id,
        std::string_view family,
        std::string_view job_digest_sha256,
        std::string_view prior_digest_sha256,
        std::string_view created_utc) {
        const std::string material = std::string(workspace_id) + '\n' + std::string(family) +
                                     '\n' + std::string(job_digest_sha256) + '\n' +
                                     std::string(prior_digest_sha256) + '\n' +
                                     std::string(created_utc);
        return std::string("b") + Sha256::hex_digest(material).substr(0, 32);
    }

private:
    SearchScheduler() = delete;

    [[nodiscard]] static StatusOr<CpuCandidateExport::Result> export_candidates(
        std::span<const Index29> cipher,
        const SearchJob& job,
        const parcae::tool::Context& ctx,
        const SearchPrior& prior) {
        if (job.backend() == parcae::tool::Backend::Cpu) {
            return CpuCandidateExport::from_job(cipher, job, ctx, &prior);
        }

        // CUDA fused export is χ²-only (GpuCandidateExport).
        if (job.score_id() != GpuCandidateExport::score_id) {
            return Status::error(
                "SearchScheduler: backend=cuda requires score_id chi2_english_gp_v0");
        }
        if (job.direction() != TransformDirection::Decrypt) {
            return Status::error(
                "SearchScheduler: backend=cuda fused export supports decrypt only");
        }

        StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
        if (!freqs.ok()) {
            return freqs.status();
        }

        StatusOr<CpuCandidateExport::Result> fused =
            export_cuda_fused(cipher, job, freqs.value());
        if (!fused.ok()) {
            return fused.status();
        }

        // Apply prior exclusions/seeds on the CPU path for parity with CpuCandidateExport.
        // Fused GPU export does not yet fold SearchPrior; re-run CPU with prior when
        // the prior is non-empty so exclusions/seeds match the CPU oracle.
        if (!prior.seeds().empty() || !prior.exclusions().empty()) {
            return CpuCandidateExport::from_job(cipher, job, ctx, &prior);
        }
        return fused;
    }

    [[nodiscard]] static StatusOr<CpuCandidateExport::Result> export_cuda_fused(
        std::span<const Index29> cipher,
        const SearchJob& job,
        const ExpectedFrequencyTable& freqs) {
        const std::string& family = job.family();
        if (family == "caesar") {
            return GpuCandidateExport::caesar(cipher, freqs, job.k(), job.direction());
        }
        if (family == "atbash") {
            return GpuCandidateExport::atbash(cipher, freqs, job.k(), job.direction());
        }
        if (family == "atbash_caesar") {
            return GpuCandidateExport::atbash_caesar(cipher, freqs, job.k(), job.direction());
        }
        if (family == "affine") {
            return GpuCandidateExport::affine(cipher, freqs, job.k(), job.direction());
        }
        if (family == "vigenere") {
            std::size_t max_len = GpuCandidateExport::default_vigenere_max_key_length;
            if (job.param_grid().is_object() && job.param_grid().contains("max_key_length") &&
                job.param_grid().at("max_key_length").is_number_integer()) {
                const std::int64_t v = job.param_grid().at("max_key_length").get<std::int64_t>();
                if (v < 1) {
                    return Status::error("SearchScheduler: param_grid.max_key_length must be >= 1");
                }
                max_len = static_cast<std::size_t>(v);
            }
            return GpuCandidateExport::vigenere_bounded(
                cipher, freqs, job.k(), max_len, job.direction());
        }
        return Status::error(
            "SearchScheduler: unsupported family for cuda export: " + family);
    }
};

#endif  // SEARCH_SCHEDULER_HPP
