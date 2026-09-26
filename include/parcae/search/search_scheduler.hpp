#ifndef SEARCH_SCHEDULER_HPP
#define SEARCH_SCHEDULER_HPP

#include "parcae/batch/batch_runner.hpp"
#include "parcae/cli/console_progress_sink.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"
#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/hypothesis_record.hpp"
#include "parcae/hypothesis/hypothesis_status.hpp"
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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// One closed-loop search cycle or multi-iteration loop.
/// Normative: `docs/spec/search-loop.md` (`SearchScheduler` / `parcae.search_cycle_result.v0`).
/// Optional `Options::progress` / `LoopOptions::progress` is observe-only (digests
/// unchanged when null).
class SearchScheduler {
public:
    static constexpr std::string_view result_schema_id = "parcae.search_cycle_result.v0";

    static constexpr std::string_view stop_completed_iterations = "completed_iterations";
    static constexpr std::string_view stop_wall_budget = "wall_budget";
    static constexpr std::string_view stop_no_new_candidates = "no_new_candidates";
    static constexpr std::string_view stop_success_promoted = "success_promoted";
    static constexpr std::string_view stop_success_validate = "success_validate";
    static constexpr std::string_view stop_error = "error";

    /// Options for `run_once` (deterministic when `created_utc` / `batch_id` are fixed).
    class Options {
    public:
        /// RFC 3339 UTC used for batch + prior build timestamps (required, non-empty).
        std::string created_utc;
        /// When set, used as `BatchArtifact.batch_id`; otherwise derived from digests.
        std::optional<std::string> batch_id;
        /// When true, omit optional `report.json` (agent-safe).
        bool omit_timing = true;
        /// Workspace prior build flags (ignored when `SearchJob.prior` is inline).
        SearchPrior::BuildOptions prior_build{};
        /// Optional observe-only progress sink (nullptr = silent). Digests unchanged.
        ConsoleProgressSink* progress = nullptr;
    };

    /// Budgets and stop policy for `run_loop` (`search-loop.md`).
    class LoopOptions {
    public:
        /// RFC 3339 UTC for iteration 0 (required). Later iterations add
        /// `created_utc_step_seconds` when `created_utcs` is empty.
        std::string created_utc;
        /// Optional per-iteration UTC overrides (index 0..max_iterations-1).
        std::vector<std::string> created_utcs;
        /// Seconds added to `created_utc` per iteration when `created_utcs` is empty.
        std::int64_t created_utc_step_seconds = 1;
        /// Optional per-iteration batch ids (same indexing). Empty → auto digest ids.
        std::vector<std::string> batch_ids;
        /// Maximum `run_once` invocations (≥ 1).
        std::size_t max_iterations = 1;
        /// Wall-time budget in seconds. `nullopt` = unlimited. `0` stops before any cycle.
        std::optional<double> max_wall_seconds;
        /// When true, omit optional `report.json` on each cycle.
        bool omit_timing = true;
        /// Stop after an iteration if any hypothesis is `promoted`.
        bool stop_on_promoted = false;
        /// Reserved: stop when a validate hook reports success (requires `validate_ok`).
        bool stop_on_validate = false;
        /// Optional validate success probe (only consulted when `stop_on_validate`).
        /// When null and `stop_on_validate` is true, `run_loop` errors.
        const bool* validate_ok = nullptr;
        /// Forwarded to each `run_once` prior rebuild from the workspace.
        SearchPrior::BuildOptions prior_build{};
        /// Optional observe-only progress sink (nullptr = silent). Digests unchanged.
        ConsoleProgressSink* progress = nullptr;
    };

    /// One batch summary line inside the cycle result.
    class BatchSummary {
    public:
        BatchSummary() = default;

        BatchSummary(std::string batch_id, std::size_t candidate_count,
                     std::string job_digest_sha256)
            : batch_id_(std::move(batch_id)), candidate_count_(candidate_count),
              job_digest_sha256_(std::move(job_digest_sha256)) {}

        [[nodiscard]] const std::string& batch_id() const noexcept { return batch_id_; }

        [[nodiscard]] std::size_t candidate_count() const noexcept { return candidate_count_; }

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

        [[nodiscard]] const std::string& workspace_id() const noexcept { return workspace_id_; }

        [[nodiscard]] std::size_t iterations() const noexcept { return iterations_; }

        [[nodiscard]] const std::string& stop_reason() const noexcept { return stop_reason_; }

        [[nodiscard]] const std::vector<BatchSummary>& batches() const noexcept { return batches_; }

        [[nodiscard]] std::size_t hypotheses_written() const noexcept {
            return hypotheses_written_;
        }

        [[nodiscard]] bool omit_timing() const noexcept { return omit_timing_; }

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

    /// Run one cycle: cipher → prior (workspace or inline job) → export → batch → bridge.
    /// Each call rebuilds `SearchPrior` so promoted seeds / rejected exclusions from the
    /// workspace feed the next job (`search-loop.md` / search-engine cycle step 6).
    [[nodiscard]] static StatusOr<CycleResult> run_once(const std::filesystem::path& data_root,
                                                        const Context& ctx, const SearchJob& job,
                                                        const Options& options) {
        if (options.created_utc.empty()) {
            return Status::error("SearchScheduler::run_once requires Options.created_utc");
        }
        if (job.workspace_id().empty()) {
            return Status::error("SearchScheduler::run_once: job.workspace_id is required");
        }

        Status usable = BackendUtil::ensure_usable(job.backend());
        if (!usable.ok()) {
            return usable;
        }

        StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::load(data_root, job.workspace_id());
        if (!cipher.ok()) {
            return cipher.status();
        }
        emit_stage(options.progress, "load", 1, 1, cipher.value().indices().size());

        StatusOr<SearchPrior> prior = resolve_prior(data_root, job, options);
        if (!prior.ok()) {
            return prior.status();
        }
        emit_stage(options.progress, "prior", 1, 1, cipher.value().indices().size());

        BatchRunner::Progress export_progress;
        export_progress.sink = options.progress;
        export_progress.rune_count = cipher.value().indices().size();

        StatusOr<CpuCandidateExport::Result> exported =
            export_candidates(cipher.value().indices(), job, ctx, prior.value(), export_progress);
        if (!exported.ok()) {
            return exported.status();
        }

        CycleResult result;
        result.workspace_id_ = job.workspace_id();
        result.iterations_ = 1;
        result.omit_timing_ = options.omit_timing;

        if (exported.value().size() == 0) {
            result.stop_reason_ = std::string(stop_no_new_candidates);
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
            batch_id = make_batch_id(job.workspace_id(), job.family(), job_digest, prior_digest,
                                     options.created_utc);
        }

        std::vector<nlohmann::json> lines;
        lines.reserve(exported.value().size());
        for (const CpuCandidateExport::Row& row : exported.value().rows()) {
            lines.push_back(BatchArtifact::candidate_wire(row.candidate(), job.score_id(),
                                                          job.score_version(), row.score(),
                                                          exported.value().backend(), row.rank()));
        }

        std::optional<nlohmann::json> report;
        if (!options.omit_timing) {
            report = nlohmann::json{
                {"candidate_count", lines.size()},
                {"backend", std::string(BackendUtil::to_string(exported.value().backend()))},
            };
        }

        StatusOr<BatchArtifact> artifact = BatchArtifact::make(
            job.workspace_id(), batch_id, options.created_utc, job_digest, prior_digest,
            job.family(), job.score_id(), job.score_version(), exported.value().backend(), job.k(),
            job.seed(), std::move(lines), std::move(report));
        if (!artifact.ok()) {
            return artifact.status();
        }

        Status stored = artifact.value().store(data_root);
        if (!stored.ok()) {
            return stored;
        }
        emit_stage(options.progress, "write", artifact.value().candidate_count(),
                   artifact.value().candidate_count(), cipher.value().indices().size());

        StatusOr<HypothesisBridge::Result> ingested =
            HypothesisBridge::ingest(data_root, artifact.value());
        if (!ingested.ok()) {
            return ingested.status();
        }
        emit_stage(options.progress, "bridge", ingested.value().written_count(),
                   ingested.value().written_count(), cipher.value().indices().size());

        result.batches_.emplace_back(artifact.value().batch_id(),
                                     artifact.value().candidate_count(), job_digest);
        result.hypotheses_written_ = ingested.value().written_count();
        result.stop_reason_ = std::string(stop_completed_iterations);
        return result;
    }

    /// Repeat `run_once` until budget / stop reason (`search-loop.md` § `run_loop`).
    [[nodiscard]] static StatusOr<CycleResult> run_loop(const std::filesystem::path& data_root,
                                                        const Context& ctx, const SearchJob& job,
                                                        const LoopOptions& options) {
        if (options.created_utc.empty() && options.created_utcs.empty()) {
            return Status::error("SearchScheduler::run_loop requires created_utc");
        }
        if (options.max_iterations == 0) {
            return Status::error("SearchScheduler::run_loop: max_iterations must be >= 1");
        }
        if (options.max_wall_seconds.has_value() && *options.max_wall_seconds < 0.0) {
            return Status::error("SearchScheduler::run_loop: max_wall_seconds must be >= 0");
        }
        if (options.stop_on_validate && options.validate_ok == nullptr) {
            return Status::error(
                "SearchScheduler::run_loop: stop_on_validate requires validate_ok pointer");
        }
        if (!options.batch_ids.empty() && options.batch_ids.size() < options.max_iterations) {
            return Status::error(
                "SearchScheduler::run_loop: batch_ids must cover max_iterations when set");
        }
        if (!options.created_utcs.empty() && options.created_utcs.size() < options.max_iterations) {
            return Status::error(
                "SearchScheduler::run_loop: created_utcs must cover max_iterations when set");
        }

        CycleResult result;
        result.workspace_id_ = job.workspace_id();
        result.omit_timing_ = options.omit_timing;
        result.iterations_ = 0;
        result.hypotheses_written_ = 0;

        const auto wall0 = std::chrono::steady_clock::now();
        auto wall_exceeded = [&]() -> bool {
            if (!options.max_wall_seconds.has_value()) {
                return false;
            }
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
            return elapsed >= *options.max_wall_seconds;
        };

        if (wall_exceeded()) {
            result.stop_reason_ = std::string(stop_wall_budget);
            return result;
        }

        for (std::size_t i = 0; i < options.max_iterations; ++i) {
            if (i > 0 && wall_exceeded()) {
                result.stop_reason_ = std::string(stop_wall_budget);
                return result;
            }

            StatusOr<std::string> utc = utc_for_iteration(options, i);
            if (!utc.ok()) {
                return utc.status();
            }

            Options once;
            once.created_utc = std::move(utc.value());
            once.omit_timing = options.omit_timing;
            once.prior_build = options.prior_build;
            once.progress = options.progress;
            if (!options.batch_ids.empty()) {
                once.batch_id = options.batch_ids[i];
            }

            emit_stage(options.progress, "iteration", i + 1, options.max_iterations,
                       /*rune_count=*/0);

            StatusOr<CycleResult> cycle = run_once(data_root, ctx, job, once);
            if (!cycle.ok()) {
                // Prior filter emptied the expansion → soft stop (search-loop.md).
                if (is_no_candidates_status(cycle.status())) {
                    result.iterations_ = i + 1;
                    result.stop_reason_ = std::string(stop_no_new_candidates);
                    return result;
                }
                return cycle.status();
            }

            result.iterations_ = i + 1;
            result.hypotheses_written_ += cycle.value().hypotheses_written();
            for (const BatchSummary& b : cycle.value().batches()) {
                result.batches_.push_back(b);
            }

            if (cycle.value().stop_reason() == stop_no_new_candidates) {
                result.stop_reason_ = std::string(stop_no_new_candidates);
                return result;
            }

            if (options.stop_on_promoted) {
                StatusOr<bool> promoted =
                    workspace_has_status(data_root, job.workspace_id(), HypothesisStatus::Promoted);
                if (!promoted.ok()) {
                    return promoted.status();
                }
                if (promoted.value()) {
                    result.stop_reason_ = std::string(stop_success_promoted);
                    return result;
                }
            }

            if (options.stop_on_validate && options.validate_ok != nullptr &&
                *options.validate_ok) {
                result.stop_reason_ = std::string(stop_success_validate);
                return result;
            }

            if (wall_exceeded()) {
                result.stop_reason_ = std::string(stop_wall_budget);
                return result;
            }
        }

        result.stop_reason_ = std::string(stop_completed_iterations);
        return result;
    }

    /// Deterministic batch id: `b` + 32 hex of SHA-256(preimage).
    [[nodiscard]] static std::string make_batch_id(std::string_view workspace_id,
                                                   std::string_view family,
                                                   std::string_view job_digest_sha256,
                                                   std::string_view prior_digest_sha256,
                                                   std::string_view created_utc) {
        const std::string material = std::string(workspace_id) + '\n' + std::string(family) + '\n' +
                                     std::string(job_digest_sha256) + '\n' +
                                     std::string(prior_digest_sha256) + '\n' +
                                     std::string(created_utc);
        return std::string("b") + Sha256::hex_digest(material).substr(0, 32);
    }

    /// Advance an RFC 3339 UTC timestamp of the form `YYYY-MM-DDTHH:MM:SSZ` by `seconds`.
    [[nodiscard]] static StatusOr<std::string> advance_utc_seconds(std::string_view utc,
                                                                   std::int64_t seconds) {
        if (utc.size() != 20 || utc[10] != 'T' || utc[19] != 'Z') {
            return Status::error(
                "SearchScheduler::advance_utc_seconds expects YYYY-MM-DDTHH:MM:SSZ");
        }
        const int year = parse_digits(utc.substr(0, 4));
        const int month = parse_digits(utc.substr(5, 2));
        const int day = parse_digits(utc.substr(8, 2));
        const int hour = parse_digits(utc.substr(11, 2));
        const int minute = parse_digits(utc.substr(14, 2));
        const int second = parse_digits(utc.substr(17, 2));
        if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 ||
            second > 60) {
            return Status::error("SearchScheduler::advance_utc_seconds: invalid timestamp fields");
        }

        const std::chrono::year_month_day ymd{std::chrono::year{year},
                                              std::chrono::month{static_cast<unsigned>(month)},
                                              std::chrono::day{static_cast<unsigned>(day)}};
        if (!ymd.ok()) {
            return Status::error("SearchScheduler::advance_utc_seconds: invalid calendar day");
        }

        const std::chrono::sys_seconds tp =
            std::chrono::sys_days{ymd} + std::chrono::hours{hour} + std::chrono::minutes{minute} +
            std::chrono::seconds{second} + std::chrono::seconds{seconds};

        const std::chrono::sys_days day_floor = std::chrono::floor<std::chrono::days>(tp);
        const std::chrono::year_month_day out_ymd{day_floor};
        const auto tod = std::chrono::hh_mm_ss{tp - day_floor};

        char buf[32];
        std::snprintf(buf, sizeof(buf), "%04d-%02u-%02uT%02d:%02d:%02dZ",
                      static_cast<int>(out_ymd.year()), static_cast<unsigned>(out_ymd.month()),
                      static_cast<unsigned>(out_ymd.day()), static_cast<int>(tod.hours().count()),
                      static_cast<int>(tod.minutes().count()),
                      static_cast<int>(tod.seconds().count()));
        return std::string(buf);
    }

private:
    SearchScheduler() = delete;

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

    [[nodiscard]] static int parse_digits(std::string_view digits) {
        int v = 0;
        for (char ch : digits) {
            v = v * 10 + (ch - '0');
        }
        return v;
    }

    [[nodiscard]] static StatusOr<std::string> utc_for_iteration(const LoopOptions& options,
                                                                 std::size_t iteration) {
        if (!options.created_utcs.empty()) {
            return options.created_utcs[iteration];
        }
        if (iteration == 0) {
            return options.created_utc;
        }
        const std::int64_t delta =
            options.created_utc_step_seconds * static_cast<std::int64_t>(iteration);
        return advance_utc_seconds(options.created_utc, delta);
    }

    [[nodiscard]] static bool is_no_candidates_status(const Status& status) {
        const std::string& msg = status.message();
        return msg.find("no candidates after prior") != std::string::npos;
    }

    [[nodiscard]] static StatusOr<bool> workspace_has_status(const std::filesystem::path& data_root,
                                                             std::string_view workspace_id,
                                                             HypothesisStatus want) {
        StatusOr<std::vector<std::string>> ids =
            HypothesisRecord::list_ids(data_root, workspace_id);
        if (!ids.ok()) {
            return ids.status();
        }
        for (const std::string& hid : ids.value()) {
            StatusOr<HypothesisRecord> record =
                HypothesisRecord::load(data_root, workspace_id, hid);
            if (!record.ok()) {
                return record.status();
            }
            if (record.value().status() == want) {
                return true;
            }
        }
        return false;
    }

    /// Inline `SearchJob.prior` wins; otherwise rebuild from workspace hypotheses.
    [[nodiscard]] static StatusOr<SearchPrior> resolve_prior(const std::filesystem::path& data_root,
                                                             const SearchJob& job,
                                                             const Options& options) {
        if (job.prior().has_value() && !job.prior()->is_null()) {
            return SearchPrior::from_json(*job.prior());
        }
        return SearchPrior::from_workspace(data_root, job.workspace_id(), options.created_utc,
                                           options.prior_build);
    }

    [[nodiscard]] static StatusOr<CpuCandidateExport::Result>
    export_candidates(std::span<const Index29> cipher, const SearchJob& job, const Context& ctx,
                      const SearchPrior& prior,
                      BatchRunner::Progress progress = BatchRunner::Progress{}) {
        // Theory-URI / hill / autokey jobs are CPU-only (no fused CUDA χ² path yet).
        if (job.backend() == Backend::Cpu ||
            SearchJob::is_cpu_export_only_family(job.family())) {
            return CpuCandidateExport::from_job(cipher, job, ctx, &prior, progress);
        }

        // Fused CUDA export is χ²-only (GpuCandidateExport). Any other score_id
        // (e.g. log_bigram_gp_v0) falls back to CPU RankCandidates — search-loop.md.
        if (job.score_id() != GpuCandidateExport::score_id) {
            return CpuCandidateExport::from_job(cipher, job, ctx, &prior, progress);
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
            export_cuda_fused(cipher, job, freqs.value(), progress);
        if (!fused.ok()) {
            return fused.status();
        }

        // Apply prior exclusions/seeds on the CPU path for parity with CpuCandidateExport.
        // Fused GPU export does not yet fold SearchPrior; re-run CPU with prior when
        // the prior is non-empty so exclusions/seeds match the CPU oracle.
        if (!prior.seeds().empty() || !prior.exclusions().empty()) {
            return CpuCandidateExport::from_job(cipher, job, ctx, &prior, progress);
        }
        return fused;
    }

    [[nodiscard]] static StatusOr<CpuCandidateExport::Result>
    export_cuda_fused(std::span<const Index29> cipher, const SearchJob& job,
                      const ExpectedFrequencyTable& freqs,
                      BatchRunner::Progress progress = BatchRunner::Progress{}) {
        const std::string& family = job.family();
        if (family == "caesar") {
            return GpuCandidateExport::caesar(cipher, freqs, job.k(), job.direction(), progress);
        }
        if (family == "atbash") {
            return GpuCandidateExport::atbash(cipher, freqs, job.k(), job.direction(), progress);
        }
        if (family == "atbash_caesar") {
            return GpuCandidateExport::atbash_caesar(cipher, freqs, job.k(), job.direction(),
                                                     progress);
        }
        if (family == "compose") {
            return GpuCandidateExport::compose_from_param_grid(cipher, freqs, job.param_grid(),
                                                               job.k(), job.direction(), progress);
        }
        if (family == "affine") {
            return GpuCandidateExport::affine(cipher, freqs, job.k(), job.direction(), progress);
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
            return GpuCandidateExport::vigenere_bounded(cipher, freqs, job.k(), max_len,
                                                        job.direction(), progress);
        }
        if (family == "beaufort") {
            if (!job.allow_extended_families()) {
                return Status::error("SearchScheduler: beaufort requires allow_extended_families");
            }
            std::size_t max_len = GpuCandidateExport::default_vigenere_max_key_length;
            if (job.param_grid().is_object() && job.param_grid().contains("max_key_length") &&
                job.param_grid().at("max_key_length").is_number_integer()) {
                const std::int64_t v = job.param_grid().at("max_key_length").get<std::int64_t>();
                if (v < 1) {
                    return Status::error("SearchScheduler: param_grid.max_key_length must be >= 1");
                }
                max_len = static_cast<std::size_t>(v);
            }
            return GpuCandidateExport::beaufort_bounded(cipher, freqs, job.k(), max_len,
                                                        job.direction(), progress);
        }
        if (family == "totient") {
            if (!job.allow_extended_families()) {
                return Status::error("SearchScheduler: totient requires allow_extended_families");
            }
            std::size_t count = GpuCandidateExport::default_totient_start_count;
            if (job.param_grid().is_object() && job.param_grid().contains("prime_start_count") &&
                job.param_grid().at("prime_start_count").is_number_integer()) {
                const std::int64_t v = job.param_grid().at("prime_start_count").get<std::int64_t>();
                if (v < 1) {
                    return Status::error(
                        "SearchScheduler: param_grid.prime_start_count must be >= 1");
                }
                count = static_cast<std::size_t>(v);
            }
            if (job.param_grid().is_object() && job.param_grid().contains("prime_start_indices") &&
                job.param_grid().at("prime_start_indices").is_array()) {
                std::vector<std::size_t> starts;
                for (const auto& item : job.param_grid().at("prime_start_indices")) {
                    if (!item.is_number_integer() || item.get<std::int64_t>() < 0) {
                        return Status::error(
                            "SearchScheduler: prime_start_indices must be non-negative integers");
                    }
                    starts.push_back(static_cast<std::size_t>(item.get<std::int64_t>()));
                }
                return GpuCandidateExport::totient(cipher, freqs, starts, job.k(), job.direction(),
                                                   progress);
            }
            return GpuCandidateExport::totient_bounded(cipher, freqs, job.k(), count,
                                                       job.direction(), progress);
        }
        return Status::error("SearchScheduler: unsupported family for cuda export: " + family);
    }
};
#endif // SEARCH_SCHEDULER_HPP
