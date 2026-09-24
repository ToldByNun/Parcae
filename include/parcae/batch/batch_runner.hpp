#ifndef BATCH_RUNNER_HPP
#define BATCH_RUNNER_HPP

#include "parcae/batch/batch_execution.hpp"
#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/cli/console_progress_clock.hpp"
#include "parcae/cli/console_progress_sink.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

/// Bounded batch / top-k over already-applied `TransformCandidate`s.
///
/// Single-thread (`BatchExecution::Serial`) is the parity source of truth.
/// `BatchExecution::Parallel` only parallelizes **score evaluation**; the
/// final top-k list is always reduced with `BatchOrdering` (documented order).
///
/// Optional `Progress` is observe-only: when `sink` is null, scoring is bit-identical
/// to the pre-progress path. Digests / ranking MUST NOT depend on the sink.
class BatchRunner {
public:
    /// Optional live-progress hook for human dashboards.
    class Progress {
    public:
        // Explicit ctor so Progress{} is usable as a default arg of BatchRunner
        // methods on GCC/Clang (nested default-member-initializers are not).
        Progress() noexcept : sink(nullptr), rune_count(0) {}

        ConsoleProgressSink* sink;
        /// Cipher length in consumable runes (for runes/s ≈ done * rune_count / t).
        std::size_t rune_count;
    };

    /// Score every candidate with `score_id`, keep the best `k` (best-first).
    /// `k == 0` is an error. If `k >= N`, returns all candidates sorted.
    [[nodiscard]] static StatusOr<BatchResult>
    run(std::span<const TransformCandidate> candidates, std::string_view score_id, std::size_t k,
        const ScoreRequest& request = ScoreRequest(),
        BatchExecution execution = BatchExecution::Serial, std::string_view score_version = "v0",
        const nlohmann::json& params = nlohmann::json::object(), Progress progress = Progress{}) {
        if (k == 0) {
            return Status::error("BatchRunner requires k >= 1");
        }

        StatusOr<ScoreOrder> order = ScoreRegistry::order_of(score_id);
        if (!order.ok()) {
            return order.status();
        }

        StatusOr<std::vector<double>> scores =
            compute_scores(candidates, score_id, score_version, params, request, execution,
                           order.value(), progress);
        if (!scores.ok()) {
            return scores.status();
        }

        std::vector<BatchHit> hits;
        hits.reserve(candidates.size());
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            hits.emplace_back(candidates[i].candidate_id(), scores.value()[i], i);
        }

        std::sort(hits.begin(), hits.end(), BatchOrdering::BestFirst{order.value()});

        if (hits.size() > k) {
            hits.erase(hits.begin() + static_cast<std::ptrdiff_t>(k), hits.end());
        }

        return BatchResult{
            std::move(hits),
            candidates.size(),
            std::string(score_id),
            std::string(score_version),
        };
    }

private:
    BatchRunner() = delete;

    class BestTracker {
    public:
        explicit BestTracker(ScoreOrder order) noexcept : order_(order) {}

        void consider(double score, std::string_view label) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!best_score_.has_value() || is_better(score, best_score_.value(), order_)) {
                best_score_ = score;
                best_label_ = std::string(label);
            }
        }

        void apply_to(ConsoleProgressSnapshot& snap) const {
            std::lock_guard<std::mutex> lock(mutex_);
            if (best_score_.has_value()) {
                snap.set_best_score(best_score_);
                snap.set_best_label(best_label_);
            }
        }

    private:
        [[nodiscard]] static bool is_better(double a, double b, ScoreOrder order) noexcept {
            if (order == ScoreOrder::Asc) {
                return a < b;
            }
            return a > b;
        }

        ScoreOrder order_;
        mutable std::mutex mutex_;
        std::optional<double> best_score_;
        std::string best_label_;
    };

    [[nodiscard]] static ConsoleProgressSnapshot make_base_snapshot(std::size_t total,
                                                                    std::size_t rune_count) {
        ConsoleProgressSnapshot snap;
        snap.set_stage("score");
        snap.set_candidates_total(total);
        snap.set_candidates_done(0);
        snap.set_rune_count(rune_count);
        return snap;
    }

    static void emit_progress(ConsoleProgressSink* sink, ConsoleProgressSnapshot snap,
                              std::size_t done, const ConsoleProgressClock& clock,
                              const BestTracker& best) {
        if (sink == nullptr) {
            return;
        }
        snap.set_candidates_done(done);
        snap.set_elapsed_seconds(clock.elapsed_seconds());
        snap.refresh_rates();
        best.apply_to(snap);
        sink->on_progress(snap);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    compute_scores(std::span<const TransformCandidate> candidates, std::string_view score_id,
                   std::string_view score_version, const nlohmann::json& params,
                   const ScoreRequest& request, BatchExecution execution, ScoreOrder order,
                   Progress progress) {
        if (execution == BatchExecution::Parallel && candidates.size() >= 2) {
            return compute_scores_parallel(candidates, score_id, score_version, params, request,
                                           order, progress);
        }
        return compute_scores_serial(candidates, score_id, score_version, params, request, order,
                                     progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    compute_scores_serial(std::span<const TransformCandidate> candidates, std::string_view score_id,
                          std::string_view score_version, const nlohmann::json& params,
                          const ScoreRequest& request, ScoreOrder order, Progress progress) {
        ConsoleProgressClock clock;
        BestTracker best(order);
        const ConsoleProgressSnapshot base =
            make_base_snapshot(candidates.size(), progress.rune_count);

        if (progress.sink != nullptr) {
            progress.sink->on_stage("score", base);
        }

        std::vector<double> scores;
        scores.reserve(candidates.size());
        for (const TransformCandidate& candidate : candidates) {
            StatusOr<double> value = ScoreRegistry::score(score_id, candidate.output_indices(),
                                                          score_version, params, request);
            if (!value.ok()) {
                return value.status();
            }
            scores.push_back(value.value());
            best.consider(value.value(), candidate.candidate_id());
            emit_progress(progress.sink, base, scores.size(), clock, best);
        }
        return scores;
    }

    /// Chunked `std::async` pool. Workers write only to disjoint `scores[i]` /
    /// `ok[i]` slots; no shared ranking structures. Worker count is
    /// `min(hw_concurrency, N)` with a floor of 1.
    ///
    /// Progress: workers may call `sink->on_progress` (sink MUST be thread-safe,
    /// e.g. `ConsoleDashboard`). Ranking / scores are unchanged by the sink.
    [[nodiscard]] static StatusOr<std::vector<double>>
    compute_scores_parallel(std::span<const TransformCandidate> candidates,
                            std::string_view score_id, std::string_view score_version,
                            const nlohmann::json& params, const ScoreRequest& request,
                            ScoreOrder order, Progress progress) {
        const std::size_t n = candidates.size();
        std::vector<double> scores(n, 0.0);
        std::vector<char> ok(n, 0);
        std::vector<Status> errors;
        errors.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            errors.push_back(Status::error("batch score slot unscored"));
        }

        ConsoleProgressClock clock;
        BestTracker best(order);
        const ConsoleProgressSnapshot base = make_base_snapshot(n, progress.rune_count);
        std::atomic<std::size_t> scored{0};

        if (progress.sink != nullptr) {
            progress.sink->on_stage("score", base);
        }

        const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t workers = std::min(hw, n);
        const std::size_t chunk = (n + workers - 1) / workers;

        const std::string score_id_owned(score_id);
        const std::string score_version_owned(score_version);
        ConsoleProgressSink* sink = progress.sink;

        std::vector<std::future<void>> futures;
        futures.reserve(workers);

        for (std::size_t w = 0; w < workers; ++w) {
            const std::size_t begin = w * chunk;
            if (begin >= n) {
                break;
            }
            const std::size_t end = std::min(begin + chunk, n);

            futures.push_back(
                std::async(std::launch::async, [&candidates, &scores, &ok, &errors, &params,
                                                &request, &best, &clock, &scored, &base, sink,
                                                score_id_owned, score_version_owned, begin, end]() {
                    for (std::size_t i = begin; i < end; ++i) {
                        StatusOr<double> value =
                            ScoreRegistry::score(score_id_owned, candidates[i].output_indices(),
                                                 score_version_owned, params, request);
                        if (!value.ok()) {
                            errors[i] = value.status();
                            ok[i] = 0;
                            continue;
                        }
                        scores[i] = value.value();
                        ok[i] = 1;
                        best.consider(value.value(), candidates[i].candidate_id());
                        const std::size_t done = scored.fetch_add(1, std::memory_order_relaxed) + 1;
                        emit_progress(sink, base, done, clock, best);
                    }
                }));
        }

        for (std::future<void>& future : futures) {
            future.get();
        }

        for (std::size_t i = 0; i < n; ++i) {
            if (ok[i] == 0) {
                return errors[i];
            }
        }
        return scores;
    }
};

#endif // BATCH_RUNNER_HPP
