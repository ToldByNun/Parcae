#ifndef BATCH_RUNNER_HPP
#define BATCH_RUNNER_HPP

#include "parcae/batch/batch_execution.hpp"
#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"

#include <algorithm>
#include <cstddef>
#include <future>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Bounded batch / top-k over already-applied `TransformCandidate`s.
///
/// Single-thread (`BatchExecution::Serial`) is the parity source of truth.
/// `BatchExecution::Parallel` only parallelizes **score evaluation**; the
/// final top-k list is always reduced with `BatchOrdering` (documented order).
class BatchRunner {
public:
    /// Score every candidate with `score_id`, keep the best `k` (best-first).
    /// `k == 0` is an error. If `k >= N`, returns all candidates sorted.
    [[nodiscard]] static StatusOr<BatchResult> run(
        std::span<const TransformCandidate> candidates,
        std::string_view score_id,
        std::size_t k,
        const ScoreRequest& request = {},
        BatchExecution execution = BatchExecution::Serial,
        std::string_view score_version = "v0",
        const nlohmann::json& params = nlohmann::json::object()) {
        if (k == 0) {
            return Status::error("BatchRunner requires k >= 1");
        }

        StatusOr<ScoreOrder> order = ScoreRegistry::order_of(score_id);
        if (!order.ok()) {
            return order.status();
        }

        StatusOr<std::vector<double>> scores =
            compute_scores(candidates, score_id, score_version, params, request, execution);
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

    [[nodiscard]] static StatusOr<std::vector<double>> compute_scores(
        std::span<const TransformCandidate> candidates,
        std::string_view score_id,
        std::string_view score_version,
        const nlohmann::json& params,
        const ScoreRequest& request,
        BatchExecution execution) {
        if (execution == BatchExecution::Parallel && candidates.size() >= 2) {
            return compute_scores_parallel(
                candidates, score_id, score_version, params, request);
        }
        return compute_scores_serial(candidates, score_id, score_version, params, request);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> compute_scores_serial(
        std::span<const TransformCandidate> candidates,
        std::string_view score_id,
        std::string_view score_version,
        const nlohmann::json& params,
        const ScoreRequest& request) {
        std::vector<double> scores;
        scores.reserve(candidates.size());
        for (const TransformCandidate& candidate : candidates) {
            StatusOr<double> value = ScoreRegistry::score(
                score_id,
                candidate.output_indices(),
                score_version,
                params,
                request);
            if (!value.ok()) {
                return value.status();
            }
            scores.push_back(value.value());
        }
        return scores;
    }

    /// Chunked `std::async` pool. Workers write only to disjoint `scores[i]` /
    /// `ok[i]` slots; no shared ranking structures. Worker count is
    /// `min(hw_concurrency, N)` with a floor of 1.
    [[nodiscard]] static StatusOr<std::vector<double>> compute_scores_parallel(
        std::span<const TransformCandidate> candidates,
        std::string_view score_id,
        std::string_view score_version,
        const nlohmann::json& params,
        const ScoreRequest& request) {
        const std::size_t n = candidates.size();
        std::vector<double> scores(n, 0.0);
        std::vector<char> ok(n, 0);
        std::vector<Status> errors;
        errors.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            errors.push_back(Status::error("batch score slot unscored"));
        }

        const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t workers = std::min(hw, n);
        const std::size_t chunk = (n + workers - 1) / workers;

        const std::string score_id_owned(score_id);
        const std::string score_version_owned(score_version);

        std::vector<std::future<void>> futures;
        futures.reserve(workers);

        for (std::size_t w = 0; w < workers; ++w) {
            const std::size_t begin = w * chunk;
            if (begin >= n) {
                break;
            }
            const std::size_t end = std::min(begin + chunk, n);

            futures.push_back(std::async(
                std::launch::async,
                [&candidates,
                 &scores,
                 &ok,
                 &errors,
                 &params,
                 &request,
                 score_id_owned,
                 score_version_owned,
                 begin,
                 end]() {
                    for (std::size_t i = begin; i < end; ++i) {
                        StatusOr<double> value = ScoreRegistry::score(
                            score_id_owned,
                            candidates[i].output_indices(),
                            score_version_owned,
                            params,
                            request);
                        if (!value.ok()) {
                            errors[i] = value.status();
                            ok[i] = 0;
                            continue;
                        }
                        scores[i] = value.value();
                        ok[i] = 1;
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
