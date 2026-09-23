#ifndef CUDA_BATCH_SCORE_HPP
#define CUDA_BATCH_SCORE_HPP

#include "candidate_batch_buffers.hpp"
#include "cuda_score.hpp"

#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// CUDA batch scoring + host top-k (`docs/architecture/cuda-abi.md` / score-reduction).
///
/// v0: score each `out_index29` lane via `CudaScore` (device twins), write
/// `scores[C]`, then single-thread `BatchOrdering` on the host after D2H.
/// Device-side top-k heaps are out of scope for v0.
class CudaBatchScore {
public:
    [[nodiscard]] static bool available() noexcept {
        return CudaScore::available();
    }

    /// Score every candidate lane in `buffers.out_index29` into `buffers.scores()`.
    /// Requires `AllocateOptions::with_scores` and filled outputs (e.g. after a
    /// batch apply kernel).
    [[nodiscard]] static Status score_lanes(
        CandidateBatchBuffers& buffers,
        std::string_view score_id,
        const ScoreRequest& request = ScoreRequest(),
        std::string_view score_version = "v0",
        const nlohmann::json& params = nlohmann::json::object()) {
        if (buffers.scores().empty()) {
            return Status::error("CudaBatchScore::score_lanes requires scores[C] buffer");
        }
        if (buffers.scores().size() != buffers.candidate_count()) {
            return Status::error("CudaBatchScore::score_lanes scores size must equal C");
        }
        if (buffers.out_index29().size() !=
            buffers.candidate_count() * buffers.token_count()) {
            return Status::error("CudaBatchScore::score_lanes out_index29 size mismatch");
        }

        std::vector<Index29> lane(buffers.token_count());
        for (std::size_t c = 0; c < buffers.candidate_count(); ++c) {
            Status copied = buffers.copy_out_candidate(c, lane);
            if (!copied.ok()) {
                return copied;
            }
            StatusOr<double> scored =
                CudaScore::score(score_id, lane, score_version, params, request);
            if (!scored.ok()) {
                return scored.status();
            }
            buffers.scores()[c] = scored.value();
        }
        return Status::success();
    }

    /// Deterministic top-k from already-materialized `scores[C]` (host reduction).
    [[nodiscard]] static StatusOr<BatchResult> top_k(
        std::span<const std::string> candidate_ids,
        std::span<const double> scores,
        std::string_view score_id,
        std::size_t k,
        std::string_view score_version = "v0") {
        if (k == 0) {
            return Status::error("CudaBatchScore::top_k requires k >= 1");
        }
        if (candidate_ids.size() != scores.size()) {
            return Status::error("CudaBatchScore::top_k candidate_ids/scores size mismatch");
        }
        if (score_version != "v0") {
            return Status::error("Unsupported score_version (only v0 is registered)");
        }

        StatusOr<ScoreOrder> order = ScoreRegistry::order_of(score_id);
        if (!order.ok()) {
            return order.status();
        }

        std::vector<BatchHit> hits;
        hits.reserve(scores.size());
        for (std::size_t i = 0; i < scores.size(); ++i) {
            hits.emplace_back(std::string(candidate_ids[i]), scores[i], i);
        }

        std::sort(hits.begin(), hits.end(), BatchOrdering::BestFirst{order.value()});

        if (hits.size() > k) {
            hits.erase(hits.begin() + static_cast<std::ptrdiff_t>(k), hits.end());
        }

        return BatchResult{
            std::move(hits),
            scores.size(),
            std::string(score_id),
            std::string(score_version),
        };
    }

    /// `score_lanes` then `top_k` using `buffers.scores()` and `candidate_ids`.
    [[nodiscard]] static StatusOr<BatchResult> score_and_top_k(
        CandidateBatchBuffers& buffers,
        std::span<const std::string> candidate_ids,
        std::string_view score_id,
        std::size_t k,
        const ScoreRequest& request = ScoreRequest(),
        std::string_view score_version = "v0",
        const nlohmann::json& params = nlohmann::json::object()) {
        if (candidate_ids.size() != buffers.candidate_count()) {
            return Status::error("CudaBatchScore::score_and_top_k candidate_ids size must equal C");
        }
        Status scored = score_lanes(buffers, score_id, request, score_version, params);
        if (!scored.ok()) {
            return scored;
        }
        return top_k(candidate_ids, buffers.scores(), score_id, k, score_version);
    }

private:
    CudaBatchScore() = delete;
};

#endif // CUDA_BATCH_SCORE_HPP
