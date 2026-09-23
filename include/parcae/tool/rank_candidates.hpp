#ifndef RANK_CANDIDATES_HPP
#define RANK_CANDIDATES_HPP

#include "parcae/batch/batch_execution.hpp"
#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/batch/batch_runner.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"

#if defined(PARCAE_HAS_CUDA)
#include "cuda_score.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Tool-facing `rank_candidates` primitive.
///
/// Scores a candidate list and returns best-first top-k with **stable ties**
/// (`BatchOrdering`: score → `candidate_id` → `source_index`).
/// `backend=cuda` scores already-materialized `output_indices` via `CudaScore`
/// then the same host `BatchOrdering` reduction (CPU remains the small-N oracle).
class RankCandidates {
public:
    /// Rank already-generated candidates. When `ctx` is non-null and `score_id`
    /// is `chi2_english_gp_v0` without `request.expected_frequencies`, loads the
    /// expected-frequency table from the data root (same as `tool::score`).
    /// `execution` applies only to `backend=cpu`. CUDA scores lanes serially via
    /// `CudaScore` then sorts once on the host.
    [[nodiscard]] static StatusOr<BatchResult> run(
        std::span<const TransformCandidate> candidates,
        std::string_view score_id,
        std::size_t k,
        const parcae::tool::Context* ctx = nullptr,
        ScoreRequest request = ScoreRequest(),
        const nlohmann::json& params = nlohmann::json::object(),
        std::string_view score_version = "v0",
        BatchExecution execution = BatchExecution::Serial,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        if (candidates.empty()) {
            return Status::error("rank_candidates: candidates must be non-empty");
        }
        if (k == 0) {
            return Status::error("rank_candidates: k must be >= 1");
        }

        Status usable = parcae::tool::BackendUtil::ensure_usable(backend);
        if (!usable.ok()) {
            return usable;
        }

        std::optional<ExpectedFrequencyTable> owned_table;
        if (score_id == ScoreId::chi2_english_gp_v0().str() &&
            request.expected_frequencies == nullptr) {
            if (ctx == nullptr) {
                return Status::error(
                    "rank_candidates: chi2_english_gp_v0 requires Context or "
                    "ScoreRequest.expected_frequencies");
            }
            StatusOr<ExpectedFrequencyTable> table = ctx->load_english_gp_expected();
            if (!table.ok()) {
                return table.status();
            }
            owned_table = std::move(table.value());
            request.expected_frequencies = &owned_table.value();
        }

        if (backend == parcae::tool::Backend::Cpu) {
            return BatchRunner::run(
                candidates, score_id, k, request, execution, score_version, params);
        }

        return run_cuda(candidates, score_id, k, request, params, score_version);
    }

    /// JSON for one hit; optionally attach the source candidate envelope + latin.
    [[nodiscard]] static nlohmann::json hit_to_json(
        const BatchHit& hit,
        std::size_t rank,
        const TransformCandidate* candidate = nullptr,
        std::optional<std::string> latin_preview = std::nullopt) {
        nlohmann::json row{
            {"rank", rank},
            {"candidate_id", hit.candidate_id()},
            {"score", hit.score()},
            {"source_index", hit.source_index()},
        };
        if (candidate != nullptr) {
            row["envelope"] = candidate->envelope();
        } else {
            row["envelope"] = nullptr;
        }
        if (latin_preview.has_value()) {
            row["latin"] = std::move(*latin_preview);
        } else {
            row["latin"] = nullptr;
        }
        return row;
    }

    /// Full ranking payload for agent CLIs (`parcae.tool_response.v0` result).
    /// When `candidates` aligns with `source_index`, envelopes are included.
    /// When `ctx` is set, also attaches a Latin preview per hit.
    [[nodiscard]] static StatusOr<nlohmann::json> result_to_json(
        const BatchResult& result,
        std::span<const TransformCandidate> candidates = {},
        const parcae::tool::Context* ctx = nullptr,
        std::size_t latin_max_chars = 64,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        StatusOr<ScoreOrder> order = ScoreRegistry::order_of(result.score_id());
        if (!order.ok()) {
            return order.status();
        }

        nlohmann::json hits = nlohmann::json::array();
        for (std::size_t rank = 0; rank < result.top().size(); ++rank) {
            const BatchHit& hit = result.top()[rank];
            const TransformCandidate* candidate = nullptr;
            std::optional<std::string> latin;
            if (hit.source_index() < candidates.size()) {
                candidate = &candidates[hit.source_index()];
                if (ctx != nullptr) {
                    StatusOr<std::string> text =
                        parcae::tool::to_latin(*ctx, candidate->output_indices());
                    if (!text.ok()) {
                        return text.status();
                    }
                    if (text.value().size() > latin_max_chars) {
                        latin = text.value().substr(0, latin_max_chars);
                    } else {
                        latin = std::move(text.value());
                    }
                }
            }
            hits.push_back(hit_to_json(hit, rank, candidate, std::move(latin)));
        }

        return nlohmann::json{
            {"score_id", result.score_id()},
            {"score_version", result.score_version()},
            {"order", std::string(ScoreOrderUtil::to_string(order.value()))},
            {"backend", std::string(parcae::tool::BackendUtil::to_string(backend))},
            {"k", result.top().size()},
            {"scored_count", result.scored_count()},
            {"hits", std::move(hits)},
        };
    }

private:
    RankCandidates() = delete;

    [[nodiscard]] static StatusOr<BatchResult> run_cuda(
        std::span<const TransformCandidate> candidates,
        std::string_view score_id,
        std::size_t k,
        const ScoreRequest& request,
        const nlohmann::json& params,
        std::string_view score_version) {
#if defined(PARCAE_HAS_CUDA)
        if (!CudaScore::available()) {
            return Status::error("rank_candidates: CUDA backend requested but CUDA is not available");
        }

        StatusOr<ScoreOrder> order = ScoreRegistry::order_of(score_id);
        if (!order.ok()) {
            return order.status();
        }

        std::vector<BatchHit> hits;
        hits.reserve(candidates.size());
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            StatusOr<double> value = CudaScore::score(
                score_id,
                candidates[i].output_indices(),
                score_version,
                params,
                request);
            if (!value.ok()) {
                return value.status();
            }
            hits.emplace_back(candidates[i].candidate_id(), value.value(), i);
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
#else
        (void)candidates;
        (void)score_id;
        (void)k;
        (void)request;
        (void)params;
        (void)score_version;
        return Status::error(
            "rank_candidates: CUDA backend requested but Parcae was built without "
            "CUDA (PARCAE_BUILD_CUDA)");
#endif
    }
};

#endif  // RANK_CANDIDATES_HPP
