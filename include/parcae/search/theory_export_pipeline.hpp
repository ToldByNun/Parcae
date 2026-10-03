#ifndef THEORY_EXPORT_PIPELINE_HPP
#define THEORY_EXPORT_PIPELINE_HPP

#include "parcae/batch/batch_runner.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/search/gpu_candidate_export.hpp"
#include "parcae/search/theory_export_cache.hpp"

#include <cstddef>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#if defined(PARCAE_HAS_CUDA)
#include "cuda_stream_pair.hpp"
#include "theory_device_scratch.hpp"
#endif

/// Multi-chunk theory fused export with param-slab ping-pong overlap.
///
/// Per chunk: stage H2D into the write slab (overlaps prior hist), collect the
/// previous in-flight scores, then launch hist on the staged slab. Scores are
/// delivered in submission order. Requires CUDA. No C++ namespaces.
class TheoryExportPipeline {
public:
#if !defined(PARCAE_HAS_CUDA)
    TheoryExportPipeline() = delete;
#else
    TheoryExportPipeline(TheoryExportCache& cache, TheoryDeviceScratch& scratch,
                         CudaStreamPair& streams)
        : cache_(&cache), scratch_(&scratch), streams_(&streams) {}

    TheoryExportPipeline(const TheoryExportPipeline&) = delete;
    TheoryExportPipeline& operator=(const TheoryExportPipeline&) = delete;

    /// Stage H2D for this chunk (overlaps prior hist), collect prior scores if
    /// any, then launch hist. Returns prior chunk scores (`nullopt` on first).
    [[nodiscard]] StatusOr<std::optional<std::vector<double>>> submit(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const std::filesystem::path& theories_root, std::string_view theory_uri_text,
        const std::vector<nlohmann::json>& params_list,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        StatusOr<GpuCandidateExport::TheoryLaunchTicket> staged =
            GpuCandidateExport::theory_stage_scores_only(cipher, freqs, theories_root,
                                                         theory_uri_text, params_list, *cache_,
                                                         *scratch_, *streams_, progress);
        if (!staged.ok()) {
            return staged.status();
        }

        std::optional<std::vector<double>> prior;
        if (in_flight_.in_flight()) {
            StatusOr<std::vector<double>> collected = GpuCandidateExport::theory_collect_scores_only(
                *scratch_, *streams_, in_flight_, progress);
            if (!collected.ok()) {
                return collected.status();
            }
            prior = std::move(collected.value());
        }

        Status launched = GpuCandidateExport::theory_launch_staged(*cache_, *scratch_, *streams_,
                                                                   staged.value());
        if (!launched.ok()) {
            return launched;
        }
        in_flight_ = std::move(staged.value());
        return prior;
    }

    /// Collect scores for the last in-flight launch (empty if none).
    [[nodiscard]] StatusOr<std::vector<double>>
    flush(BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (!in_flight_.in_flight()) {
            return std::vector<double>{};
        }
        return GpuCandidateExport::theory_collect_scores_only(*scratch_, *streams_, in_flight_,
                                                              progress);
    }

    /// Submit all chunks then flush; one score vector per chunk (in order).
    [[nodiscard]] StatusOr<std::vector<std::vector<double>>> run_chunks(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const std::filesystem::path& theories_root, std::string_view theory_uri_text,
        const std::vector<std::vector<nlohmann::json>>& chunks,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        std::vector<std::vector<double>> out;
        out.reserve(chunks.size());
        for (const std::vector<nlohmann::json>& chunk : chunks) {
            StatusOr<std::optional<std::vector<double>>> prior =
                submit(cipher, freqs, theories_root, theory_uri_text, chunk, progress);
            if (!prior.ok()) {
                return prior.status();
            }
            if (prior.value().has_value()) {
                out.push_back(std::move(*prior.value()));
            }
        }
        StatusOr<std::vector<double>> last = flush(progress);
        if (!last.ok()) {
            return last.status();
        }
        if (!last.value().empty()) {
            out.push_back(std::move(last.value()));
        }
        if (out.size() != chunks.size()) {
            return Status::error("TheoryExportPipeline::run_chunks size mismatch");
        }
        return out;
    }

    [[nodiscard]] bool has_in_flight() const noexcept { return in_flight_.in_flight(); }

private:
    TheoryExportCache* cache_ = nullptr;
    TheoryDeviceScratch* scratch_ = nullptr;
    CudaStreamPair* streams_ = nullptr;
    GpuCandidateExport::TheoryLaunchTicket in_flight_{};
#endif
};

#endif // THEORY_EXPORT_PIPELINE_HPP
