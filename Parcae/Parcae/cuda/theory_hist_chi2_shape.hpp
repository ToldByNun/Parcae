#ifndef THEORY_HIST_CHI2_SHAPE_HPP
#define THEORY_HIST_CHI2_SHAPE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// ShapeInline fused χ² hist twins for algebraic ShapeIds
/// (`docs/architecture/dsl-smart-hist.md`).
///
/// Atbash uses `HistFast::dec_atbash` (same decode as catalog `FamilyChi2Batch`
/// atbash). Param-free: every candidate row histograms the same decrypt.
/// Finalize via `Chi2BatchScore`. No C++ namespaces.
class TheoryHistChi2Shape {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Atbash shape twin: `out = 28 - x` per token (uchar4 + fat-tile).
    [[nodiscard]] static Status
    launch_atbash_async(const std::uint8_t* device_in, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count,
                        cudaStream_t stream = nullptr);

private:
    TheoryHistChi2Shape() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // THEORY_HIST_CHI2_SHAPE_HPP
