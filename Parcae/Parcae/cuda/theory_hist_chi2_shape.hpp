#ifndef THEORY_HIST_CHI2_SHAPE_HPP
#define THEORY_HIST_CHI2_SHAPE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// ShapeInline fused χ² hist twins for algebraic ShapeIds
/// (`docs/architecture/dsl-smart-hist.md`).
///
/// - Atbash: `HistFast::dec_atbash` (param-free)
/// - Caesar: `HistFast::dec_caesar` with per-candidate shifts
/// - Affine decrypt: `inv(a)·(x−b)` via `Z29Device` (host must reject `a==0`)
///
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

    /// Caesar shape twin. `device_shifts` length `candidate_count`.
    [[nodiscard]] static Status
    launch_caesar_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                        const double* device_probabilities, std::uint32_t* device_counts,
                        double* device_scores, std::size_t candidate_count,
                        std::size_t token_count, cudaStream_t stream = nullptr);

    /// Affine decrypt twin: LUT `inv(a)·(x−b)`. Caller must ensure every `a` is
    /// invertible (≠0 in ℤ₂₉); otherwise soft-fallback to S0 before launch.
    [[nodiscard]] static Status
    launch_affine_async(const std::uint8_t* device_in, const std::uint8_t* device_a,
                        const std::uint8_t* device_b, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count,
                        cudaStream_t stream = nullptr);

private:
    TheoryHistChi2Shape() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // THEORY_HIST_CHI2_SHAPE_HPP
