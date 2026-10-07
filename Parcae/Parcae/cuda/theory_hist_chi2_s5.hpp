#ifndef THEORY_HIST_CHI2_S5_HPP
#define THEORY_HIST_CHI2_S5_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// S5 quadratic poly fused χ² hist (bitmask_blend / poly2 class):
/// `out = x ± (b0 + b1·(t mod 29) + b2·(t mod 29)²)` (mod 29).
///
/// Period-29 keystream table in shared memory (same traffic class as S2).
/// `grid.y` via `HistTileCap::kS5` (production default 32). Soft S0 on bind
/// fail stays in emit.
class TheoryHistChi2S5 {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    [[nodiscard]] static Status launch_poly_async(
        const std::uint8_t* device_in, const std::uint8_t* device_b0,
        const std::uint8_t* device_b1, const std::uint8_t* device_b2,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr);

    [[nodiscard]] static int tiles_for_public(std::size_t token_count);

private:
    TheoryHistChi2S5() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // THEORY_HIST_CHI2_S5_HPP
