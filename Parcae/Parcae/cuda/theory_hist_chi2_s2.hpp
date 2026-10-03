#ifndef THEORY_HIST_CHI2_S2_HPP
#define THEORY_HIST_CHI2_S2_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// S2 uchar4 fused χ² hist for bitmask_blend / progressive shapes:
/// `out = x - (b0 + b1·(t mod 29))` or `out = x + (b0 + b1·(t mod 29))` (mod 29).
///
/// No interpreter; params are per-candidate `b0`/`b1` rows. Per-block shared
/// period-29 keystream (`ks[i]=b0+b1·i`) then fat-tile hist. Finalize via
/// `Chi2BatchScore`.
class TheoryHistChi2S2 {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// `cipher_minus_ks == true` → `HistFast::dec_sub(x, ks)`; else `enc_caesar(x, ks)`
    /// (add). `device_b0` / `device_b1` length `candidate_count`.
    [[nodiscard]] static Status launch_linear_async(
        const std::uint8_t* device_in, const std::uint8_t* device_b0,
        const std::uint8_t* device_b1, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, bool cipher_minus_ks, cudaStream_t stream = nullptr);

private:
    TheoryHistChi2S2() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // THEORY_HIST_CHI2_S2_HPP
