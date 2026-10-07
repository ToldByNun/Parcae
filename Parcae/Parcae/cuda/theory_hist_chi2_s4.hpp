#ifndef THEORY_HIST_CHI2_S4_HPP
#define THEORY_HIST_CHI2_S4_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// S4 AutokeyRing fused χ² hist for vigenere_lag-class theories:
/// `out = x - AutokeyRing::shift(x, t, lag)` or `out = x + …` (mod 29).
///
/// No interpreter; per-candidate lag rows. Ring reads the cipher stream on
/// device (`AutokeyRingDevice`). `grid.y` via `HistTileCap::kS4`. Soft S0 on
/// lag bind fail stays in emit — this façade never invents keys. Finalize via
/// `Chi2BatchScore`.
class TheoryHistChi2S4 {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// `cipher_minus_ks == true` → `HistFast::dec_sub(x, key)`; else add.
    /// `device_lags` length `candidate_count`.
    [[nodiscard]] static Status launch_autokey_async(
        const std::uint8_t* device_in, const std::uint8_t* device_lags,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr);

    [[nodiscard]] static int tiles_for_public(std::size_t token_count);

private:
    TheoryHistChi2S4() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // THEORY_HIST_CHI2_S4_HPP
