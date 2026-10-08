#ifndef THEORY_HIST_CHI2_S4_HPP
#define THEORY_HIST_CHI2_S4_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// S4 AutokeyRing fused χ² for vigenere_lag-class theories:
/// `out = x - AutokeyRing::shift(x, t, lag)` or `out = x + …` (mod 29).
///
/// **Production:** LagDiff/LagSum once per unique lag + key-0 ring prefix
/// (`CipherHistOnce` on `min(L,T)`); identical merged hist broadcast to all
/// candidates sharing that lag. Legacy decode→hist remains as
/// `launch_autokey_decode_hist_async`. Finalize via `Chi2BatchScore`.
class TheoryHistChi2S4 {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Production AutokeyRing remap. `device_lags` length `candidate_count`.
    /// `cipher_minus_ks == true` → `HistFast::dec_sub(x, key)`; else add.
    [[nodiscard]] static Status launch_autokey_async(
        const std::uint8_t* device_in, const std::uint8_t* device_lags,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr);

    /// Legacy per-candidate AutokeyRing decode→hist.
    [[nodiscard]] static Status launch_autokey_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_lags,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr);

    [[nodiscard]] static int tiles_for_public(std::size_t token_count);

private:
    TheoryHistChi2S4() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status ensure_lag_scratch(std::uint32_t** out_lag);

    [[nodiscard]] static Status ensure_prefix_scratch(std::uint32_t** out_prefix);

    [[nodiscard]] static Status launch_autokey_remap_async(
        const std::uint8_t* device_in, const std::uint8_t* device_lags,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream);
};

#endif // THEORY_HIST_CHI2_S4_HPP
