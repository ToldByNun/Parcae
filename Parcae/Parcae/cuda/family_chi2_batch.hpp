#ifndef FAMILY_CHI2_BATCH_HPP
#define FAMILY_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// Fused decrypt+χ² histogram for batch families (device-resident, scores only D2H).
///
/// **Atbash / Atbash∘Caesar production:** alphabet remap after one ciphertext
/// hist (`AlphabetChi2Batch`). Legacy decode-hist remains via
/// `launch_*_decode_hist_async` (tile-cap A/B on `HistTileCap::kAtbash`).
///
/// Other families still use warp-private decode-hist + fat-tile + uchar4.
class FamilyChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// `grid.y` under Atbash / totient slot (decode-hist A/B sweeps).
    [[nodiscard]] static int tiles_for_atbash(std::size_t token_count);
    [[nodiscard]] static int tiles_for_totient(std::size_t token_count);

    /// Production Atbash: alphabet mirror remap.
    [[nodiscard]] static Status
    launch_atbash_async(const std::uint8_t* device_in, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count);

    /// Legacy per-candidate Atbash decode→hist (Golden / tile-cap A/B).
    [[nodiscard]] static Status
    launch_atbash_decode_hist_async(const std::uint8_t* device_in,
                                    const double* device_probabilities,
                                    std::uint32_t* device_counts, double* device_scores,
                                    std::size_t candidate_count, std::size_t token_count);

    /// Production Atbash∘Caesar-encrypt compose remap.
    [[nodiscard]] static Status
    launch_atbash_caesar_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                               const double* device_probabilities, std::uint32_t* device_counts,
                               double* device_scores, std::size_t candidate_count,
                               std::size_t token_count);

    /// Legacy Atbash∘Caesar decode→hist.
    [[nodiscard]] static Status
    launch_atbash_caesar_decode_hist_async(const std::uint8_t* device_in,
                                           const std::uint8_t* device_shifts,
                                           const double* device_probabilities,
                                           std::uint32_t* device_counts, double* device_scores,
                                           std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status
    launch_affine_async(const std::uint8_t* device_in, const std::uint8_t* device_a,
                        const std::uint8_t* device_b, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status
    launch_vigenere_async(const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
                          const std::uint32_t* device_key_begin,
                          const std::uint32_t* device_key_len, const double* device_probabilities,
                          std::uint32_t* device_counts, double* device_scores,
                          std::size_t candidate_count, std::size_t token_count);

    /// Beaufort decrypt hist: `out = key[j] - in` (involution; same API as Vigenère).
    [[nodiscard]] static Status
    launch_beaufort_async(const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
                          const std::uint32_t* device_key_begin,
                          const std::uint32_t* device_key_len, const double* device_probabilities,
                          std::uint32_t* device_counts, double* device_scores,
                          std::size_t candidate_count, std::size_t token_count);

    /// Totient / prime−1 stream decrypt hist: `out = in - shifts[begin[c] + t]`.
    /// `device_shifts` must cover `begin[c] + token_count` for every candidate.
    [[nodiscard]] static Status
    launch_totient_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                         const std::uint32_t* device_shift_begin,
                         const double* device_probabilities, std::uint32_t* device_counts,
                         double* device_scores, std::size_t candidate_count,
                         std::size_t token_count);

private:
    FamilyChi2Batch() = delete;

    /// `tile_slot < 0` → `HistFast::tiles_for` (production clamp, no slot).
    [[nodiscard]] static Status clear_and_grid(std::uint32_t* device_counts,
                                               std::size_t candidate_count, std::size_t token_count,
                                               int tile_slot, dim3* grid_out);

    [[nodiscard]] static Status ensure_cipher_hist_scratch(std::uint32_t** out_hist);
};

#endif // FAMILY_CHI2_BATCH_HPP
