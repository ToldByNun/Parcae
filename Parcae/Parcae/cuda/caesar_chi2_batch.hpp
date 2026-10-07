#ifndef CAESAR_CHI2_BATCH_HPP
#define CAESAR_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Device-resident Caesar decrypt sweep + χ² histogram/finalize.
///
/// Does **not** materialize `out[C*T]`. Counts are `uint32` (T ≤ 4M).
///
/// **Production decrypt** (`launch_decrypt_async`): ciphertext hist once +
/// alphabet remap (`AlphabetChi2Batch`), then `Chi2BatchScore` finalize.
///
/// **Legacy decode-hist** (`launch_decode_hist_async`): per-candidate stream
/// decode + warp-private hist (tile-cap A/B via `set_hist_tile_cap`). Direction
/// mixes still use the decode-hist kernels (`launch` / `launch_async`).
class CaesarChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Same as `HistFast::production_tile_cap` (legacy decode-hist grid.y).
    static constexpr int kProductionTileCap = 64;

    /// `0` = production (`kProductionTileCap`). Positive = override clamp
    /// (use `HistFast::max_tiles` to A/B uncapped tiling).
    /// Forwards to `HistTileCap::set(HistTileCap::kCaesar, cap)`.
    /// Applies to decode-hist launches only — not the remap production path.
    static void set_hist_tile_cap(int cap) noexcept;

    [[nodiscard]] static int hist_tile_cap() noexcept;

    /// Effective `grid.y` for the current decode-hist tile cap.
    [[nodiscard]] static int tiles_for_public(std::size_t token_count);

    [[nodiscard]] static Status launch(const std::uint8_t* device_in,
                                       const std::uint8_t* device_shifts,
                                       const std::uint8_t* device_directions,
                                       const double* device_probabilities,
                                       std::uint32_t* device_counts, double* device_scores,
                                       std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status launch_async(const std::uint8_t* device_in,
                                             const std::uint8_t* device_shifts,
                                             const std::uint8_t* device_directions,
                                             const double* device_probabilities,
                                             std::uint32_t* device_counts, double* device_scores,
                                             std::size_t candidate_count, std::size_t token_count);

    /// Production search decrypt: alphabet remap after one ciphertext hist.
    [[nodiscard]] static Status
    launch_decrypt_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                         const double* device_probabilities, std::uint32_t* device_counts,
                         double* device_scores, std::size_t candidate_count,
                         std::size_t token_count);

    /// Legacy per-candidate decrypt→hist (Golden / tile-cap A/B).
    [[nodiscard]] static Status
    launch_decode_hist_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                             const double* device_probabilities, std::uint32_t* device_counts,
                             double* device_scores, std::size_t candidate_count,
                             std::size_t token_count);

private:
    CaesarChi2Batch() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status
    validate(std::size_t candidate_count, std::size_t token_count, const std::uint8_t* device_in,
             const std::uint8_t* device_shifts, const std::uint8_t* device_directions,
             const double* device_probabilities, std::uint32_t* device_counts,
             double* device_scores, bool decrypt_only);

    [[nodiscard]] static Status
    launch_decode_impl(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                       const std::uint8_t* device_directions, const double* device_probabilities,
                       std::uint32_t* device_counts, double* device_scores,
                       std::size_t candidate_count, std::size_t token_count, bool synchronize,
                       bool decrypt_only);
};

#endif // CAESAR_CHI2_BATCH_HPP
