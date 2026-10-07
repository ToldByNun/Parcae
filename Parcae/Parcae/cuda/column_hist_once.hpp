#ifndef COLUMN_HIST_ONCE_HPP
#define COLUMN_HIST_ONCE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <span>

/// Dense period-`L` column histograms of a ciphertext stream.
///
/// `device_cols[j * 29 + x]` = count of `in[t] == x` with `t % L == j`.
/// Interrupt-free indexing only (`t % L`). Used by Vigenère / Beaufort remap.
class ColumnHistOnce {
public:
    static constexpr std::size_t alphabet = 29;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    /// Upper bound on period (matches FamilyChi2Batch key-length ceiling).
    static constexpr std::uint32_t kMaxPeriod = 16384;
    static constexpr int kProductionTileCap = 64;

    /// Clears `device_cols[0 .. L*29)`, then fills column hists.
    /// `period == 0` or `period > kMaxPeriod` → error. `T == 0` → all-zero cols.
    [[nodiscard]] static Status launch_async(const std::uint8_t* device_in,
                                             std::uint32_t* device_cols, std::size_t token_count,
                                             std::uint32_t period, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch(const std::uint8_t* device_in, std::uint32_t* device_cols,
                                       std::size_t token_count, std::uint32_t period);

    /// Host convenience: H2D → launch → D2H (`host_cols.size() == period * 29`).
    [[nodiscard]] static Status count_host(std::span<const std::uint8_t> host_cipher,
                                           std::uint32_t period,
                                           std::span<std::uint32_t> host_cols);

    [[nodiscard]] static int tiles_for_public(std::size_t token_count, std::uint32_t period);

private:
    ColumnHistOnce() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count, std::uint32_t period);

    [[nodiscard]] static Status validate(const std::uint8_t* device_in, std::uint32_t* device_cols,
                                         std::size_t token_count, std::uint32_t period);
};

#endif // COLUMN_HIST_ONCE_HPP
