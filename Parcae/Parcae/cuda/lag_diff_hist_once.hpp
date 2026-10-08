#ifndef LAG_DIFF_HIST_ONCE_HPP
#define LAG_DIFF_HIST_ONCE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <span>

/// Once-count lag-diff histogram: `D[b] = |{t ≥ L : (in[t] - in[t-L]) ≡ b}|`.
///
/// Shared by dense CTAK remap (`DeepScoreBatch`) and AutokeyRing / S4.
/// `L == 0` → error. `L >= T` → all-zero hist (no kernel).
class LagDiffHistOnce {
public:
    static constexpr std::size_t alphabet = 29;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr int kProductionTileCap = 64;

    /// Clears `device_counts[0..28]`, then fills lag-diff hist for `lag`.
    [[nodiscard]] static Status launch_async(const std::uint8_t* device_in,
                                             std::uint32_t* device_counts, std::size_t token_count,
                                             std::uint32_t lag, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                       std::size_t token_count, std::uint32_t lag);

    /// Host convenience: H2D → launch → D2H (`host_counts.size() == 29`).
    [[nodiscard]] static Status count_host(std::span<const std::uint8_t> host_cipher,
                                           std::uint32_t lag,
                                           std::span<std::uint32_t> host_counts);

    [[nodiscard]] static int tiles_for_public(std::size_t token_count, std::uint32_t lag);

private:
    LagDiffHistOnce() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count, std::uint32_t lag);

    [[nodiscard]] static Status validate(const std::uint8_t* device_in,
                                         std::uint32_t* device_counts, std::size_t token_count,
                                         std::uint32_t lag);
};

#endif // LAG_DIFF_HIST_ONCE_HPP
