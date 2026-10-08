#ifndef BIGRAM_COUNT_ONCE_HPP
#define BIGRAM_COUNT_ONCE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <span>

/// Once-count ciphertext adjacent-pair histogram `B[29×29]`.
///
/// `B[x0 * 29 + x1] = |{ t : 0 ≤ t < T−1 ∧ in[t]=x0 ∧ in[t+1]=x1 }|`.
/// `T < 2` → all-zero matrix (no pairs). Used by Caesar bigram-LL remap.
class BigramCountOnce {
public:
    static constexpr std::size_t alphabet = 29;
    static constexpr std::size_t bins = alphabet * alphabet; // 841
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr int kProductionTileCap = 64;

    /// Clears `device_counts[0 .. 840]`, then fills pair counts.
    /// `T == 0` succeeds with zeros. `device_in` may be null only when `T == 0`.
    [[nodiscard]] static Status launch_async(const std::uint8_t* device_in,
                                             std::uint32_t* device_counts, std::size_t token_count,
                                             cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                       std::size_t token_count);

    /// Host convenience: H2D → launch → D2H (`host_counts.size() == 841`).
    [[nodiscard]] static Status count_host(std::span<const std::uint8_t> host_cipher,
                                           std::span<std::uint32_t> host_counts);

    [[nodiscard]] static int tiles_for_public(std::size_t token_count);

private:
    BigramCountOnce() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status validate(const std::uint8_t* device_in,
                                         std::uint32_t* device_counts, std::size_t token_count);
};

#endif // BIGRAM_COUNT_ONCE_HPP
