#ifndef CIPHER_HIST_ONCE_HPP
#define CIPHER_HIST_ONCE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <span>

/// Single dense ciphertext histogram on device (Identity bins, no decrypt).
///
/// Production once-count for alphabet-remap χ²: thread-local shared hist
/// (`HistFast::add_local` / `flush_local`, 32 KiB) + fat-tile grid-stride uchar4
/// loads. No C++ namespaces; `__global__` lives file-scope in the `.cu`.
class CipherHistOnce {
public:
    static constexpr std::size_t alphabet = 29;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Same fat-tile clamp as catalog fused hist (`HistFast::production_tile_cap`).
    static constexpr int kProductionTileCap = 64;

    /// Clears `device_counts[0..28]`, then histograms `device_in[0..T)`.
    /// `T == 0` succeeds with an all-zero hist (no kernel launch).
    /// `device_in` should be 4-byte aligned for the uchar4 path (cudaMalloc is).
    [[nodiscard]] static Status launch_async(const std::uint8_t* device_in,
                                             std::uint32_t* device_counts, std::size_t token_count,
                                             cudaStream_t stream = nullptr);

    /// `launch_async` + `cudaDeviceSynchronize`.
    [[nodiscard]] static Status launch(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                       std::size_t token_count);

    /// Host convenience: H2D → launch → D2H into `host_counts` (size 29).
    [[nodiscard]] static Status count_host(std::span<const std::uint8_t> host_cipher,
                                           std::span<std::uint32_t> host_counts);

    [[nodiscard]] static int tiles_for_public(std::size_t token_count);

private:
    CipherHistOnce() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status validate(const std::uint8_t* device_in,
                                         std::uint32_t* device_counts, std::size_t token_count);
};

#endif // CIPHER_HIST_ONCE_HPP
