#ifndef CIPHERTEXT_AUTOKEY_KERNEL_HPP
#define CIPHERTEXT_AUTOKEY_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "interrupt_device_view.hpp"
#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `CiphertextAutokeyTransform::kernel` (CTAK materialize).
/// Dense decrypt is parallel (same formula as `DeepScoreBatch` autokey hist).
/// Encrypt and interrupt paths run serially (ciphertext feedback dependency).
///
/// Dense decrypt: uchar4 packs when stream pointers are aligned.
/// `launch_device_async` does not synchronize; `launch_device` / `apply_host` sync
/// (host / compose API, not SLO).
class CiphertextAutokeyKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips,
                                              CudaDir direction);

    /// Same as `launch_device` but does not `cudaDeviceSynchronize`.
    [[nodiscard]] static Status
    launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out, std::size_t count,
                        const std::uint8_t* device_key, std::uint32_t key_len,
                        const InterruptDeviceView& interrupts,
                        const std::uint32_t* device_bitmask_or_skips, CudaDir direction);

    /// Host convenience only — not a Kernel SLO path.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_key,
                                           const InterruptDeviceView& interrupts,
                                           CudaDir direction);

private:
    CiphertextAutokeyKernel() = delete;
};

#endif // CIPHERTEXT_AUTOKEY_KERNEL_HPP
