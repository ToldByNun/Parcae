#ifndef CIPHERTEXT_AUTOKEY_KERNEL_HPP
#define CIPHERTEXT_AUTOKEY_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "interrupt_device_view.hpp"
#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `CiphertextAutokeyTransform::kernel` (CTAK).
/// Dense decrypt is parallel (same formula as `DeepScoreBatch` autokey hist).
/// Encrypt and interrupt paths run serially (ciphertext feedback dependency).
class CiphertextAutokeyKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips,
                                              CudaDir direction);

    /// H2D key + interrupt encoding + stream, kernel, D2H.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_key,
                                           const InterruptDeviceView& interrupts,
                                           CudaDir direction);

private:
    CiphertextAutokeyKernel() = delete;
};

#endif // CIPHERTEXT_AUTOKEY_KERNEL_HPP
