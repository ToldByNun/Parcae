#ifndef PLAINTEXT_AUTOKEY_KERNEL_HPP
#define PLAINTEXT_AUTOKEY_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "interrupt_device_view.hpp"
#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `PlaintextAutokeyTransform::kernel` (PTAK).
/// Scaffold for toolkit 1.1.0 — device path lands in catalog-parity roadmap D.
class PlaintextAutokeyKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips,
                                              CudaDir direction);

    [[nodiscard]] static Status
    launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out, std::size_t count,
                        const std::uint8_t* device_key, std::uint32_t key_len,
                        const InterruptDeviceView& interrupts,
                        const std::uint32_t* device_bitmask_or_skips, CudaDir direction);

    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_key,
                                           const InterruptDeviceView& interrupts,
                                           CudaDir direction);

private:
    PlaintextAutokeyKernel() = delete;
};

#endif // PLAINTEXT_AUTOKEY_KERNEL_HPP
