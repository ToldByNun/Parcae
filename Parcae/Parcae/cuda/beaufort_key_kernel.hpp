#ifndef BEAUFORT_KEY_KERNEL_HPP
#define BEAUFORT_KEY_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "interrupt_device_view.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `BeaufortKeyTransform::kernel`: `out = key[j] - in` (mod 29).
/// Direction ignored (involution). Skip: pass-through; key does not advance. In-place OK.
class BeaufortKeyKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips);

    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_key,
                                           const InterruptDeviceView& interrupts);

private:
    BeaufortKeyKernel() = delete;
};

#endif // BEAUFORT_KEY_KERNEL_HPP
