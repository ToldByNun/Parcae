#ifndef BEAUFORT_KEY_KERNEL_HPP
#define BEAUFORT_KEY_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "interrupt_device_view.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `BeaufortKeyTransform::kernel`: `out = key[j] - in` (mod 29).
/// Direction ignored (involution). Skip: pass-through; key does not advance. In-place OK.
///
/// Dense (no skips): uchar4 packs when stream pointers are aligned.
/// `launch_device_async` does not synchronize; `launch_device` / `apply_host` sync
/// (host / compose API, not SLO).
class BeaufortKeyKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips);

    /// Same as `launch_device` but does not `cudaDeviceSynchronize`.
    [[nodiscard]] static Status
    launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out, std::size_t count,
                        const std::uint8_t* device_key, std::uint32_t key_len,
                        const InterruptDeviceView& interrupts,
                        const std::uint32_t* device_bitmask_or_skips);

    /// Host convenience only — not a Kernel SLO path.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_key,
                                           const InterruptDeviceView& interrupts);

private:
    BeaufortKeyKernel() = delete;
};

#endif // BEAUFORT_KEY_KERNEL_HPP
