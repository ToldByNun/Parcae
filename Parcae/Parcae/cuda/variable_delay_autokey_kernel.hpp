#ifndef VARIABLE_DELAY_AUTOKEY_KERNEL_HPP
#define VARIABLE_DELAY_AUTOKEY_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "interrupt_device_view.hpp"
#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `VariableDelayAutokeyTransform::kernel` (VDA).
/// Scaffold for toolkit 1.1.0 — device path lands in catalog-parity roadmap D.
class VariableDelayAutokeyKernel {
public:
    [[nodiscard]] static Status
    launch_device(const std::uint8_t* device_in, std::uint8_t* device_out, std::size_t count,
                  const std::uint8_t* device_key, std::uint32_t key_len, std::uint32_t lag,
                  CudaAutokeyMode mode, const InterruptDeviceView& interrupts,
                  const std::uint32_t* device_bitmask_or_skips, CudaDir direction);

    [[nodiscard]] static Status
    launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out, std::size_t count,
                        const std::uint8_t* device_key, std::uint32_t key_len, std::uint32_t lag,
                        CudaAutokeyMode mode, const InterruptDeviceView& interrupts,
                        const std::uint32_t* device_bitmask_or_skips, CudaDir direction);

    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_key, std::uint32_t lag,
                                           CudaAutokeyMode mode,
                                           const InterruptDeviceView& interrupts,
                                           CudaDir direction);

private:
    VariableDelayAutokeyKernel() = delete;
};

#endif // VARIABLE_DELAY_AUTOKEY_KERNEL_HPP
