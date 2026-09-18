#ifndef TOTIENT_PRIME_STREAM_KERNEL_HPP
#define TOTIENT_PRIME_STREAM_KERNEL_HPP

#include "interrupt_device_view.hpp"
#include "params.hpp"

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `TotientPrimeStreamTransform::kernel`.
///
/// Shifts are **host-materialized** (`TotientKeystream::shifts_into`); device only
/// reads `shifts[j]` for non-skip positions (`j` = consumable cursor). Encrypt adds,
/// decrypt subtracts. Skip: pass-through. In-place OK.
class TotientPrimeStreamKernel {
public:
    [[nodiscard]] static Status launch_device(
        const std::uint8_t* device_in,
        std::uint8_t* device_out,
        std::size_t count,
        const std::uint8_t* device_shifts,
        std::uint32_t shift_len,
        const InterruptDeviceView& interrupts,
        const std::uint32_t* device_bitmask_or_skips,
        CudaDir direction);

    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> host_in,
        std::span<std::uint8_t> host_out,
        std::span<const std::uint8_t> host_shifts,
        const InterruptDeviceView& interrupts,
        CudaDir direction);

private:
    TotientPrimeStreamKernel() = delete;
};

#endif // TOTIENT_PRIME_STREAM_KERNEL_HPP
