#ifndef VIGENERE_KEY_KERNEL_HPP
#define VIGENERE_KEY_KERNEL_HPP

#include "interrupt_device_view.hpp"
#include "params.hpp"

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `VigenereKeyTransform::kernel`.
/// Non-skip: encrypt `in+key[j]`, decrypt `in-key[j]`, then advance key cursor.
/// Skip: pass-through; key does not advance. In-place OK.
class VigenereKeyKernel {
public:
    [[nodiscard]] static Status launch_device(
        const std::uint8_t* device_in,
        std::uint8_t* device_out,
        std::size_t count,
        const std::uint8_t* device_key,
        std::uint32_t key_len,
        const InterruptDeviceView& interrupts,
        const std::uint32_t* device_bitmask_or_skips,
        CudaDir direction);

    /// H2D key + interrupt encoding + stream, kernel, D2H.
    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> host_in,
        std::span<std::uint8_t> host_out,
        std::span<const std::uint8_t> host_key,
        const InterruptDeviceView& interrupts,
        CudaDir direction);

private:
    VigenereKeyKernel() = delete;
};

#endif // VIGENERE_KEY_KERNEL_HPP
