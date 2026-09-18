#ifndef ATBASH_KERNEL_HPP
#define ATBASH_KERNEL_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `AtbashTransform::kernel`: `out[i] = 28 - in[i]` (Index29 as `uint8_t`).
/// Direction and interrupts unused (same as CPU). In-place OK (`device_in == device_out`).
class AtbashKernel {
public:
    [[nodiscard]] static Status launch_device(
        const std::uint8_t* device_in,
        std::uint8_t* device_out,
        std::size_t count);

    /// H2D → kernel → D2H. If `host_in.data() == host_out.data()`, runs in-place on device.
    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> host_in,
        std::span<std::uint8_t> host_out);

private:
    AtbashKernel() = delete;
};

#endif // ATBASH_KERNEL_HPP
