#ifndef CAESAR_KERNEL_HPP
#define CAESAR_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `CaesarTransform::kernel`: encrypt adds `shift`, decrypt subtracts (mod 29).
/// Interrupts unused (same as CPU). In-place OK (`device_in == device_out`).
class CaesarKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              std::uint8_t shift, CudaDir direction);

    /// H2D → kernel → D2H. If `host_in.data() == host_out.data()`, runs in-place on device.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out, std::uint8_t shift,
                                           CudaDir direction);

private:
    CaesarKernel() = delete;
};

#endif // CAESAR_KERNEL_HPP
