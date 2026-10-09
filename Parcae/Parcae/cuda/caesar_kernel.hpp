#ifndef CAESAR_KERNEL_HPP
#define CAESAR_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `CaesarTransform::kernel`: encrypt adds `shift`, decrypt subtracts (mod 29).
/// Interrupts unused (same as CPU). In-place OK (`device_in == device_out`).
///
/// Device path: uchar4 + `HistFast::{enc,dec}_caesar` when pointers are
/// `uchar4`-aligned; scalar `__ldg` otherwise. `launch_device_async` does not
/// synchronize; `launch_device` / `apply_host` sync (host / compose API, not SLO).
class CaesarKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              std::uint8_t shift, CudaDir direction);

    /// Same as `launch_device` but does not `cudaDeviceSynchronize`.
    [[nodiscard]] static Status launch_device_async(const std::uint8_t* device_in,
                                                    std::uint8_t* device_out, std::size_t count,
                                                    std::uint8_t shift, CudaDir direction);

    /// H2D → kernel → D2H. If `host_in.data() == host_out.data()`, runs in-place on device.
    /// Host convenience only — not a Kernel SLO path.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out, std::uint8_t shift,
                                           CudaDir direction);

private:
    CaesarKernel() = delete;

    [[nodiscard]] static Status validate_shift(std::uint8_t shift);
};

#endif // CAESAR_KERNEL_HPP
