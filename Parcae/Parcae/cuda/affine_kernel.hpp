#ifndef AFFINE_KERNEL_HPP
#define AFFINE_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `AffineTransform::kernel`: encrypt `a·x+b`, decrypt `inv(a)·(x-b)` mod 29.
/// Device inverse via `Z29Device::inv`. Interrupts unused. In-place OK.
///
/// Device path: uchar4 + `__ldg` when pointers are aligned; scalar otherwise.
/// `launch_device_async` does not synchronize; `launch_device` / `apply_host` sync
/// (host / compose API, not SLO).
class AffineKernel {
public:
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              std::uint8_t a, std::uint8_t b, CudaDir direction);

    /// Same as `launch_device` but does not `cudaDeviceSynchronize`.
    [[nodiscard]] static Status launch_device_async(const std::uint8_t* device_in,
                                                    std::uint8_t* device_out, std::size_t count,
                                                    std::uint8_t a, std::uint8_t b,
                                                    CudaDir direction);

    /// H2D → kernel → D2H. Host convenience only — not a Kernel SLO path.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out, std::uint8_t a,
                                           std::uint8_t b, CudaDir direction);

private:
    AffineKernel() = delete;

    [[nodiscard]] static Status validate_params(std::uint8_t a, std::uint8_t b);
};

#endif // AFFINE_KERNEL_HPP
