#ifndef AFFINE_KERNEL_HPP
#define AFFINE_KERNEL_HPP

#include "params.hpp"

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `AffineTransform::kernel`: encrypt `a·x+b`, decrypt `inv(a)·(x-b)` mod 29.
/// Device inverse via `Z29Device::inv`. Interrupts unused. In-place OK.
class AffineKernel {
public:
    [[nodiscard]] static Status launch_device(
        const std::uint8_t* device_in,
        std::uint8_t* device_out,
        std::size_t count,
        std::uint8_t a,
        std::uint8_t b,
        CudaDir direction);

    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> host_in,
        std::span<std::uint8_t> host_out,
        std::uint8_t a,
        std::uint8_t b,
        CudaDir direction);

private:
    AffineKernel() = delete;
};

#endif // AFFINE_KERNEL_HPP
