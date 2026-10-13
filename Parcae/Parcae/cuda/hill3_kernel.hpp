#ifndef HILL3_KERNEL_HPP
#define HILL3_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `Hill3Transform::kernel` (3×3 ℤ₂₉ block multiply).
/// Scaffold for toolkit 1.1.0 — device path lands in catalog-parity roadmap C.
class Hill3Kernel {
public:
    /// `device_matrix` is 9 row-major ℤ₂₉ entries (invertible key; host-validated).
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_matrix, CudaDir direction);

    [[nodiscard]] static Status launch_device_async(const std::uint8_t* device_in,
                                                    std::uint8_t* device_out, std::size_t count,
                                                    const std::uint8_t* device_matrix,
                                                    CudaDir direction);

    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_matrix,
                                           CudaDir direction);

private:
    Hill3Kernel() = delete;
};

#endif // HILL3_KERNEL_HPP
