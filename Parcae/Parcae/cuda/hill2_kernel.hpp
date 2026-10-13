#ifndef HILL2_KERNEL_HPP
#define HILL2_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `Hill2Transform::kernel` (2×2 ℤ₂₉ block multiply).
/// Encrypt: `M·v`; decrypt: `M⁻¹·v`. Interrupt policy unused (block cipher).
///
/// `device_matrix` / `host_matrix` are 4 row-major ℤ₂₉ entries. Callers MUST
/// pass an invertible key (host-validated); device launch trusts that contract.
/// Input length MUST be even. In-place OK. `launch_device_async` does not
/// synchronize; `launch_device` / `apply_host` sync (host convenience, not SLO).
class Hill2Kernel {
public:
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
    Hill2Kernel() = delete;

    [[nodiscard]] static Status validate_matrix(std::span<const std::uint8_t> host_matrix);
};

#endif // HILL2_KERNEL_HPP
