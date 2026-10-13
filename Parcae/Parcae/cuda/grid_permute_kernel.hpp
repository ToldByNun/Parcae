#ifndef GRID_PERMUTE_KERNEL_HPP
#define GRID_PERMUTE_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA gather/scatter twin for grid-read families (spiral / boustrophedon /
/// diagonal / columnar). Order indices are host-filled (`GridReadOrder`) and
/// uploaded as uint32. Scaffold for toolkit 1.1.0 — device path lands in
/// catalog-parity roadmap E.
class GridPermuteKernel {
public:
    /// Decrypt-style gather: `out[i] = in[order[i]]`. Encrypt-style scatter flips.
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint32_t* device_order, CudaDir direction);

    [[nodiscard]] static Status launch_device_async(const std::uint8_t* device_in,
                                                    std::uint8_t* device_out, std::size_t count,
                                                    const std::uint32_t* device_order,
                                                    CudaDir direction);

    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint32_t> host_order,
                                           CudaDir direction);

private:
    GridPermuteKernel() = delete;
};

#endif // GRID_PERMUTE_KERNEL_HPP
