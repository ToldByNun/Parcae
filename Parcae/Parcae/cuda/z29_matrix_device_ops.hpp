#ifndef Z29_MATRIX_DEVICE_OPS_HPP
#define Z29_MATRIX_DEVICE_OPS_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// Host/device façade for batched \(\mathbb{Z}_{29}\) matrix2/3 register ops.
/// Each matrix is contiguous row-major (`4` or `9` bytes). Singularity is one
/// `uint8_t` per matrix (`1` = singular, `0` = invertible inverse written).
class Z29MatrixDeviceOps {
public:
    static constexpr std::size_t kMatrix2Entries = 4;
    static constexpr std::size_t kMatrix3Entries = 9;
    static constexpr std::size_t kVec2 = 2;
    static constexpr std::size_t kVec3 = 3;

    /// Invert `count` 2×2 matrices. `matrices_in`/`matrices_out` size `4*count`;
    /// `singular` size `count`.
    [[nodiscard]] static Status launch_device_inverse2(const std::uint8_t* device_matrices_in,
                                                      std::uint8_t* device_matrices_out,
                                                      std::uint8_t* device_singular,
                                                      std::size_t count);

    [[nodiscard]] static Status apply_host_inverse2(std::span<const std::uint8_t> matrices_in,
                                                    std::span<std::uint8_t> matrices_out,
                                                    std::span<std::uint8_t> singular);

    [[nodiscard]] static Status launch_device_inverse3(const std::uint8_t* device_matrices_in,
                                                      std::uint8_t* device_matrices_out,
                                                      std::uint8_t* device_singular,
                                                      std::size_t count);

    [[nodiscard]] static Status apply_host_inverse3(std::span<const std::uint8_t> matrices_in,
                                                    std::span<std::uint8_t> matrices_out,
                                                    std::span<std::uint8_t> singular);

    /// `out_vecs[2*i..]` = `M_i · v_i` for `count` 2×2 matrices / vectors.
    [[nodiscard]] static Status launch_device_mul_vec2(const std::uint8_t* device_matrices,
                                                      const std::uint8_t* device_vecs_in,
                                                      std::uint8_t* device_vecs_out,
                                                      std::size_t count);

    [[nodiscard]] static Status apply_host_mul_vec2(std::span<const std::uint8_t> matrices,
                                                    std::span<const std::uint8_t> vecs_in,
                                                    std::span<std::uint8_t> vecs_out);

    [[nodiscard]] static Status launch_device_mul_vec3(const std::uint8_t* device_matrices,
                                                      const std::uint8_t* device_vecs_in,
                                                      std::uint8_t* device_vecs_out,
                                                      std::size_t count);

    [[nodiscard]] static Status apply_host_mul_vec3(std::span<const std::uint8_t> matrices,
                                                    std::span<const std::uint8_t> vecs_in,
                                                    std::span<std::uint8_t> vecs_out);

private:
    Z29MatrixDeviceOps() = delete;
};

#endif // Z29_MATRIX_DEVICE_OPS_HPP
