#ifndef Z29_MATRIX2_DEVICE_HPP
#define Z29_MATRIX2_DEVICE_HPP

#include "z29_device.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_HD __host__ __device__
#else
#define PARCAE_HD
#endif

/// Device/host 2×2 matrix ops over \(\mathbb{Z}_{29}\) matching CPU `Z29Matrix2`.
/// Row-major layout: `m = {a, b, c, d}` for `[[a,b],[c,d]]`.
/// Singular inverse sets `*singular = 1` and does not abort.
class Z29Matrix2Device {
public:
    /// \(\det = ad - bc \pmod{29}\).
    [[nodiscard]] PARCAE_HD static std::uint8_t det(const std::uint8_t m[4]) noexcept {
        return Z29Device::sub(Z29Device::mul(m[0], m[3]), Z29Device::mul(m[1], m[2]));
    }

    /// Inverse when invertible. Sets `*singular` to 1 if \(\det \equiv 0\) (leaves `out`
    /// unspecified); otherwise writes the inverse and sets `*singular` to 0.
    PARCAE_HD static void try_inverse(const std::uint8_t m[4], std::uint8_t out[4],
                                      std::uint8_t* singular) noexcept {
        const std::uint8_t determinant = det(m);
        if (determinant == 0) {
            *singular = 1;
            return;
        }
        *singular = 0;
        const std::uint8_t inv_det = Z29Device::inv(determinant);
        // (1/det) * [[d, -b], [-c, a]]
        out[0] = Z29Device::mul(inv_det, m[3]);
        out[1] = Z29Device::mul(inv_det, Z29Device::neg(m[1]));
        out[2] = Z29Device::mul(inv_det, Z29Device::neg(m[2]));
        out[3] = Z29Device::mul(inv_det, m[0]);
    }

    /// `out = M · [x, y]^T` over \(\mathbb{Z}_{29}\).
    PARCAE_HD static void mul_vec(const std::uint8_t m[4], std::uint8_t x, std::uint8_t y,
                                  std::uint8_t out[2]) noexcept {
        out[0] = Z29Device::add(Z29Device::mul(m[0], x), Z29Device::mul(m[1], y));
        out[1] = Z29Device::add(Z29Device::mul(m[2], x), Z29Device::mul(m[3], y));
    }

private:
    Z29Matrix2Device() = delete;
};

#undef PARCAE_HD

#endif // Z29_MATRIX2_DEVICE_HPP
