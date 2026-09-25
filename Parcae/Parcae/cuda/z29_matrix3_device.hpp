#ifndef Z29_MATRIX3_DEVICE_HPP
#define Z29_MATRIX3_DEVICE_HPP

#include "z29_device.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_HD __host__ __device__
#else
#define PARCAE_HD
#endif

/// Device/host 3×3 matrix ops over \(\mathbb{Z}_{29}\) matching CPU `Z29Matrix3`.
/// Row-major nine entries. Singular inverse sets `*singular = 1` and does not abort.
class Z29Matrix3Device {
public:
    [[nodiscard]] PARCAE_HD static std::uint8_t det(const std::uint8_t m[9]) noexcept {
        const std::uint8_t a = m[0];
        const std::uint8_t b = m[1];
        const std::uint8_t c = m[2];
        const std::uint8_t d = m[3];
        const std::uint8_t e = m[4];
        const std::uint8_t f = m[5];
        const std::uint8_t g = m[6];
        const std::uint8_t h = m[7];
        const std::uint8_t i = m[8];
        const std::uint8_t t0 = minor2(e, f, h, i);
        const std::uint8_t t1 = minor2(d, f, g, i);
        const std::uint8_t t2 = minor2(d, e, g, h);
        return Z29Device::add(Z29Device::sub(Z29Device::mul(a, t0), Z29Device::mul(b, t1)),
                              Z29Device::mul(c, t2));
    }

    /// Inverse when invertible. Sets `*singular` to 1 if \(\det \equiv 0\) (leaves `out`
    /// unspecified); otherwise writes the inverse and sets `*singular` to 0.
    PARCAE_HD static void try_inverse(const std::uint8_t m[9], std::uint8_t out[9],
                                      std::uint8_t* singular) noexcept {
        const std::uint8_t determinant = det(m);
        if (determinant == 0) {
            *singular = 1;
            return;
        }
        *singular = 0;
        const std::uint8_t inv_det = Z29Device::inv(determinant);

        const std::uint8_t a = m[0];
        const std::uint8_t b = m[1];
        const std::uint8_t c = m[2];
        const std::uint8_t d = m[3];
        const std::uint8_t e = m[4];
        const std::uint8_t f = m[5];
        const std::uint8_t g = m[6];
        const std::uint8_t h = m[7];
        const std::uint8_t i = m[8];

        const std::uint8_t adj00 = minor2(e, f, h, i);
        const std::uint8_t adj01 = Z29Device::neg(minor2(b, c, h, i));
        const std::uint8_t adj02 = minor2(b, c, e, f);
        const std::uint8_t adj10 = Z29Device::neg(minor2(d, f, g, i));
        const std::uint8_t adj11 = minor2(a, c, g, i);
        const std::uint8_t adj12 = Z29Device::neg(minor2(a, c, d, f));
        const std::uint8_t adj20 = minor2(d, e, g, h);
        const std::uint8_t adj21 = Z29Device::neg(minor2(a, b, g, h));
        const std::uint8_t adj22 = minor2(a, b, d, e);

        out[0] = Z29Device::mul(inv_det, adj00);
        out[1] = Z29Device::mul(inv_det, adj01);
        out[2] = Z29Device::mul(inv_det, adj02);
        out[3] = Z29Device::mul(inv_det, adj10);
        out[4] = Z29Device::mul(inv_det, adj11);
        out[5] = Z29Device::mul(inv_det, adj12);
        out[6] = Z29Device::mul(inv_det, adj20);
        out[7] = Z29Device::mul(inv_det, adj21);
        out[8] = Z29Device::mul(inv_det, adj22);
    }

    /// `out = M · [x, y, z]^T` over \(\mathbb{Z}_{29}\).
    PARCAE_HD static void mul_vec(const std::uint8_t m[9], std::uint8_t x, std::uint8_t y,
                                  std::uint8_t z, std::uint8_t out[3]) noexcept {
        out[0] = Z29Device::add(
            Z29Device::add(Z29Device::mul(m[0], x), Z29Device::mul(m[1], y)),
            Z29Device::mul(m[2], z));
        out[1] = Z29Device::add(
            Z29Device::add(Z29Device::mul(m[3], x), Z29Device::mul(m[4], y)),
            Z29Device::mul(m[5], z));
        out[2] = Z29Device::add(
            Z29Device::add(Z29Device::mul(m[6], x), Z29Device::mul(m[7], y)),
            Z29Device::mul(m[8], z));
    }

private:
    [[nodiscard]] PARCAE_HD static std::uint8_t minor2(std::uint8_t p, std::uint8_t q,
                                                       std::uint8_t r, std::uint8_t s) noexcept {
        return Z29Device::sub(Z29Device::mul(p, s), Z29Device::mul(q, r));
    }

    Z29Matrix3Device() = delete;
};

#undef PARCAE_HD

#endif // Z29_MATRIX3_DEVICE_HPP
