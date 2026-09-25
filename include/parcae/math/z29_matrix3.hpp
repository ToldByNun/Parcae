#ifndef Z29_MATRIX3_HPP
#define Z29_MATRIX3_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"

#include <array>
#include <cstddef>

/// 3×3 matrix over \(\mathbb{Z}_{29}\) (row-major: nine entries).
class Z29Matrix3 {
public:
    Z29Matrix3() noexcept : entries_{} {}

    explicit Z29Matrix3(std::array<Index29, 9> entries) noexcept : entries_(entries) {}

    Z29Matrix3(Index29 e00, Index29 e01, Index29 e02, Index29 e10, Index29 e11, Index29 e12,
               Index29 e20, Index29 e21, Index29 e22) noexcept
        : entries_{e00, e01, e02, e10, e11, e12, e20, e21, e22} {}

    [[nodiscard]] static Z29Matrix3 from_row_major(std::array<Index29, 9> entries) noexcept {
        return Z29Matrix3{entries};
    }

    [[nodiscard]] Index29 at(std::size_t row, std::size_t col) const noexcept {
        return entries_[row * 3u + col];
    }

    [[nodiscard]] const std::array<Index29, 9>& row_major() const noexcept { return entries_; }

    /// \(\det\) via cofactor expansion along the first row, all ops mod 29.
    [[nodiscard]] Index29 det() const noexcept {
        // | a b c |
        // | d e f |  → a(ei−fh) − b(di−fg) + c(dh−eg)
        // | g h i |
        const Index29 a = at(0, 0);
        const Index29 b = at(0, 1);
        const Index29 c = at(0, 2);
        const Index29 d = at(1, 0);
        const Index29 e = at(1, 1);
        const Index29 f = at(1, 2);
        const Index29 g = at(2, 0);
        const Index29 h = at(2, 1);
        const Index29 i = at(2, 2);
        const Index29 t0 = minor2(e, f, h, i);
        const Index29 t1 = minor2(d, f, g, i);
        const Index29 t2 = minor2(d, e, g, h);
        return Z29::add(Z29::sub(Z29::mul(a, t0), Z29::mul(b, t1)), Z29::mul(c, t2));
    }

    /// Inverse when \(\det \not\equiv 0 \pmod{29}\); otherwise `Status` error (no abort).
    [[nodiscard]] StatusOr<Z29Matrix3> try_inverse() const {
        const Index29 determinant = det();
        if (determinant.value() == 0) {
            return Status::error("Z29Matrix3::try_inverse: singular matrix (det ≡ 0 mod 29)");
        }
        const Index29 inv_det = Z29::inv(determinant);

        const Index29 a = at(0, 0);
        const Index29 b = at(0, 1);
        const Index29 c = at(0, 2);
        const Index29 d = at(1, 0);
        const Index29 e = at(1, 1);
        const Index29 f = at(1, 2);
        const Index29 g = at(2, 0);
        const Index29 h = at(2, 1);
        const Index29 i = at(2, 2);

        // Adjugate = transpose of cofactor matrix.
        const Index29 adj00 = minor2(e, f, h, i);
        const Index29 adj01 = Z29::neg(minor2(b, c, h, i));
        const Index29 adj02 = minor2(b, c, e, f);
        const Index29 adj10 = Z29::neg(minor2(d, f, g, i));
        const Index29 adj11 = minor2(a, c, g, i);
        const Index29 adj12 = Z29::neg(minor2(a, c, d, f));
        const Index29 adj20 = minor2(d, e, g, h);
        const Index29 adj21 = Z29::neg(minor2(a, b, g, h));
        const Index29 adj22 = minor2(a, b, d, e);

        return Z29Matrix3{Z29::mul(inv_det, adj00), Z29::mul(inv_det, adj01),
                          Z29::mul(inv_det, adj02), Z29::mul(inv_det, adj10),
                          Z29::mul(inv_det, adj11), Z29::mul(inv_det, adj12),
                          Z29::mul(inv_det, adj20), Z29::mul(inv_det, adj21),
                          Z29::mul(inv_det, adj22)};
    }

    /// Column vector multiply: `out = M · [x, y, z]^T` over \(\mathbb{Z}_{29}\).
    [[nodiscard]] std::array<Index29, 3> mul_vec(Index29 x, Index29 y, Index29 z) const noexcept {
        return {Z29::add(Z29::add(Z29::mul(at(0, 0), x), Z29::mul(at(0, 1), y)),
                         Z29::mul(at(0, 2), z)),
                Z29::add(Z29::add(Z29::mul(at(1, 0), x), Z29::mul(at(1, 1), y)),
                         Z29::mul(at(1, 2), z)),
                Z29::add(Z29::add(Z29::mul(at(2, 0), x), Z29::mul(at(2, 1), y)),
                         Z29::mul(at(2, 2), z))};
    }

    [[nodiscard]] std::array<Index29, 3> mul_vec(std::array<Index29, 3> v) const noexcept {
        return mul_vec(v[0], v[1], v[2]);
    }

    [[nodiscard]] bool operator==(const Z29Matrix3&) const noexcept = default;

private:
    /// 2×2 minor determinant \(ps - qr \pmod{29}\).
    [[nodiscard]] static Index29 minor2(Index29 p, Index29 q, Index29 r, Index29 s) noexcept {
        return Z29::sub(Z29::mul(p, s), Z29::mul(q, r));
    }

    std::array<Index29, 9> entries_;
};

#endif // Z29_MATRIX3_HPP
