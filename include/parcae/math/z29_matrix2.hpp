#ifndef Z29_MATRIX2_HPP
#define Z29_MATRIX2_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"

#include <array>
#include <cstddef>

/// 2×2 matrix over \(\mathbb{Z}_{29}\) (row-major: `[[a,b],[c,d]]`).
class Z29Matrix2 {
public:
    Z29Matrix2() noexcept
        : entries_{Index29{0}, Index29{0}, Index29{0}, Index29{0}} {}

    Z29Matrix2(Index29 a, Index29 b, Index29 c, Index29 d) noexcept
        : entries_{a, b, c, d} {}

    [[nodiscard]] static Z29Matrix2 from_row_major(std::array<Index29, 4> entries) noexcept {
        return Z29Matrix2{entries[0], entries[1], entries[2], entries[3]};
    }

    [[nodiscard]] Index29 a() const noexcept { return entries_[0]; }
    [[nodiscard]] Index29 b() const noexcept { return entries_[1]; }
    [[nodiscard]] Index29 c() const noexcept { return entries_[2]; }
    [[nodiscard]] Index29 d() const noexcept { return entries_[3]; }

    [[nodiscard]] Index29 at(std::size_t row, std::size_t col) const noexcept {
        return entries_[row * 2u + col];
    }

    [[nodiscard]] const std::array<Index29, 4>& row_major() const noexcept { return entries_; }

    /// \(\det = ad - bc \pmod{29}\).
    [[nodiscard]] Index29 det() const noexcept {
        return Z29::sub(Z29::mul(a(), d()), Z29::mul(b(), c()));
    }

    /// Inverse when \(\det \not\equiv 0 \pmod{29}\); otherwise `Status` error (no abort).
    [[nodiscard]] StatusOr<Z29Matrix2> try_inverse() const {
        const Index29 determinant = det();
        if (determinant.value() == 0) {
            return Status::error("Z29Matrix2::try_inverse: singular matrix (det ≡ 0 mod 29)");
        }
        const Index29 inv_det = Z29::inv(determinant);
        // (1/det) * [[d, -b], [-c, a]]
        return Z29Matrix2{Z29::mul(inv_det, d()), Z29::mul(inv_det, Z29::neg(b())),
                          Z29::mul(inv_det, Z29::neg(c())), Z29::mul(inv_det, a())};
    }

    /// Column vector multiply: `out = M · [x, y]^T` over \(\mathbb{Z}_{29}\).
    [[nodiscard]] std::array<Index29, 2> mul_vec(Index29 x, Index29 y) const noexcept {
        return {Z29::add(Z29::mul(a(), x), Z29::mul(b(), y)),
                Z29::add(Z29::mul(c(), x), Z29::mul(d(), y))};
    }

    [[nodiscard]] std::array<Index29, 2> mul_vec(std::array<Index29, 2> v) const noexcept {
        return mul_vec(v[0], v[1]);
    }

    [[nodiscard]] bool operator==(const Z29Matrix2&) const noexcept = default;

private:
    std::array<Index29, 4> entries_;
};

#endif // Z29_MATRIX2_HPP
