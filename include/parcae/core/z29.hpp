#ifndef Z29_HPP
#define Z29_HPP

#include "parcae/core/index29.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>

class Z29 {
public:
    [[nodiscard]] static constexpr Index29 add(Index29 x, Index29 y) noexcept {
        const auto sum = static_cast<std::uint8_t>((x.value() + y.value()) % Index29::modulus);
        return Index29::unchecked(sum);
    }

    [[nodiscard]] static constexpr Index29 neg(Index29 x) noexcept {
        if (x.value() == 0) {
            return Index29::unchecked(0);
        }
        return Index29::unchecked(static_cast<std::uint8_t>(Index29::modulus - x.value()));
    }

    [[nodiscard]] static constexpr Index29 sub(Index29 x, Index29 y) noexcept {
        return add(x, neg(y));
    }

    [[nodiscard]] static constexpr Index29 mul(Index29 x, Index29 y) noexcept {
        const auto product = static_cast<std::uint8_t>((x.value() * y.value()) % Index29::modulus);
        return Index29::unchecked(product);
    }

    /// Modular inverse in Z_29. `inv(0)` aborts at runtime / is not constexpr.
    [[nodiscard]] static constexpr Index29 inv(Index29 a) {
        if (a.value() == 0) {
            fatal_invalid();
        }
        return Index29::unchecked(inv_table[a.value()]);
    }

    /// Atbash on Gematria indices: `28 - x`.
    [[nodiscard]] static constexpr Index29 atbash(Index29 x) noexcept {
        return Index29::unchecked(static_cast<std::uint8_t>(28 - x.value()));
    }

    /// Modular exponentiation `base^exp mod 29`. `0^0` → 1.
    [[nodiscard]] static constexpr Index29 pow(Index29 base, Index29 exp) noexcept {
        std::uint8_t result = 1;
        std::uint8_t b = base.value();
        std::uint8_t e = exp.value();
        while (e != 0) {
            if ((e & 1u) != 0) {
                result = static_cast<std::uint8_t>((result * b) % Index29::modulus);
            }
            b = static_cast<std::uint8_t>((b * b) % Index29::modulus);
            e = static_cast<std::uint8_t>(e >> 1);
        }
        return Index29::unchecked(result);
    }

    /// Integer floor-division of representatives (`//`); caller must ensure `y != 0`.
    [[nodiscard]] static constexpr Index29 floor_div(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(static_cast<std::uint8_t>(x.value() / y.value()));
    }

    /// Bitwise ops on representatives, then reduce mod 29 into Index29.
    [[nodiscard]] static constexpr Index29 bit_and(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(
            static_cast<std::uint8_t>((x.value() & y.value()) % Index29::modulus));
    }

    [[nodiscard]] static constexpr Index29 bit_or(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(
            static_cast<std::uint8_t>((x.value() | y.value()) % Index29::modulus));
    }

    [[nodiscard]] static constexpr Index29 bit_xor(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(
            static_cast<std::uint8_t>((x.value() ^ y.value()) % Index29::modulus));
    }

    /// Python-style `~x` reduced mod 29: `(-x-1) mod 29` == `28 - x` (Atbash).
    [[nodiscard]] static constexpr Index29 bit_not(Index29 x) noexcept { return atbash(x); }

    [[nodiscard]] static constexpr Index29 lshift(Index29 x, Index29 y) noexcept {
        const unsigned shift = y.value();
        if (shift >= 64u) {
            return Index29::unchecked(0);
        }
        const unsigned long long v = static_cast<unsigned long long>(x.value()) << shift;
        return Index29::unchecked(static_cast<std::uint8_t>(v % Index29::modulus));
    }

    [[nodiscard]] static constexpr Index29 rshift(Index29 x, Index29 y) noexcept {
        const unsigned shift = y.value();
        if (shift >= 8u) {
            return Index29::unchecked(0);
        }
        return Index29::unchecked(static_cast<std::uint8_t>(x.value() >> shift));
    }

    [[nodiscard]] static constexpr Index29 eq(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(x.value() == y.value() ? 1u : 0u);
    }

    [[nodiscard]] static constexpr Index29 ne(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(x.value() != y.value() ? 1u : 0u);
    }

    [[nodiscard]] static constexpr Index29 lt(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(x.value() < y.value() ? 1u : 0u);
    }

    [[nodiscard]] static constexpr Index29 le(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(x.value() <= y.value() ? 1u : 0u);
    }

    [[nodiscard]] static constexpr Index29 gt(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(x.value() > y.value() ? 1u : 0u);
    }

    [[nodiscard]] static constexpr Index29 ge(Index29 x, Index29 y) noexcept {
        return Index29::unchecked(x.value() >= y.value() ? 1u : 0u);
    }

    /// Boolean-ish on Index29: nonzero is true. Results are 0 or 1.
    [[nodiscard]] static constexpr Index29 bool_and(Index29 x, Index29 y) noexcept {
        return Index29::unchecked((x.value() != 0 && y.value() != 0) ? 1u : 0u);
    }

    [[nodiscard]] static constexpr Index29 bool_or(Index29 x, Index29 y) noexcept {
        return Index29::unchecked((x.value() != 0 || y.value() != 0) ? 1u : 0u);
    }

    [[nodiscard]] static constexpr Index29 bool_not(Index29 x) noexcept {
        return Index29::unchecked(x.value() == 0 ? 1u : 0u);
    }

    /// Mux: nonzero `cond` → `t`, else `f` (matches Z29Expr::Select / eval).
    [[nodiscard]] static constexpr Index29 select(Index29 cond, Index29 t, Index29 f) noexcept {
        return Index29::unchecked(cond.value() != 0 ? t.value() : f.value());
    }

private:
    static constexpr std::array<std::uint8_t, Index29::modulus> inv_table = []() {
        std::array<std::uint8_t, Index29::modulus> table{};
        for (std::uint8_t a = 1; a < Index29::modulus; ++a) {
            for (std::uint8_t x = 1; x < Index29::modulus; ++x) {
                if (static_cast<std::uint8_t>((a * x) % Index29::modulus) == 1) {
                    table[a] = x;
                    break;
                }
            }
        }
        return table;
    }();

    [[noreturn]] static void fatal_invalid() noexcept { std::abort(); }
};

#endif // Z29_HPP
