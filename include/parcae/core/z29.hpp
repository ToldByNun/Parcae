#ifndef Z29_HPP
#define Z29_HPP

#include "parcae/core/index29.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>

class Z29 {
public:
    [[nodiscard]] static constexpr Index29 add(Index29 x, Index29 y) noexcept {
        const auto sum =
            static_cast<std::uint8_t>((x.value() + y.value()) % Index29::modulus);
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
        const auto product =
            static_cast<std::uint8_t>((x.value() * y.value()) % Index29::modulus);
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

    [[noreturn]] static void fatal_invalid() noexcept {
        std::abort();
    }
};

#endif // Z29_HPP
