#ifndef INDEX29_HPP
#define INDEX29_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <compare>
#include <cstdint>
#include <cstdlib>

class Index29 {
public:
    static constexpr std::uint8_t modulus = 29;

    constexpr Index29() noexcept : value_(0) {}

    /// Checked construction. Out-of-range values abort at runtime / are not constexpr.
    explicit constexpr Index29(std::uint8_t value) : value_(value) {
        if (value >= modulus) {
            fatal_invalid();
        }
    }

    /// Soft construction: out-of-range → `Status` error (no abort).
    [[nodiscard]] static StatusOr<Index29> try_make(std::uint8_t value) {
        if (value >= modulus) {
            return Status::error("Index29::try_make: value out of range 0..28");
        }
        return unchecked(value);
    }

    [[nodiscard]] constexpr std::uint8_t value() const noexcept { return value_; }

    [[nodiscard]] constexpr explicit operator std::uint8_t() const noexcept { return value_; }

    [[nodiscard]] constexpr bool operator==(const Index29&) const noexcept = default;
    [[nodiscard]] constexpr auto operator<=>(const Index29&) const noexcept = default;

private:
    friend class Z29;

    struct UncheckedTag {};

    constexpr Index29(UncheckedTag /*tag*/, std::uint8_t value) noexcept : value_(value) {}

    [[nodiscard]] static constexpr Index29 unchecked(std::uint8_t value) noexcept {
        return Index29{UncheckedTag{}, value};
    }

    [[noreturn]] static void fatal_invalid() noexcept { std::abort(); }

    std::uint8_t value_;
};

#endif // INDEX29_HPP
