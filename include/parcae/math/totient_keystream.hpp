#ifndef TOTIENT_KEYSTREAM_HPP
#define TOTIENT_KEYSTREAM_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/math/primes.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/// Totient / prime−1 keystream over Z29: `shift[j] = (p_j - 1) % 29`.
/// Prime indices are 0-based (`p_0 = 2`). Used by `totient_prime_stream`.
class TotientKeystream {
public:
    /// Shift for a single 0-based prime index.
    [[nodiscard]] static StatusOr<Index29> shift_at(std::size_t prime_index) {
        StatusOr<std::uint64_t> prime = Primes::nth(prime_index);
        if (!prime.ok()) {
            return prime.status();
        }
        return from_prime(prime.value());
    }

    /// `count` consecutive shifts starting at `prime_start_index` (inclusive).
    [[nodiscard]] static StatusOr<std::vector<Index29>> shifts(
        std::size_t count,
        std::size_t prime_start_index = 0) {
        if (count == 0) {
            return std::vector<Index29>{};
        }

        StatusOr<std::vector<std::uint64_t>> primes =
            Primes::first(prime_start_index + count);
        if (!primes.ok()) {
            return primes.status();
        }

        std::vector<Index29> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(from_prime(primes.value()[prime_start_index + i]));
        }
        return out;
    }

    [[nodiscard]] static Index29 from_prime(std::uint64_t prime) noexcept {
        const auto reduced =
            static_cast<std::uint8_t>((prime - 1) % Index29::modulus);
        return Index29{reduced};
    }
};

#endif // TOTIENT_KEYSTREAM_HPP
