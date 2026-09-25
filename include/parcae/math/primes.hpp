#ifndef PRIMES_HPP
#define PRIMES_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

/// Deterministic prime utilities (sieve + 0-based nth-prime).
/// Indexing: `nth(0) == 2`, `nth(1) == 3`, … — matches totient stream `p0, p1, …`.
///
/// `first` / `nth` share a process-lifetime sieve cache (grows monotonically).
/// Not safe for concurrent `first`/`nth` from multiple threads without external sync.
class Primes {
public:
    /// All primes `<= limit` via Eratosthenes. `limit < 2` yields an empty list.
    [[nodiscard]] static std::vector<std::uint64_t> sieve_upto(std::uint64_t limit) {
        if (limit < 2) {
            return {};
        }

        std::vector<bool> is_composite(static_cast<std::size_t>(limit) + 1, false);
        std::vector<std::uint64_t> primes;
        primes.reserve(static_cast<std::size_t>(limit / 2) + 1);

        for (std::uint64_t candidate = 2; candidate <= limit; ++candidate) {
            if (is_composite[static_cast<std::size_t>(candidate)]) {
                continue;
            }
            primes.push_back(candidate);
            if (candidate > limit / candidate) {
                continue;
            }
            for (std::uint64_t multiple = candidate * candidate; multiple <= limit;
                 multiple += candidate) {
                is_composite[static_cast<std::size_t>(multiple)] = true;
            }
        }
        return primes;
    }

    /// First `count` primes in ascending order. `count == 0` → empty.
    [[nodiscard]] static StatusOr<std::vector<std::uint64_t>> first(std::size_t count) {
        if (count == 0) {
            return std::vector<std::uint64_t>{};
        }

        Status ensured = ensure_at_least(count);
        if (!ensured.ok()) {
            return ensured;
        }

        const std::vector<std::uint64_t>& cached = cache();
        return std::vector<std::uint64_t>(cached.begin(),
                                          cached.begin() + static_cast<std::ptrdiff_t>(count));
    }

    /// 0-based nth prime (`nth(0) == 2`).
    [[nodiscard]] static StatusOr<std::uint64_t> nth(std::size_t index) {
        Status ensured = ensure_at_least(index + 1);
        if (!ensured.ok()) {
            return ensured;
        }
        return cache()[index];
    }

private:
    /// Process-lifetime prime table shared by `first` / `nth` (never shrinks).
    [[nodiscard]] static std::vector<std::uint64_t>& cache() {
        static std::vector<std::uint64_t> primes;
        return primes;
    }

    /// Grow the cache until it holds at least `count` primes.
    [[nodiscard]] static Status ensure_at_least(std::size_t count) {
        std::vector<std::uint64_t>& primes = cache();
        if (primes.size() >= count) {
            return Status::success();
        }

        const std::uint64_t limit = upper_bound_for_nth(count - 1);
        std::vector<std::uint64_t> sieved = sieve_upto(limit);
        if (sieved.size() < count) {
            return Status::error("prime sieve upper bound insufficient");
        }
        primes = std::move(sieved);
        return Status::success();
    }

    /// Safe upper bound for the 0-based `index`-th prime.
    [[nodiscard]] static std::uint64_t upper_bound_for_nth(std::size_t index) {
        // Small indices: hard-coded ceilings (p_0..p_5 known).
        static constexpr std::uint64_t small_limits[] = {2, 3, 5, 7, 11, 13};
        if (index < 6) {
            return small_limits[index];
        }

        // Rosser–Schoenfeld: for n >= 6, p_n < n (ln n + ln ln n)
        // Here n is 1-based prime ordinal = index + 1.
        const double n = static_cast<double>(index) + 1.0;
        const double bound = n * (std::log(n) + std::log(std::log(n)));
        // +1 and ceil for floating error; pad a little for safety.
        return static_cast<std::uint64_t>(bound) + 3;
    }
};

#endif // PRIMES_HPP
