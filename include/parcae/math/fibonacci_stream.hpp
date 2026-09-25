#ifndef FIBONACCI_STREAM_HPP
#define FIBONACCI_STREAM_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/z29.hpp"

#include <cstddef>
#include <vector>

/// Deterministic Fibonacci stream over \(\mathbb{Z}_{29}\).
/// Classic indexing: `nth(0) == 0`, `nth(1) == 1`, then \(F_n = F_{n-1}+F_{n-2} \pmod{29}\).
/// Seeded variants start from arbitrary `(s0, s1)` in `0..28`.
class FibonacciStream {
public:
    /// Classic \(F_n \bmod 29\) (0-based index).
    [[nodiscard]] static Index29 nth(std::size_t index) noexcept {
        return nth_seeded(Index29{0}, Index29{1}, index);
    }

    /// First `count` classic terms. `count == 0` → empty.
    [[nodiscard]] static std::vector<Index29> first(std::size_t count) {
        return first_seeded(Index29{0}, Index29{1}, count);
    }

    /// Seeded stream: term 0 = `s0`, term 1 = `s1`, then sum mod 29.
    [[nodiscard]] static Index29 nth_seeded(Index29 s0, Index29 s1, std::size_t index) noexcept {
        if (index == 0) {
            return s0;
        }
        if (index == 1) {
            return s1;
        }
        Index29 a = s0;
        Index29 b = s1;
        for (std::size_t i = 2; i <= index; ++i) {
            const Index29 next = Z29::add(a, b);
            a = b;
            b = next;
        }
        return b;
    }

    /// First `count` seeded terms. `count == 0` → empty.
    [[nodiscard]] static std::vector<Index29> first_seeded(Index29 s0, Index29 s1,
                                                           std::size_t count) {
        std::vector<Index29> out;
        out.reserve(count);
        if (count == 0) {
            return out;
        }
        out.push_back(s0);
        if (count == 1) {
            return out;
        }
        out.push_back(s1);
        for (std::size_t i = 2; i < count; ++i) {
            out.push_back(Z29::add(out[i - 2], out[i - 1]));
        }
        return out;
    }

private:
    FibonacciStream() = delete;
};

#endif // FIBONACCI_STREAM_HPP
