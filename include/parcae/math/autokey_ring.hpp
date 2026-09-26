#ifndef AUTOKEY_RING_HPP
#define AUTOKEY_RING_HPP

#include "parcae/core/index29.hpp"

#include <cstddef>
#include <span>

/// Host lag / ringbuffer read matching `AutokeyRingDevice::shift` (DSL `z29_autokey_shift`).
class AutokeyRing {
public:
    /// Primer-less v0: `i < lag` → 0; else `stream[i - lag]`.
    [[nodiscard]] static Index29 shift(std::span<const Index29> stream, std::size_t i,
                                       Index29 lag) noexcept {
        const std::uint8_t L = lag.value();
        if (L == 0u || i < static_cast<std::size_t>(L) || i >= stream.size()) {
            return Index29{0};
        }
        return stream[i - static_cast<std::size_t>(L)];
    }

private:
    AutokeyRing() = delete;
};

#endif // AUTOKEY_RING_HPP
