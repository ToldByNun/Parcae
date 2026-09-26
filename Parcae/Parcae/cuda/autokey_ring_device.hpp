#ifndef AUTOKEY_RING_DEVICE_HPP
#define AUTOKEY_RING_DEVICE_HPP

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_RING_HD __host__ __device__
#else
#define PARCAE_RING_HD
#endif

/// Dense autokey lag / ringbuffer read for DSL `z29_autokey_shift(stream, lag)`.
///
/// Primer-less v0: when `i < lag` returns 0; otherwise `stream[i - lag]`.
/// Matches CTAK/PTAK prior-stream keying once the primer window has passed
/// (primer symbols themselves are supplied separately by the theory).
class AutokeyRingDevice {
public:
    [[nodiscard]] PARCAE_RING_HD static std::uint8_t
    shift(const std::uint8_t* stream, std::size_t i, std::uint8_t lag) noexcept {
        if (stream == nullptr || lag == 0u) {
            return 0u;
        }
        if (i < static_cast<std::size_t>(lag)) {
            return 0u;
        }
        return stream[i - static_cast<std::size_t>(lag)];
    }

private:
    AutokeyRingDevice() = delete;
};

#undef PARCAE_RING_HD

#endif // AUTOKEY_RING_DEVICE_HPP
