#ifndef Z29_DEVICE_HPP
#define Z29_DEVICE_HPP

#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_HD __host__ __device__
#else
#define PARCAE_HD
#endif

/// Device/host \\(\\mathbb{Z}_{29}\\) ops matching CPU `Z29` (Index29 as `uint8_t`).
/// Include from `.cu` for kernels; host builds may include for parity checks.
class Z29Device {
public:
    static constexpr std::uint8_t modulus = 29;

    [[nodiscard]] PARCAE_HD static std::uint8_t add(std::uint8_t x, std::uint8_t y) noexcept {
        const unsigned s = static_cast<unsigned>(x) + static_cast<unsigned>(y);
        return static_cast<std::uint8_t>(s >= modulus ? s - modulus : s);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t neg(std::uint8_t x) noexcept {
        return x == 0 ? static_cast<std::uint8_t>(0)
                      : static_cast<std::uint8_t>(modulus - x);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t sub(std::uint8_t x, std::uint8_t y) noexcept {
        const unsigned s = static_cast<unsigned>(x) + modulus - static_cast<unsigned>(y);
        return static_cast<std::uint8_t>(s >= modulus ? s - modulus : s);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t mul(std::uint8_t x, std::uint8_t y) noexcept {
        // Product < 841; NVCC lowers `% 29` to a mul-high reciprocal.
        return static_cast<std::uint8_t>(
            (static_cast<unsigned>(x) * static_cast<unsigned>(y)) % modulus);
    }

    /// Modular inverse for `a` in 1..28. Index 0 is unused (do not call with 0).
    [[nodiscard]] PARCAE_HD static std::uint8_t inv(std::uint8_t a) noexcept {
        // Must match `Z29::inv` / CPU inv_table for 1..28.
        constexpr std::uint8_t inv_table[modulus] = {
            0,  1,  15, 10, 22, 6,  5,  25, 11, 13, 3,  8,  17, 9,  27,
            2,  20, 12, 21, 26, 16, 18, 4,  24, 23, 7,  19, 14, 28};
        return inv_table[a];
    }

private:
    Z29Device() = delete;
};

#undef PARCAE_HD

#endif // Z29_DEVICE_HPP
