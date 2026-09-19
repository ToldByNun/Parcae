#ifndef HIST_FAST_HPP
#define HIST_FAST_HPP

#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_HD __host__ __device__
#define PARCAE_D __device__
#else
#define PARCAE_HD
#define PARCAE_D
#endif

/// Fast Z/29 decrypt helpers + shared-hist clear (device hot path).
class HistFast {
public:
    static constexpr int alphabet = 29;
    static constexpr int threads = 256;

    [[nodiscard]] PARCAE_HD static std::uint8_t dec_caesar(
        std::uint8_t x, std::uint8_t shift) noexcept {
        const unsigned s = static_cast<unsigned>(x) + 29u - static_cast<unsigned>(shift);
        return static_cast<std::uint8_t>(s >= 29u ? s - 29u : s);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t enc_caesar(
        std::uint8_t x, std::uint8_t shift) noexcept {
        const unsigned s = static_cast<unsigned>(x) + static_cast<unsigned>(shift);
        return static_cast<std::uint8_t>(s >= 29u ? s - 29u : s);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t dec_atbash(std::uint8_t x) noexcept {
        return static_cast<std::uint8_t>(28u - x);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t dec_sub(
        std::uint8_t x, std::uint8_t key) noexcept {
        const unsigned s = static_cast<unsigned>(x) + 29u - static_cast<unsigned>(key);
        return static_cast<std::uint8_t>(s >= 29u ? s - 29u : s);
    }

#if defined(__CUDACC__)
    PARCAE_D static void clear_shared(std::uint32_t* shared) {
        if (threadIdx.x < alphabet) {
            shared[threadIdx.x] = 0u;
        }
        __syncthreads();
    }
#endif

private:
    HistFast() = delete;
};

#undef PARCAE_HD
#undef PARCAE_D

#endif  // HIST_FAST_HPP
