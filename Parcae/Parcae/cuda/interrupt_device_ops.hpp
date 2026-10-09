#ifndef INTERRUPT_DEVICE_OPS_HPP
#define INTERRUPT_DEVICE_OPS_HPP

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)

/// Device helpers for `InterruptDeviceView` encodings (bitmask / sorted skips).
/// Shared by single-stream key kernels (Vigenère / Beaufort / Totient / CTAK).
class InterruptDeviceOps {
public:
    [[nodiscard]] __device__ static bool bitmask_should_skip(const std::uint32_t* words,
                                                             std::size_t index) {
        return (words[index >> 5] & (1u << (index & 31u))) != 0u;
    }

    /// Non-skip count in `[0, index)` for bitmask encoding.
    [[nodiscard]] __device__ static std::uint32_t
    bitmask_consumed_before(const std::uint32_t* words, std::size_t index) {
        std::uint32_t consumed = 0;
        const std::size_t full_words = index >> 5;
        for (std::size_t w = 0; w < full_words; ++w) {
            consumed += 32u - static_cast<std::uint32_t>(__popc(words[w]));
        }
        const std::uint32_t rem = static_cast<std::uint32_t>(index & 31u);
        if (rem != 0u) {
            const std::uint32_t mask = (1u << rem) - 1u;
            consumed += rem - static_cast<std::uint32_t>(__popc(words[full_words] & mask));
        }
        return consumed;
    }

    [[nodiscard]] __device__ static std::uint32_t
    sorted_skips_before(const std::uint32_t* skips, std::uint32_t skip_count, std::uint32_t index) {
        std::uint32_t lo = 0;
        std::uint32_t hi = skip_count;
        while (lo < hi) {
            const std::uint32_t mid = lo + (hi - lo) / 2u;
            if (skips[mid] < index) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    [[nodiscard]] __device__ static bool sorted_should_skip(const std::uint32_t* skips,
                                                           std::uint32_t skip_count,
                                                           std::uint32_t index) {
        const std::uint32_t lo = sorted_skips_before(skips, skip_count, index);
        return lo < skip_count && skips[lo] == index;
    }

private:
    InterruptDeviceOps() = delete;
};

#endif // __CUDACC__

#endif // INTERRUPT_DEVICE_OPS_HPP
