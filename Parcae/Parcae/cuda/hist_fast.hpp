#ifndef HIST_FAST_HPP
#define HIST_FAST_HPP

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_HD __host__ __device__
#define PARCAE_D __device__
#else
#define PARCAE_HD
#define PARCAE_D
#endif

/// Fast Z/29 decrypt helpers + warp-privatized shared histogram.
class HistFast {
public:
    static constexpr int alphabet = 29;
    static constexpr int threads = 256;
    static constexpr int warps = threads / 32; // 8
    static constexpr int priv_stride = 32;     // bins padded to 32
    static constexpr int max_tiles = 1024;

    /// Grid.y for uchar4-first hist kernels: one tile covers `threads` packs.
    [[nodiscard]] static int tiles_for(std::size_t token_count) {
        const std::size_t packs = (token_count + 3u) / 4u; // scalar epilogue covers remainder
        const int by_work = static_cast<int>((packs + static_cast<std::size_t>(threads) - 1u) /
                                             static_cast<std::size_t>(threads));
        if (by_work < 1) {
            return 1;
        }
        return by_work < max_tiles ? by_work : max_tiles;
    }
    [[nodiscard]] PARCAE_HD static std::uint8_t dec_caesar(std::uint8_t x,
                                                           std::uint8_t shift) noexcept {
        const unsigned s = static_cast<unsigned>(x) + 29u - static_cast<unsigned>(shift);
        return static_cast<std::uint8_t>(s >= 29u ? s - 29u : s);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t enc_caesar(std::uint8_t x,
                                                           std::uint8_t shift) noexcept {
        const unsigned s = static_cast<unsigned>(x) + static_cast<unsigned>(shift);
        return static_cast<std::uint8_t>(s >= 29u ? s - 29u : s);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t dec_atbash(std::uint8_t x) noexcept {
        return static_cast<std::uint8_t>(28u - x);
    }

    [[nodiscard]] PARCAE_HD static std::uint8_t dec_sub(std::uint8_t x, std::uint8_t key) noexcept {
        const unsigned s = static_cast<unsigned>(x) + 29u - static_cast<unsigned>(key);
        return static_cast<std::uint8_t>(s >= 29u ? s - 29u : s);
    }

#if defined(__CUDACC__)
    /// `priv` must be `warps * priv_stride` uint32 in shared memory.
    PARCAE_D static void clear_private(std::uint32_t* priv) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        if (lane < alphabet) {
            priv[warp * priv_stride + lane] = 0u;
        }
        __syncthreads();
    }

    PARCAE_D static void add_private(std::uint32_t* priv, std::uint8_t y) {
        atomicAdd(&priv[(threadIdx.x >> 5) * priv_stride + y], 1u);
    }

    PARCAE_D static void flush_private(std::uint32_t* priv, std::uint32_t* global_row) {
        __syncthreads();
        if (threadIdx.x < alphabet) {
            std::uint32_t sum = 0u;
#pragma unroll
            for (int w = 0; w < warps; ++w) {
                sum += priv[w * priv_stride + threadIdx.x];
            }
            atomicAdd(&global_row[threadIdx.x], sum);
        }
    }
#endif

private:
    HistFast() = delete;
};

#undef PARCAE_HD
#undef PARCAE_D

#endif // HIST_FAST_HPP
