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

/// Fast Z/29 decrypt helpers + fused-χ² histogram primitives.
///
/// Hist accumulation paths (same final global counts when used correctly):
/// - **Warp-private** (`clear_private` / `add_private` / `flush_private`): shared
///   bins per warp with per-rune `atomicAdd` — legacy; compute-bound on sm_120.
/// - **Register-local + warp dump** (`clear_local_regs` / `add_local_regs` /
///   `flush_regs_via_warp`): `++` in registers during the hot loop, then ≤29
///   atomics into small warp-private shared + `flush_private`. **Preferred**
///   production climb path (keeps shared ≈1 KiB, occupancy-friendly).
/// - **Shared-row local** (`clear_local` / `add_local` / `flush_local`): 32 KiB
///   stage; correct but occupancy-hostile on sm_120 — keep for parity tests.
class HistFast {
public:
    static constexpr int alphabet = 29;
    static constexpr int threads = 256;
    static constexpr int warps = threads / 32; // 8
    static constexpr int priv_stride = 32;     // bins padded to 32 (warp path)
    /// Padded stride for per-thread local rows (bank-friendly; ≥ alphabet).
    static constexpr int local_stride = 32;
    /// `__shared__ uint32_t stage[local_shared_uints]` for the local path.
    static constexpr int local_shared_uints = threads * local_stride; // 8192
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
    // --- Warp-private shared hist (legacy) -----------------------------------

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

    // --- Thread-local shared hist (climb path) --------------------------------

    /// Zero this thread's local row. `stage` size = `local_shared_uints`.
    /// No cross-thread sync required before `add_local` (disjoint rows).
    PARCAE_D static void clear_local(std::uint32_t* stage) {
#pragma unroll
        for (int i = 0; i < alphabet; ++i) {
            stage[threadIdx.x * local_stride + i] = 0u;
        }
    }

    /// Increment bin `y` (must be `< alphabet`) in this thread's row — no atomics.
    PARCAE_D static void add_local(std::uint32_t* stage, std::uint8_t y) {
        stage[threadIdx.x * local_stride + static_cast<int>(y)] += 1u;
    }

    /// Reduce all thread rows into `global_row[0..alphabet)` via one atomic per bin.
    /// Callers must not race other threads still in `add_local` on the same stage.
    PARCAE_D static void flush_local(std::uint32_t* stage, std::uint32_t* global_row) {
        __syncthreads();
        if (threadIdx.x < alphabet) {
            std::uint32_t sum = 0u;
            const int bin = threadIdx.x;
#pragma unroll 8
            for (int t = 0; t < threads; ++t) {
                sum += stage[t * local_stride + bin];
            }
            atomicAdd(&global_row[bin], sum);
        }
    }

    /// Register-backed clear (hot-loop climb path).
    PARCAE_D static void clear_local_regs(std::uint32_t bins[alphabet]) {
#pragma unroll
        for (int i = 0; i < alphabet; ++i) {
            bins[i] = 0u;
        }
    }

    PARCAE_D static void add_local_regs(std::uint32_t bins[alphabet], std::uint8_t y) {
        bins[static_cast<int>(y)] += 1u;
    }

    /// Deposit register bins into warp-private shared (≤29 atomics / thread).
    /// `priv` must already be cleared (`clear_private`).
    PARCAE_D static void deposit_local_regs(const std::uint32_t bins[alphabet],
                                            std::uint32_t* priv) {
        const int warp = threadIdx.x >> 5;
#pragma unroll
        for (int i = 0; i < alphabet; ++i) {
            const std::uint32_t v = bins[i];
            if (v != 0u) {
                atomicAdd(&priv[warp * priv_stride + i], v);
            }
        }
    }

    /// Preferred flush after register accumulation: clear warp priv → deposit →
    /// flush_private. Shared footprint stays `warps * priv_stride` (~1 KiB).
    PARCAE_D static void flush_regs_via_warp(std::uint32_t bins[alphabet], std::uint32_t* priv,
                                             std::uint32_t* global_row) {
        clear_private(priv);
        deposit_local_regs(bins, priv);
        flush_private(priv, global_row);
    }

    /// Copy register bins into `stage`, then `flush_local` (32 KiB path).
    PARCAE_D static void flush_local_regs(std::uint32_t bins[alphabet], std::uint32_t* stage,
                                          std::uint32_t* global_row) {
#pragma unroll
        for (int i = 0; i < alphabet; ++i) {
            stage[threadIdx.x * local_stride + i] = bins[i];
        }
        flush_local(stage, global_row);
    }
#endif

private:
    HistFast() = delete;
};

#undef PARCAE_HD
#undef PARCAE_D

#endif // HIST_FAST_HPP
