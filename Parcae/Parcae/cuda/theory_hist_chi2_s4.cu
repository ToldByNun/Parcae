#include "theory_hist_chi2_s4.hpp"

#include "autokey_ring_device.hpp"
#include "chi2_batch_score.hpp"
#include "cipher_hist_once.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "hist_tile_cap.hpp"
#include "lag_diff_hist_once.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <vector>

/// File-scope — no anonymous namespace (theory hist emit contract).
/// Cipher loaded as uchar4; keys remain lag-dependent (AutokeyRing). When the
/// four-pack is fully past the primer (`t0 >= lag`), keys are four consecutive
/// prior-stream bytes and use `__ldg` (still no cross-pack inventing).
__global__ void theory_hist_chi2_s4_autokey_kernel(const std::uint8_t* __restrict__ in,
                                                   const std::uint8_t* __restrict__ lags,
                                                   std::uint32_t* __restrict__ counts,
                                                   std::size_t token_count,
                                                   std::uint8_t cipher_minus_ks) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t lag = __ldg(lags + candidate);

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);

    auto out_byte = [&](std::uint8_t x, std::uint8_t key) -> std::uint8_t {
        return cipher_minus_ks != 0u ? HistFast::dec_sub(x, key) : HistFast::enc_caesar(x, key);
    };

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        const std::size_t t0 = i * 4u;
        if (lag != 0u && t0 >= static_cast<std::size_t>(lag)) {
            // Fully past primer: keys = in[t0-lag .. t0-lag+3] (consecutive).
            const std::size_t kb = t0 - static_cast<std::size_t>(lag);
            const std::uint8_t k0 = __ldg(in + kb);
            const std::uint8_t k1 = __ldg(in + kb + 1u);
            const std::uint8_t k2 = __ldg(in + kb + 2u);
            const std::uint8_t k3 = __ldg(in + kb + 3u);
            HistFast::add_private(priv, out_byte(v.x, k0));
            HistFast::add_private(priv, out_byte(v.y, k1));
            HistFast::add_private(priv, out_byte(v.z, k2));
            HistFast::add_private(priv, out_byte(v.w, k3));
        } else {
            // Primer / mixed pack — keep ring semantics per index.
            HistFast::add_private(priv, out_byte(v.x, AutokeyRingDevice::shift(in, t0, lag)));
            HistFast::add_private(priv,
                                  out_byte(v.y, AutokeyRingDevice::shift(in, t0 + 1u, lag)));
            HistFast::add_private(priv,
                                  out_byte(v.z, AutokeyRingDevice::shift(in, t0 + 2u, lag)));
            HistFast::add_private(priv,
                                  out_byte(v.w, AutokeyRingDevice::shift(in, t0 + 3u, lag)));
        }
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(
            priv, out_byte(__ldg(in + t), AutokeyRingDevice::shift(in, t, lag)));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

/// Broadcast merged `lag + prefix` hist to candidates with `lags[c] == period_filter`.
__global__ void theory_hist_chi2_s4_ring_remap_kernel(
    const std::uint32_t* __restrict__ lag_hist, const std::uint32_t* __restrict__ prefix_hist,
    const std::uint8_t* __restrict__ lags, std::uint32_t* __restrict__ counts,
    std::size_t candidate_count, std::uint8_t period_filter) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }
    if (__ldg(lags + c) != period_filter) {
        return;
    }

    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int b = 0; b < HistFast::alphabet; ++b) {
        row[b] = __ldg(lag_hist + b) + __ldg(prefix_hist + b);
    }
}

int TheoryHistChi2S4::tiles_for(std::size_t token_count) {
    return HistTileCap::tiles_for(HistTileCap::kS4, token_count);
}

int TheoryHistChi2S4::tiles_for_public(std::size_t token_count) {
    return tiles_for(token_count);
}

Status TheoryHistChi2S4::ensure_lag_scratch(std::uint32_t** out_lag) {
    static std::uint32_t* device_lag = nullptr;
    if (device_lag == nullptr) {
        Status allocated = CudaError::to_status(
            cudaMalloc(reinterpret_cast<void**>(&device_lag),
                       alphabet_size * sizeof(std::uint32_t)),
            "TheoryHistChi2S4::lag scratch");
        if (!allocated.ok()) {
            device_lag = nullptr;
            return allocated;
        }
    }
    *out_lag = device_lag;
    return Status::success();
}

Status TheoryHistChi2S4::ensure_prefix_scratch(std::uint32_t** out_prefix) {
    static std::uint32_t* device_prefix = nullptr;
    if (device_prefix == nullptr) {
        Status allocated = CudaError::to_status(
            cudaMalloc(reinterpret_cast<void**>(&device_prefix),
                       alphabet_size * sizeof(std::uint32_t)),
            "TheoryHistChi2S4::prefix scratch");
        if (!allocated.ok()) {
            device_prefix = nullptr;
            return allocated;
        }
    }
    *out_prefix = device_prefix;
    return Status::success();
}

Status TheoryHistChi2S4::launch_autokey_decode_hist_async(
    const std::uint8_t* device_in, const std::uint8_t* device_lags,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
    cudaStream_t stream) {
    if (device_in == nullptr || device_lags == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S4::launch_autokey_decode_hist_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S4: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2S4: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2S4::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    theory_hist_chi2_s4_autokey_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_lags, device_counts, token_count, cipher_minus_ks ? 1u : 0u);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S4::hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status TheoryHistChi2S4::launch_autokey_remap_async(
    const std::uint8_t* device_in, const std::uint8_t* device_lags,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
    cudaStream_t stream) {
    if (device_in == nullptr || device_lags == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S4::launch_autokey_remap_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S4: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2S4: bad T");
    }

    std::uint32_t* lag_hist = nullptr;
    std::uint32_t* prefix_hist = nullptr;
    Status lag_scratch = ensure_lag_scratch(&lag_hist);
    if (!lag_scratch.ok()) {
        return lag_scratch;
    }
    Status prefix_scratch = ensure_prefix_scratch(&prefix_hist);
    if (!prefix_scratch.ok()) {
        return prefix_scratch;
    }

    std::vector<std::uint8_t> host_lags(candidate_count);
    Status copied = CudaError::to_status(
        cudaMemcpyAsync(host_lags.data(), device_lags, candidate_count * sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost, stream),
        "TheoryHistChi2S4::lags D2H");
    if (!copied.ok()) {
        return copied;
    }
    Status synced = CudaError::to_status(
        stream == nullptr ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream),
        "TheoryHistChi2S4::lags sync");
    if (!synced.ok()) {
        return synced;
    }

    std::vector<std::uint8_t> unique_lags;
    unique_lags.reserve(8);
    for (std::size_t i = 0; i < candidate_count; ++i) {
        const std::uint8_t L = host_lags[i];
        if (std::find(unique_lags.begin(), unique_lags.end(), L) == unique_lags.end()) {
            unique_lags.push_back(L);
        }
    }

    constexpr int threads = 128;
    const int blocks =
        static_cast<int>((candidate_count + static_cast<std::size_t>(threads) - 1u) /
                         static_cast<std::size_t>(threads));

    for (std::uint8_t lag : unique_lags) {
        if (lag == 0u) {
            // AutokeyRingDevice::shift(..., 0) ≡ 0 → full cipher hist; empty lag.
            Status zero_lag = CudaError::to_status(
                cudaMemsetAsync(lag_hist, 0, alphabet_size * sizeof(std::uint32_t), stream),
                "TheoryHistChi2S4::zero lag for L=0");
            if (!zero_lag.ok()) {
                return zero_lag;
            }
            Status prefix_ok =
                CipherHistOnce::launch_async(device_in, prefix_hist, token_count, stream);
            if (!prefix_ok.ok()) {
                return prefix_ok;
            }
        } else {
            const std::uint32_t lag32 = lag;
            Status lag_ok =
                cipher_minus_ks
                    ? LagDiffHistOnce::launch_async(device_in, lag_hist, token_count, lag32, stream)
                    : LagDiffHistOnce::launch_sum_async(device_in, lag_hist, token_count, lag32,
                                                        stream);
            if (!lag_ok.ok()) {
                return lag_ok;
            }
            const std::size_t prefix_len =
                static_cast<std::size_t>(lag) < token_count ? static_cast<std::size_t>(lag)
                                                            : token_count;
            Status prefix_ok =
                CipherHistOnce::launch_async(device_in, prefix_hist, prefix_len, stream);
            if (!prefix_ok.ok()) {
                return prefix_ok;
            }
        }

        theory_hist_chi2_s4_ring_remap_kernel<<<blocks, threads, 0, stream>>>(
            lag_hist, prefix_hist, device_lags, device_counts, candidate_count, lag);
        Status remap =
            CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S4::ring remap");
        if (!remap.ok()) {
            return remap;
        }
    }

    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status TheoryHistChi2S4::launch_autokey_async(
    const std::uint8_t* device_in, const std::uint8_t* device_lags,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
    cudaStream_t stream) {
    return launch_autokey_remap_async(device_in, device_lags, device_probabilities, device_counts,
                                      device_scores, candidate_count, token_count, cipher_minus_ks,
                                      stream);
}
