#include "theory_hist_chi2_s4.hpp"

#include "autokey_ring_device.hpp"
#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "hist_tile_cap.hpp"

#include <cuda_runtime_api.h>

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

int TheoryHistChi2S4::tiles_for(std::size_t token_count) {
    return HistTileCap::tiles_for(HistTileCap::kS4, token_count);
}

int TheoryHistChi2S4::tiles_for_public(std::size_t token_count) {
    return tiles_for(token_count);
}

Status TheoryHistChi2S4::launch_autokey_async(
    const std::uint8_t* device_in, const std::uint8_t* device_lags,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
    cudaStream_t stream) {
    if (device_in == nullptr || device_lags == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S4::launch_autokey_async null");
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
