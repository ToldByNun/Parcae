#include "theory_hist_chi2_s2.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

/// File-scope — no anonymous namespace (theory hist emit contract).
__global__ void theory_hist_chi2_s2_linear_kernel(const std::uint8_t* in, const std::uint8_t* b0,
                                                  const std::uint8_t* b1, std::uint32_t* counts,
                                                  std::size_t token_count,
                                                  std::uint8_t cipher_minus_ks) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t pb0 = b0[candidate];
    const std::uint8_t pb1 = b1[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    auto ks_at = [&](std::size_t t) -> std::uint8_t {
        const std::uint8_t im = static_cast<std::uint8_t>(t % static_cast<std::size_t>(Z29Device::modulus));
        return Z29Device::add(pb0, Z29Device::mul(pb1, im));
    };
    auto out_byte = [&](std::uint8_t x, std::uint8_t ks) -> std::uint8_t {
        return cipher_minus_ks != 0u ? HistFast::dec_sub(x, ks) : HistFast::enc_caesar(x, ks);
    };

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = in4[i];
        const std::size_t t0 = i * 4u;
        HistFast::add_private(priv, out_byte(v.x, ks_at(t0)));
        HistFast::add_private(priv, out_byte(v.y, ks_at(t0 + 1u)));
        HistFast::add_private(priv, out_byte(v.z, ks_at(t0 + 2u)));
        HistFast::add_private(priv, out_byte(v.w, ks_at(t0 + 3u)));
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, out_byte(in[t], ks_at(t)));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

int TheoryHistChi2S2::tiles_for(std::size_t token_count) {
    // Commit 11: same fat-64 clamp as Caesar (HistFast::production_tile_cap).
    return HistFast::tiles_for(token_count);
}

Status TheoryHistChi2S2::launch_linear_async(
    const std::uint8_t* device_in, const std::uint8_t* device_b0, const std::uint8_t* device_b1,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
    cudaStream_t stream) {
    if (device_in == nullptr || device_b0 == nullptr || device_b1 == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S2::launch_linear_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S2: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2S2: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2S2::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    theory_hist_chi2_s2_linear_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_b0, device_b1, device_counts, token_count,
        cipher_minus_ks ? 1u : 0u);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S2::hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}
