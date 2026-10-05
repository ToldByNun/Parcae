#include "theory_hist_chi2_s5.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

/// File-scope — no anonymous namespace.
__global__ void theory_hist_chi2_s5_poly_kernel(const std::uint8_t* in, const std::uint8_t* b0,
                                                const std::uint8_t* b1, const std::uint8_t* b2,
                                                std::uint32_t* counts, std::size_t token_count,
                                                std::uint8_t cipher_minus_ks) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    __shared__ std::uint8_t ks[HistFast::alphabet];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t pb0 = b0[candidate];
    const std::uint8_t pb1 = b1[candidate];
    const std::uint8_t pb2 = b2[candidate];

    if (threadIdx.x < HistFast::alphabet) {
        const std::uint8_t i = static_cast<std::uint8_t>(threadIdx.x);
        const std::uint8_t i2 = Z29Device::mul(i, i);
        ks[threadIdx.x] =
            Z29Device::add(pb0, Z29Device::add(Z29Device::mul(pb1, i), Z29Device::mul(pb2, i2)));
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);
    constexpr std::size_t mod = static_cast<std::size_t>(Z29Device::modulus);

    auto out_byte = [&](std::uint8_t x, std::uint8_t key) -> std::uint8_t {
        return cipher_minus_ks != 0u ? HistFast::dec_sub(x, key) : HistFast::enc_caesar(x, key);
    };

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = in4[i];
        const std::size_t t0 = i * 4u;
        HistFast::add_private(priv, out_byte(v.x, ks[t0 % mod]));
        HistFast::add_private(priv, out_byte(v.y, ks[(t0 + 1u) % mod]));
        HistFast::add_private(priv, out_byte(v.z, ks[(t0 + 2u) % mod]));
        HistFast::add_private(priv, out_byte(v.w, ks[(t0 + 3u) % mod]));
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, out_byte(in[t], ks[t % mod]));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

int TheoryHistChi2S5::tiles_for(std::size_t token_count) {
    return HistFast::tiles_for(token_count);
}

Status TheoryHistChi2S5::launch_poly_async(
    const std::uint8_t* device_in, const std::uint8_t* device_b0, const std::uint8_t* device_b1,
    const std::uint8_t* device_b2, const double* device_probabilities, std::uint32_t* device_counts,
    double* device_scores, std::size_t candidate_count, std::size_t token_count,
    bool cipher_minus_ks, cudaStream_t stream) {
    if (device_in == nullptr || device_b0 == nullptr || device_b1 == nullptr ||
        device_b2 == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("TheoryHistChi2S5::launch_poly_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S5: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2S5: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2S5::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    theory_hist_chi2_s5_poly_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_b0, device_b1, device_b2, device_counts, token_count,
        cipher_minus_ks ? 1u : 0u);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S5::hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}
