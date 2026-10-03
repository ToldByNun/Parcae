#include "theory_hist_chi2_s1.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

/// File-scope — no anonymous namespace (theory hist emit contract).
__global__ void theory_hist_chi2_s1_lut_kernel(const std::uint8_t* in, const std::uint8_t* luts,
                                               std::uint32_t* counts, std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    __shared__ std::uint8_t lut[HistFast::alphabet];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t* row = luts + candidate * static_cast<std::size_t>(HistFast::alphabet);
    if (threadIdx.x < HistFast::alphabet) {
        lut[threadIdx.x] = row[threadIdx.x];
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = in4[i];
        HistFast::add_private(priv, lut[v.x]);
        HistFast::add_private(priv, lut[v.y]);
        HistFast::add_private(priv, lut[v.z]);
        HistFast::add_private(priv, lut[v.w]);
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, lut[in[t]]);
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

int TheoryHistChi2S1::tiles_for(std::size_t token_count) {
    return HistFast::tiles_for(token_count);
}

Status TheoryHistChi2S1::launch_lut_async(const std::uint8_t* device_in,
                                          const std::uint8_t* device_luts,
                                          const double* device_probabilities,
                                          std::uint32_t* device_counts, double* device_scores,
                                          std::size_t candidate_count, std::size_t token_count,
                                          cudaStream_t stream) {
    if (device_in == nullptr || device_luts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S1::launch_lut_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S1: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2S1: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2S1::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    theory_hist_chi2_s1_lut_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_luts, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S1::hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}
