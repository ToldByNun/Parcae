#include "chi2_batch_score.hpp"

#include "cuda_error.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

__global__ void chi2_hist_from_out_kernel(
    const std::uint8_t* out,
    std::uint32_t* counts,
    std::size_t token_count) {
    __shared__ std::uint32_t shared[HistFast::alphabet];
    HistFast::clear_shared(shared);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::uint8_t* lane = out + candidate * token_count;

    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        atomicAdd(&shared[lane[t]], 1u);
    }
    __syncthreads();
    if (threadIdx.x < HistFast::alphabet) {
        atomicAdd(
            &counts[candidate * static_cast<std::size_t>(HistFast::alphabet) +
                    static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void chi2_finalize_kernel(
    const std::uint32_t* counts,
    const double* probabilities,
    double* scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }

    const double n_d = static_cast<double>(token_count);
    double chi2 = 0.0;
    const std::uint32_t* obs = counts + c * HistFast::alphabet;
    #pragma unroll
    for (int i = 0; i < HistFast::alphabet; ++i) {
        const double e = probabilities[i] * n_d;
        const double diff = static_cast<double>(obs[i]) - e;
        chi2 += (diff * diff) / e;
    }
    scores[c] = chi2;
}

int Chi2BatchScore::tiles_for(std::size_t token_count) {
    const int by_work = static_cast<int>(
        (token_count + static_cast<std::size_t>(HistFast::threads) - 1u) /
        static_cast<std::size_t>(HistFast::threads));
    constexpr int kMaxTiles = 1024;
    if (by_work < 1) {
        return 1;
    }
    return by_work < kMaxTiles ? by_work : kMaxTiles;
}

Status Chi2BatchScore::histogram_from_out_async(
    const std::uint8_t* device_out,
    std::uint32_t* device_counts,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("Chi2BatchScore::histogram_from_out_async bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("Chi2BatchScore::histogram_from_out_async bad T");
    }
    if (device_out == nullptr || device_counts == nullptr) {
        return Status::error("Chi2BatchScore::histogram_from_out_async null");
    }

    const dim3 grid(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    chi2_hist_from_out_kernel<<<grid, HistFast::threads>>>(
        device_out, device_counts, token_count);
    return CudaError::to_status(cudaGetLastError(), "Chi2BatchScore::histogram_from_out");
}

Status Chi2BatchScore::finalize_async(
    const std::uint32_t* device_counts,
    const double* device_probabilities,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("Chi2BatchScore::finalize_async bad C");
    }
    if (device_counts == nullptr || device_probabilities == nullptr || device_scores == nullptr) {
        return Status::error("Chi2BatchScore::finalize_async null");
    }

    const int threads = 128;
    const int blocks = static_cast<int>(
        (candidate_count + static_cast<std::size_t>(threads) - 1u) /
        static_cast<std::size_t>(threads));
    chi2_finalize_kernel<<<blocks, threads>>>(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
    return CudaError::to_status(cudaGetLastError(), "Chi2BatchScore::finalize");
}

Status Chi2BatchScore::score_from_out_async(
    const std::uint8_t* device_out,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared = CudaError::to_status(
        cudaMemsetAsync(device_counts, 0, hist_bytes, 0),
        "Chi2BatchScore::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }
    Status hist =
        histogram_from_out_async(device_out, device_counts, candidate_count, token_count);
    if (!hist.ok()) {
        return hist;
    }
    return finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}
