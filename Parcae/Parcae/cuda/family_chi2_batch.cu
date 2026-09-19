#include "family_chi2_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

int FamilyChi2Batch::tiles_for(std::size_t token_count) {
    const int by_work = static_cast<int>(
        (token_count + static_cast<std::size_t>(HistFast::threads) - 1u) /
        static_cast<std::size_t>(HistFast::threads));
    constexpr int kMaxTiles = 1024;
    if (by_work < 1) {
        return 1;
    }
    return by_work < kMaxTiles ? by_work : kMaxTiles;
}

Status FamilyChi2Batch::clear_and_grid(
    std::uint32_t* device_counts,
    std::size_t candidate_count,
    std::size_t token_count,
    dim3* grid_out) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("FamilyChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("FamilyChi2Batch: bad T");
    }
    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared = CudaError::to_status(
        cudaMemsetAsync(device_counts, 0, hist_bytes, 0),
        "FamilyChi2Batch::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }
    *grid_out = dim3(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    return Status::success();
}

__global__ void atbash_chi2_hist_kernel(
    const std::uint8_t* in,
    std::uint32_t* counts,
    std::size_t token_count) {
    __shared__ std::uint32_t shared[HistFast::alphabet];
    HistFast::clear_shared(shared);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         i < n4;
         i += stride) {
        const uchar4 v = in4[i];
        atomicAdd(&shared[HistFast::dec_atbash(v.x)], 1u);
        atomicAdd(&shared[HistFast::dec_atbash(v.y)], 1u);
        atomicAdd(&shared[HistFast::dec_atbash(v.z)], 1u);
        atomicAdd(&shared[HistFast::dec_atbash(v.w)], 1u);
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        atomicAdd(&shared[HistFast::dec_atbash(in[t])], 1u);
    }
    __syncthreads();
    if (threadIdx.x < HistFast::alphabet) {
        atomicAdd(
            &counts[candidate * static_cast<std::size_t>(HistFast::alphabet) +
                    static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void atbash_caesar_chi2_hist_kernel(
    const std::uint8_t* in,
    const std::uint8_t* shifts,
    std::uint32_t* counts,
    std::size_t token_count) {
    __shared__ std::uint32_t shared[HistFast::alphabet];
    HistFast::clear_shared(shared);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = shifts[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         i < n4;
         i += stride) {
        const uchar4 v = in4[i];
        atomicAdd(&shared[HistFast::dec_caesar(HistFast::dec_atbash(v.x), shift)], 1u);
        atomicAdd(&shared[HistFast::dec_caesar(HistFast::dec_atbash(v.y), shift)], 1u);
        atomicAdd(&shared[HistFast::dec_caesar(HistFast::dec_atbash(v.z), shift)], 1u);
        atomicAdd(&shared[HistFast::dec_caesar(HistFast::dec_atbash(v.w), shift)], 1u);
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        atomicAdd(
            &shared[HistFast::dec_caesar(HistFast::dec_atbash(in[t]), shift)], 1u);
    }
    __syncthreads();
    if (threadIdx.x < HistFast::alphabet) {
        atomicAdd(
            &counts[candidate * static_cast<std::size_t>(HistFast::alphabet) +
                    static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void affine_chi2_hist_kernel(
    const std::uint8_t* in,
    const std::uint8_t* affine_a,
    const std::uint8_t* affine_b,
    std::uint32_t* counts,
    std::size_t token_count) {
    __shared__ std::uint32_t shared[HistFast::alphabet];
    HistFast::clear_shared(shared);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t inv_a = Z29Device::inv(affine_a[candidate]);
    const std::uint8_t b = affine_b[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         i < n4;
         i += stride) {
        const uchar4 v = in4[i];
        atomicAdd(&shared[Z29Device::mul(inv_a, Z29Device::sub(v.x, b))], 1u);
        atomicAdd(&shared[Z29Device::mul(inv_a, Z29Device::sub(v.y, b))], 1u);
        atomicAdd(&shared[Z29Device::mul(inv_a, Z29Device::sub(v.z, b))], 1u);
        atomicAdd(&shared[Z29Device::mul(inv_a, Z29Device::sub(v.w, b))], 1u);
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        atomicAdd(&shared[Z29Device::mul(inv_a, Z29Device::sub(in[t], b))], 1u);
    }
    __syncthreads();
    if (threadIdx.x < HistFast::alphabet) {
        atomicAdd(
            &counts[candidate * static_cast<std::size_t>(HistFast::alphabet) +
                    static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void vigenere_chi2_hist_kernel(
    const std::uint8_t* in,
    const std::uint8_t* key_bytes,
    const std::uint32_t* key_begin,
    const std::uint32_t* key_len,
    std::uint32_t* counts,
    std::size_t token_count) {
    __shared__ std::uint32_t shared[HistFast::alphabet];
    __shared__ std::uint8_t key_cache[64];
    HistFast::clear_shared(shared);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint32_t begin = key_begin[candidate];
    const std::uint32_t len = key_len[candidate];
    const std::uint32_t cached = len < 64u ? len : 64u;
    for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < cached;
         i += static_cast<std::uint32_t>(blockDim.x)) {
        key_cache[i] = key_bytes[begin + i];
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        const std::uint32_t ki = static_cast<std::uint32_t>(t) % len;
        const std::uint8_t key_symbol =
            ki < cached ? key_cache[ki] : key_bytes[begin + ki];
        atomicAdd(&shared[HistFast::dec_sub(in[t], key_symbol)], 1u);
    }
    __syncthreads();
    if (threadIdx.x < HistFast::alphabet) {
        atomicAdd(
            &counts[candidate * static_cast<std::size_t>(HistFast::alphabet) +
                    static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

Status FamilyChi2Batch::launch_atbash_async(
    const std::uint8_t* device_in,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (device_in == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::atbash null");
    }
    dim3 grid;
    Status prep = clear_and_grid(device_counts, candidate_count, token_count, &grid);
    if (!prep.ok()) {
        return prep;
    }
    atbash_chi2_hist_kernel<<<grid, HistFast::threads>>>(device_in, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::atbash hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status FamilyChi2Batch::launch_atbash_caesar_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (device_in == nullptr || device_shifts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::atbash_caesar null");
    }
    dim3 grid;
    Status prep = clear_and_grid(device_counts, candidate_count, token_count, &grid);
    if (!prep.ok()) {
        return prep;
    }
    atbash_caesar_chi2_hist_kernel<<<grid, HistFast::threads>>>(
        device_in, device_shifts, device_counts, token_count);
    Status hist =
        CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::atbash_caesar hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status FamilyChi2Batch::launch_affine_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_a,
    const std::uint8_t* device_b,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (device_in == nullptr || device_a == nullptr || device_b == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::affine null");
    }
    dim3 grid;
    Status prep = clear_and_grid(device_counts, candidate_count, token_count, &grid);
    if (!prep.ok()) {
        return prep;
    }
    affine_chi2_hist_kernel<<<grid, HistFast::threads>>>(
        device_in, device_a, device_b, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::affine hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status FamilyChi2Batch::launch_vigenere_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin,
    const std::uint32_t* device_key_len,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (device_in == nullptr || device_key_bytes == nullptr || device_key_begin == nullptr ||
        device_key_len == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::vigenere null");
    }
    dim3 grid;
    Status prep = clear_and_grid(device_counts, candidate_count, token_count, &grid);
    if (!prep.ok()) {
        return prep;
    }
    vigenere_chi2_hist_kernel<<<grid, HistFast::threads>>>(
        device_in,
        device_key_bytes,
        device_key_begin,
        device_key_len,
        device_counts,
        token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::vigenere hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}
