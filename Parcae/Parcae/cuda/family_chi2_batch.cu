#include "family_chi2_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kHistThreads = 256;
constexpr std::uint8_t kModulus = 29;

[[nodiscard]] int tiles_for(std::size_t token_count) {
    const int by_work = static_cast<int>(
        (token_count + static_cast<std::size_t>(kHistThreads) - 1u) /
        static_cast<std::size_t>(kHistThreads));
    constexpr int kMaxTiles = 512;
    if (by_work < 1) {
        return 1;
    }
    return by_work < kMaxTiles ? by_work : kMaxTiles;
}

[[nodiscard]] Status clear_and_grid(
    unsigned long long* device_counts,
    std::size_t candidate_count,
    std::size_t token_count,
    dim3* grid_out) {
    if (candidate_count == 0 || candidate_count > FamilyChi2Batch::kMaxCandidates) {
        return Status::error("FamilyChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > FamilyChi2Batch::kMaxTokens) {
        return Status::error("FamilyChi2Batch: bad T");
    }
    const std::size_t hist_bytes =
        candidate_count * FamilyChi2Batch::alphabet_size * sizeof(unsigned long long);
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

[[nodiscard]] Status finish_scores(
    unsigned long long* device_counts,
    const double* device_probabilities,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    return Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

__global__ void atbash_chi2_hist_kernel(
    const std::uint8_t* in,
    unsigned long long* counts,
    std::size_t token_count) {
    __shared__ unsigned long long shared[kModulus];
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    if (threadIdx.x < static_cast<int>(kModulus)) {
        shared[threadIdx.x] = 0ull;
    }
    __syncthreads();
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        const std::uint8_t y = static_cast<std::uint8_t>(28u - in[t]);
        atomicAdd(&shared[y], 1ull);
    }
    __syncthreads();
    if (threadIdx.x < static_cast<int>(kModulus)) {
        atomicAdd(
            &counts[candidate * kModulus + static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void atbash_caesar_chi2_hist_kernel(
    const std::uint8_t* in,
    const std::uint8_t* shifts,
    unsigned long long* counts,
    std::size_t token_count) {
    __shared__ unsigned long long shared[kModulus];
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    if (threadIdx.x < static_cast<int>(kModulus)) {
        shared[threadIdx.x] = 0ull;
    }
    __syncthreads();
    const std::uint8_t shift = shifts[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        // decrypt atbash∘caesar: (28 - x + shift) % 29
        const std::uint8_t y =
            static_cast<std::uint8_t>((28u - in[t] + shift) % kModulus);
        atomicAdd(&shared[y], 1ull);
    }
    __syncthreads();
    if (threadIdx.x < static_cast<int>(kModulus)) {
        atomicAdd(
            &counts[candidate * kModulus + static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void affine_chi2_hist_kernel(
    const std::uint8_t* in,
    const std::uint8_t* affine_a,
    const std::uint8_t* affine_b,
    unsigned long long* counts,
    std::size_t token_count) {
    __shared__ unsigned long long shared[kModulus];
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    if (threadIdx.x < static_cast<int>(kModulus)) {
        shared[threadIdx.x] = 0ull;
    }
    __syncthreads();
    const std::uint8_t inv_a = Z29Device::inv(affine_a[candidate]);
    const std::uint8_t b = affine_b[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        const std::uint8_t y = Z29Device::mul(inv_a, Z29Device::sub(in[t], b));
        atomicAdd(&shared[y], 1ull);
    }
    __syncthreads();
    if (threadIdx.x < static_cast<int>(kModulus)) {
        atomicAdd(
            &counts[candidate * kModulus + static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void vigenere_chi2_hist_kernel(
    const std::uint8_t* in,
    const std::uint8_t* key_bytes,
    const std::uint32_t* key_begin,
    const std::uint32_t* key_len,
    unsigned long long* counts,
    std::size_t token_count) {
    __shared__ unsigned long long shared[kModulus];
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    if (threadIdx.x < static_cast<int>(kModulus)) {
        shared[threadIdx.x] = 0ull;
    }
    __syncthreads();
    const std::uint32_t begin = key_begin[candidate];
    const std::uint32_t len = key_len[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        const std::uint8_t key_symbol = key_bytes[begin + (static_cast<std::uint32_t>(t) % len)];
        const std::uint8_t y = Z29Device::sub(in[t], key_symbol);
        atomicAdd(&shared[y], 1ull);
    }
    __syncthreads();
    if (threadIdx.x < static_cast<int>(kModulus)) {
        atomicAdd(
            &counts[candidate * kModulus + static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

}  // namespace

Status FamilyChi2Batch::launch_atbash_async(
    const std::uint8_t* device_in,
    const double* device_probabilities,
    unsigned long long* device_counts,
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
    atbash_chi2_hist_kernel<<<grid, kHistThreads>>>(device_in, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::atbash hist");
    if (!hist.ok()) {
        return hist;
    }
    return finish_scores(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status FamilyChi2Batch::launch_atbash_caesar_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const double* device_probabilities,
    unsigned long long* device_counts,
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
    atbash_caesar_chi2_hist_kernel<<<grid, kHistThreads>>>(
        device_in, device_shifts, device_counts, token_count);
    Status hist =
        CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::atbash_caesar hist");
    if (!hist.ok()) {
        return hist;
    }
    return finish_scores(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status FamilyChi2Batch::launch_affine_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_a,
    const std::uint8_t* device_b,
    const double* device_probabilities,
    unsigned long long* device_counts,
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
    affine_chi2_hist_kernel<<<grid, kHistThreads>>>(
        device_in, device_a, device_b, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::affine hist");
    if (!hist.ok()) {
        return hist;
    }
    return finish_scores(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status FamilyChi2Batch::launch_vigenere_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin,
    const std::uint32_t* device_key_len,
    const double* device_probabilities,
    unsigned long long* device_counts,
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
    vigenere_chi2_hist_kernel<<<grid, kHistThreads>>>(
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
    return finish_scores(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}
