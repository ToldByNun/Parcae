#include "caesar_chi2_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"

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

[[nodiscard]] Status validate(
    std::size_t candidate_count,
    std::size_t token_count,
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    unsigned long long* device_counts,
    double* device_scores) {
    if (candidate_count == 0 || candidate_count > CaesarChi2Batch::kMaxCandidates) {
        return Status::error("CaesarChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > CaesarChi2Batch::kMaxTokens) {
        return Status::error("CaesarChi2Batch: bad T");
    }
    if (device_in == nullptr || device_shifts == nullptr || device_directions == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("CaesarChi2Batch: null device pointer");
    }
    return Status::success();
}

/// 2D grid: blockIdx.x = candidate, blockIdx.y = tile over T.
__global__ void caesar_chi2_histogram_kernel(
    const std::uint8_t* in,
    const std::uint8_t* shifts,
    const std::uint8_t* directions,
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
    const std::uint8_t encrypt = directions[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        const std::uint8_t x = in[t];
        std::uint8_t y;
        if (encrypt != 0u) {
            y = static_cast<std::uint8_t>((x + shift) % kModulus);
        } else {
            y = static_cast<std::uint8_t>((x + kModulus - shift) % kModulus);
        }
        atomicAdd(&shared[y], 1ull);
    }
    __syncthreads();

    if (threadIdx.x < static_cast<int>(kModulus)) {
        atomicAdd(
            &counts[candidate * kModulus + static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

[[nodiscard]] Status launch_impl(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    unsigned long long* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count,
    bool synchronize) {
    Status valid = validate(
        candidate_count,
        token_count,
        device_in,
        device_shifts,
        device_directions,
        device_probabilities,
        device_counts,
        device_scores);
    if (!valid.ok()) {
        return valid;
    }

    const std::size_t hist_bytes =
        candidate_count * CaesarChi2Batch::alphabet_size * sizeof(unsigned long long);
    Status cleared = CudaError::to_status(
        synchronize ? cudaMemset(device_counts, 0, hist_bytes)
                    : cudaMemsetAsync(device_counts, 0, hist_bytes, 0),
        "CaesarChi2Batch::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    caesar_chi2_histogram_kernel<<<grid, kHistThreads>>>(
        device_in, device_shifts, device_directions, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "CaesarChi2Batch::histogram");
    if (!hist.ok()) {
        return hist;
    }

    Status fin = Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
    if (!fin.ok()) {
        return fin;
    }

    if (synchronize) {
        return CudaError::to_status(cudaDeviceSynchronize(), "CaesarChi2Batch::sync");
    }
    return Status::success();
}

}  // namespace

Status CaesarChi2Batch::launch(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    unsigned long long* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    return launch_impl(
        device_in,
        device_shifts,
        device_directions,
        device_probabilities,
        device_counts,
        device_scores,
        candidate_count,
        token_count,
        true);
}

Status CaesarChi2Batch::launch_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    unsigned long long* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    return launch_impl(
        device_in,
        device_shifts,
        device_directions,
        device_probabilities,
        device_counts,
        device_scores,
        candidate_count,
        token_count,
        false);
}
