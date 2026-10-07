#include "alphabet_chi2_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cipher_hist_once.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

/// Per-candidate Caesar decrypt remap: `counts[c*29 + b] = H[(b + shift) % 29]`.
__global__ void alphabet_chi2_caesar_decrypt_remap_kernel(
    const std::uint32_t* __restrict__ cipher_hist, const std::uint8_t* __restrict__ shifts,
    std::uint32_t* __restrict__ counts, std::size_t candidate_count) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }

    const unsigned shift = static_cast<unsigned>(__ldg(shifts + c)) % 29u;
    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int b = 0; b < HistFast::alphabet; ++b) {
        unsigned src = static_cast<unsigned>(b) + shift;
        if (src >= 29u) {
            src -= 29u;
        }
        row[b] = __ldg(cipher_hist + src);
    }
}

Status AlphabetChi2Batch::validate_caesar(std::size_t candidate_count, std::size_t token_count,
                                          const std::uint8_t* device_in,
                                          const std::uint8_t* device_shifts,
                                          const double* device_probabilities,
                                          const std::uint32_t* device_cipher_hist,
                                          const std::uint32_t* device_counts,
                                          const double* device_scores) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("AlphabetChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("AlphabetChi2Batch: bad T");
    }
    if (device_in == nullptr || device_shifts == nullptr || device_probabilities == nullptr ||
        device_cipher_hist == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("AlphabetChi2Batch: null device pointer");
    }
    return Status::success();
}

Status AlphabetChi2Batch::launch_caesar_decrypt_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    Status valid =
        validate_caesar(candidate_count, token_count, device_in, device_shifts,
                        device_probabilities, device_cipher_hist, device_counts, device_scores);
    if (!valid.ok()) {
        return valid;
    }

    Status hist =
        CipherHistOnce::launch_async(device_in, device_cipher_hist, token_count, stream);
    if (!hist.ok()) {
        return hist;
    }

    constexpr int threads = 128;
    const int blocks =
        static_cast<int>((candidate_count + static_cast<std::size_t>(threads) - 1u) /
                         static_cast<std::size_t>(threads));
    alphabet_chi2_caesar_decrypt_remap_kernel<<<blocks, threads, 0, stream>>>(
        device_cipher_hist, device_shifts, device_counts, candidate_count);
    Status remap =
        CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::caesar remap");
    if (!remap.ok()) {
        return remap;
    }

    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status AlphabetChi2Batch::launch_caesar_decrypt(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count) {
    Status launched = launch_caesar_decrypt_async(
        device_in, device_shifts, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::caesar sync");
}
