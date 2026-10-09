#include "alphabet_chi2_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cipher_hist_once.hpp"
#include "column_hist_once.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <vector>

static_assert(AlphabetChi2Batch::kMaxPeriod == ColumnHistOnce::kMaxPeriod,
              "Vigenère period ceiling must match ColumnHistOnce");

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

/// Occupancy lanes share one mirrored hist: `P[b] = H[28 - b]`.
__global__ void alphabet_chi2_atbash_remap_kernel(const std::uint32_t* __restrict__ cipher_hist,
                                                  std::uint32_t* __restrict__ counts,
                                                  std::size_t candidate_count) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }

    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int b = 0; b < HistFast::alphabet; ++b) {
        row[b] = __ldg(cipher_hist + (HistFast::alphabet - 1 - b));
    }
}

/// Catalog Atbash∘Caesar-encrypt: `P[b] = H[(28 + shift - b) mod 29]`.
__global__ void alphabet_chi2_atbash_caesar_remap_kernel(
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
        const unsigned src = (28u + shift + 29u - static_cast<unsigned>(b)) % 29u;
        row[b] = __ldg(cipher_hist + src);
    }
}

/// Affine decrypt permute: `P[inv(a)·(x-b)] = H[x]` (bijection over 0..28).
__global__ void alphabet_chi2_affine_decrypt_remap_kernel(
    const std::uint32_t* __restrict__ cipher_hist, const std::uint8_t* __restrict__ affine_a,
    const std::uint8_t* __restrict__ affine_b, std::uint32_t* __restrict__ counts,
    std::size_t candidate_count) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }

    const std::uint8_t a = __ldg(affine_a + c);
    const std::uint8_t b = __ldg(affine_b + c);
    const std::uint8_t inv_a = Z29Device::inv(a);
    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int i = 0; i < HistFast::alphabet; ++i) {
        row[i] = 0u;
    }
#pragma unroll
    for (int x = 0; x < HistFast::alphabet; ++x) {
        const std::uint8_t y =
            Z29Device::mul(inv_a, Z29Device::sub(static_cast<std::uint8_t>(x), b));
        row[y] = __ldg(cipher_hist + x);
    }
}

/// Per-candidate LUT-29 decrypt: `P[lut[x]] += H[x]` (non-bijective safe).
__global__ void alphabet_chi2_lut_decrypt_remap_kernel(
    const std::uint32_t* __restrict__ cipher_hist, const std::uint8_t* __restrict__ luts,
    std::uint32_t* __restrict__ counts, std::size_t candidate_count) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }

    const std::uint8_t* __restrict__ row_lut =
        luts + c * static_cast<std::size_t>(HistFast::alphabet);
    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int i = 0; i < HistFast::alphabet; ++i) {
        row[i] = 0u;
    }
#pragma unroll
    for (int x = 0; x < HistFast::alphabet; ++x) {
        const std::uint8_t y = __ldg(row_lut + x);
        if (y < HistFast::alphabet) {
            row[y] += __ldg(cipher_hist + x);
        }
    }
}

/// Interrupt-free Vigenère remap for one period filter:
/// `P[b] = Σ_j Col[j][(b + key[j]) mod 29]` when `key_len[c] == period_filter`.
__global__ void alphabet_chi2_vigenere_remap_kernel(
    const std::uint32_t* __restrict__ cols, const std::uint8_t* __restrict__ key_bytes,
    const std::uint32_t* __restrict__ key_begin, const std::uint32_t* __restrict__ key_len,
    std::uint32_t* __restrict__ counts, std::size_t candidate_count,
    std::uint32_t period_filter) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }
    if (__ldg(key_len + c) != period_filter) {
        return;
    }

    const std::uint32_t begin = __ldg(key_begin + c);
    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int b = 0; b < HistFast::alphabet; ++b) {
        row[b] = 0u;
    }

    for (std::uint32_t j = 0; j < period_filter; ++j) {
        const unsigned kj = static_cast<unsigned>(__ldg(key_bytes + begin + j)) % 29u;
        const std::uint32_t* __restrict__ col =
            cols + static_cast<std::size_t>(j) * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
        for (int b = 0; b < HistFast::alphabet; ++b) {
            unsigned src = static_cast<unsigned>(b) + kj;
            if (src >= 29u) {
                src -= 29u;
            }
            row[b] += __ldg(col + src);
        }
    }
}

/// Interrupt-free Beaufort remap: `P[b] = Σ_j Col[j][(key[j] - b) mod 29]`.
__global__ void alphabet_chi2_beaufort_remap_kernel(
    const std::uint32_t* __restrict__ cols, const std::uint8_t* __restrict__ key_bytes,
    const std::uint32_t* __restrict__ key_begin, const std::uint32_t* __restrict__ key_len,
    std::uint32_t* __restrict__ counts, std::size_t candidate_count,
    std::uint32_t period_filter) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }
    if (__ldg(key_len + c) != period_filter) {
        return;
    }

    const std::uint32_t begin = __ldg(key_begin + c);
    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int b = 0; b < HistFast::alphabet; ++b) {
        row[b] = 0u;
    }

    for (std::uint32_t j = 0; j < period_filter; ++j) {
        const unsigned kj = static_cast<unsigned>(__ldg(key_bytes + begin + j)) % 29u;
        const std::uint32_t* __restrict__ col =
            cols + static_cast<std::size_t>(j) * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
        for (int b = 0; b < HistFast::alphabet; ++b) {
            unsigned src = kj + 29u - static_cast<unsigned>(b);
            if (src >= 29u) {
                src -= 29u;
            }
            row[b] += __ldg(col + src);
        }
    }
}

Status AlphabetChi2Batch::validate_common(std::size_t candidate_count, std::size_t token_count,
                                          const std::uint8_t* device_in,
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
    if (device_in == nullptr || device_probabilities == nullptr || device_cipher_hist == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("AlphabetChi2Batch: null device pointer");
    }
    return Status::success();
}

Status AlphabetChi2Batch::after_hist_finalize(const double* device_probabilities,
                                              std::uint32_t* device_counts, double* device_scores,
                                              std::size_t candidate_count, std::size_t token_count,
                                              cudaStream_t stream) {
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status AlphabetChi2Batch::launch_caesar_decrypt_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    Status valid = validate_common(candidate_count, token_count, device_in, device_probabilities,
                                   device_cipher_hist, device_counts, device_scores);
    if (!valid.ok()) {
        return valid;
    }
    if (device_shifts == nullptr) {
        return Status::error("AlphabetChi2Batch: null shifts");
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

    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
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

Status AlphabetChi2Batch::launch_atbash_async(const std::uint8_t* device_in,
                                              const double* device_probabilities,
                                              std::uint32_t* device_cipher_hist,
                                              std::uint32_t* device_counts, double* device_scores,
                                              std::size_t candidate_count, std::size_t token_count,
                                              cudaStream_t stream) {
    Status valid = validate_common(candidate_count, token_count, device_in, device_probabilities,
                                   device_cipher_hist, device_counts, device_scores);
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
    alphabet_chi2_atbash_remap_kernel<<<blocks, threads, 0, stream>>>(device_cipher_hist,
                                                                     device_counts,
                                                                     candidate_count);
    Status remap = CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::atbash remap");
    if (!remap.ok()) {
        return remap;
    }

    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
}

Status AlphabetChi2Batch::launch_atbash(const std::uint8_t* device_in,
                                        const double* device_probabilities,
                                        std::uint32_t* device_cipher_hist,
                                        std::uint32_t* device_counts, double* device_scores,
                                        std::size_t candidate_count, std::size_t token_count) {
    Status launched =
        launch_atbash_async(device_in, device_probabilities, device_cipher_hist, device_counts,
                            device_scores, candidate_count, token_count, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::atbash sync");
}

Status AlphabetChi2Batch::launch_atbash_caesar_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    Status valid = validate_common(candidate_count, token_count, device_in, device_probabilities,
                                   device_cipher_hist, device_counts, device_scores);
    if (!valid.ok()) {
        return valid;
    }
    if (device_shifts == nullptr) {
        return Status::error("AlphabetChi2Batch: null shifts");
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
    alphabet_chi2_atbash_caesar_remap_kernel<<<blocks, threads, 0, stream>>>(
        device_cipher_hist, device_shifts, device_counts, candidate_count);
    Status remap =
        CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::atbash_caesar remap");
    if (!remap.ok()) {
        return remap;
    }

    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
}

Status AlphabetChi2Batch::launch_atbash_caesar(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count) {
    Status launched = launch_atbash_caesar_async(
        device_in, device_shifts, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::atbash_caesar sync");
}

Status AlphabetChi2Batch::launch_affine_decrypt_async(
    const std::uint8_t* device_in, const std::uint8_t* device_a, const std::uint8_t* device_b,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    Status valid = validate_common(candidate_count, token_count, device_in, device_probabilities,
                                   device_cipher_hist, device_counts, device_scores);
    if (!valid.ok()) {
        return valid;
    }
    if (device_a == nullptr || device_b == nullptr) {
        return Status::error("AlphabetChi2Batch: null affine params");
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
    alphabet_chi2_affine_decrypt_remap_kernel<<<blocks, threads, 0, stream>>>(
        device_cipher_hist, device_a, device_b, device_counts, candidate_count);
    Status remap =
        CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::affine remap");
    if (!remap.ok()) {
        return remap;
    }

    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
}

Status AlphabetChi2Batch::launch_affine_decrypt(
    const std::uint8_t* device_in, const std::uint8_t* device_a, const std::uint8_t* device_b,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count) {
    Status launched = launch_affine_decrypt_async(
        device_in, device_a, device_b, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::affine sync");
}

Status AlphabetChi2Batch::ensure_column_scratch(std::uint32_t** out_cols) {
    // Process-lifetime scratch for max period. Not safe for overlapping concurrent
    // Vigenère / Beaufort remaps on this device.
    static std::uint32_t* device_cols = nullptr;
    if (device_cols == nullptr) {
        const std::size_t bytes =
            static_cast<std::size_t>(kMaxPeriod) * alphabet_size * sizeof(std::uint32_t);
        Status allocated =
            CudaError::to_status(cudaMalloc(reinterpret_cast<void**>(&device_cols), bytes),
                                 "AlphabetChi2Batch::column scratch");
        if (!allocated.ok()) {
            device_cols = nullptr;
            return allocated;
        }
    }
    *out_cols = device_cols;
    return Status::success();
}

Status AlphabetChi2Batch::launch_periodic_column_remap_async(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_column_scratch,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, PeriodicColumnKind kind, cudaStream_t stream) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("AlphabetChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("AlphabetChi2Batch: bad T");
    }
    if (device_in == nullptr || device_key_bytes == nullptr || device_key_begin == nullptr ||
        device_key_len == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("AlphabetChi2Batch: null periodic-key pointer");
    }

    std::uint32_t* cols = device_column_scratch;
    if (cols == nullptr) {
        Status scratch = ensure_column_scratch(&cols);
        if (!scratch.ok()) {
            return scratch;
        }
    }

    std::vector<std::uint32_t> host_len(candidate_count);
    Status copied = CudaError::to_status(
        cudaMemcpyAsync(host_len.data(), device_key_len,
                        candidate_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "AlphabetChi2Batch::periodic key_len D2H");
    if (!copied.ok()) {
        return copied;
    }
    Status synced = CudaError::to_status(
        stream == nullptr ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream),
        "AlphabetChi2Batch::periodic key_len sync");
    if (!synced.ok()) {
        return synced;
    }

    std::vector<std::uint32_t> unique_periods;
    unique_periods.reserve(8);
    for (std::size_t i = 0; i < candidate_count; ++i) {
        const std::uint32_t L = host_len[i];
        if (L == 0u || L > kMaxPeriod) {
            return Status::error("AlphabetChi2Batch: key_len out of range 1..kMaxPeriod");
        }
        if (std::find(unique_periods.begin(), unique_periods.end(), L) == unique_periods.end()) {
            unique_periods.push_back(L);
        }
    }

    constexpr int threads = 128;
    const int blocks =
        static_cast<int>((candidate_count + static_cast<std::size_t>(threads) - 1u) /
                         static_cast<std::size_t>(threads));

    for (std::uint32_t period : unique_periods) {
        Status cols_ok =
            ColumnHistOnce::launch_async(device_in, cols, token_count, period, stream);
        if (!cols_ok.ok()) {
            return cols_ok;
        }
        if (kind == PeriodicColumnKind::Beaufort) {
            alphabet_chi2_beaufort_remap_kernel<<<blocks, threads, 0, stream>>>(
                cols, device_key_bytes, device_key_begin, device_key_len, device_counts,
                candidate_count, period);
            Status remap =
                CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::beaufort remap");
            if (!remap.ok()) {
                return remap;
            }
        } else {
            alphabet_chi2_vigenere_remap_kernel<<<blocks, threads, 0, stream>>>(
                cols, device_key_bytes, device_key_begin, device_key_len, device_counts,
                candidate_count, period);
            Status remap =
                CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::vigenere remap");
            if (!remap.ok()) {
                return remap;
            }
        }
    }

    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
}

Status AlphabetChi2Batch::launch_vigenere_decrypt_async(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_column_scratch,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    return launch_periodic_column_remap_async(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_probabilities,
        device_column_scratch, device_counts, device_scores, candidate_count, token_count,
        PeriodicColumnKind::Vigenere, stream);
}

Status AlphabetChi2Batch::launch_vigenere_decrypt(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_column_scratch,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count) {
    Status launched = launch_vigenere_decrypt_async(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_probabilities,
        device_column_scratch, device_counts, device_scores, candidate_count, token_count,
        nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::vigenere sync");
}

Status AlphabetChi2Batch::launch_beaufort_async(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_column_scratch,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    return launch_periodic_column_remap_async(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_probabilities,
        device_column_scratch, device_counts, device_scores, candidate_count, token_count,
        PeriodicColumnKind::Beaufort, stream);
}

Status AlphabetChi2Batch::launch_beaufort(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_column_scratch,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count) {
    Status launched = launch_beaufort_async(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_probabilities,
        device_column_scratch, device_counts, device_scores, candidate_count, token_count,
        nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::beaufort sync");
}

Status AlphabetChi2Batch::launch_lut_decrypt_async(
    const std::uint8_t* device_in, const std::uint8_t* device_luts,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    Status valid = validate_common(candidate_count, token_count, device_in, device_probabilities,
                                   device_cipher_hist, device_counts, device_scores);
    if (!valid.ok()) {
        return valid;
    }
    if (device_luts == nullptr) {
        return Status::error("AlphabetChi2Batch: null luts");
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
    alphabet_chi2_lut_decrypt_remap_kernel<<<blocks, threads, 0, stream>>>(
        device_cipher_hist, device_luts, device_counts, candidate_count);
    Status remap = CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::lut remap");
    if (!remap.ok()) {
        return remap;
    }

    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
}

Status AlphabetChi2Batch::launch_lut_decrypt(
    const std::uint8_t* device_in, const std::uint8_t* device_luts,
    const double* device_probabilities, std::uint32_t* device_cipher_hist,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count) {
    Status launched = launch_lut_decrypt_async(
        device_in, device_luts, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::lut sync");
}

/// S2 period-29 linear: `ks[j]=b0+b1·j`; remap like Vigenère ∓.
__global__ void alphabet_chi2_linear_period29_remap_kernel(
    const std::uint32_t* __restrict__ cols, const std::uint8_t* __restrict__ b0,
    const std::uint8_t* __restrict__ b1, std::uint32_t* __restrict__ counts,
    std::size_t candidate_count, std::uint8_t cipher_minus_ks) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }

    const std::uint8_t pb0 = __ldg(b0 + c);
    const std::uint8_t pb1 = __ldg(b1 + c);
    std::uint8_t ks[HistFast::alphabet];
#pragma unroll
    for (int j = 0; j < HistFast::alphabet; ++j) {
        ks[j] = Z29Device::add(pb0, Z29Device::mul(pb1, static_cast<std::uint8_t>(j)));
    }

    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int b = 0; b < HistFast::alphabet; ++b) {
        std::uint32_t sum = 0u;
#pragma unroll
        for (int j = 0; j < HistFast::alphabet; ++j) {
            unsigned src = static_cast<unsigned>(b);
            if (cipher_minus_ks != 0u) {
                src += static_cast<unsigned>(ks[j]);
            } else {
                src += 29u - static_cast<unsigned>(ks[j]);
            }
            if (src >= 29u) {
                src -= 29u;
            }
            sum += __ldg(cols + static_cast<std::size_t>(j) * HistFast::alphabet + src);
        }
        row[b] = sum;
    }
}

/// S5 period-29 poly: `ks[j]=b0+b1·j+b2·j²`.
__global__ void alphabet_chi2_poly_period29_remap_kernel(
    const std::uint32_t* __restrict__ cols, const std::uint8_t* __restrict__ b0,
    const std::uint8_t* __restrict__ b1, const std::uint8_t* __restrict__ b2,
    std::uint32_t* __restrict__ counts, std::size_t candidate_count,
    std::uint8_t cipher_minus_ks) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }

    const std::uint8_t pb0 = __ldg(b0 + c);
    const std::uint8_t pb1 = __ldg(b1 + c);
    const std::uint8_t pb2 = __ldg(b2 + c);
    std::uint8_t ks[HistFast::alphabet];
#pragma unroll
    for (int j = 0; j < HistFast::alphabet; ++j) {
        const std::uint8_t jj = static_cast<std::uint8_t>(j);
        const std::uint8_t j2 = Z29Device::mul(jj, jj);
        ks[j] = Z29Device::add(pb0, Z29Device::add(Z29Device::mul(pb1, jj), Z29Device::mul(pb2, j2)));
    }

    std::uint32_t* __restrict__ row =
        counts + c * static_cast<std::size_t>(HistFast::alphabet);
#pragma unroll
    for (int b = 0; b < HistFast::alphabet; ++b) {
        std::uint32_t sum = 0u;
#pragma unroll
        for (int j = 0; j < HistFast::alphabet; ++j) {
            unsigned src = static_cast<unsigned>(b);
            if (cipher_minus_ks != 0u) {
                src += static_cast<unsigned>(ks[j]);
            } else {
                src += 29u - static_cast<unsigned>(ks[j]);
            }
            if (src >= 29u) {
                src -= 29u;
            }
            sum += __ldg(cols + static_cast<std::size_t>(j) * HistFast::alphabet + src);
        }
        row[b] = sum;
    }
}

Status AlphabetChi2Batch::launch_linear_period29_async(
    const std::uint8_t* device_in, const std::uint8_t* device_b0, const std::uint8_t* device_b1,
    const double* device_probabilities, std::uint32_t* device_column_scratch,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, bool cipher_minus_ks, cudaStream_t stream) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("AlphabetChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("AlphabetChi2Batch: bad T");
    }
    if (device_in == nullptr || device_b0 == nullptr || device_b1 == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("AlphabetChi2Batch: null linear-period29 pointer");
    }

    std::uint32_t* cols = device_column_scratch;
    if (cols == nullptr) {
        Status scratch = ensure_column_scratch(&cols);
        if (!scratch.ok()) {
            return scratch;
        }
    }

    Status cols_ok =
        ColumnHistOnce::launch_async(device_in, cols, token_count, kPeriod29, stream);
    if (!cols_ok.ok()) {
        return cols_ok;
    }

    constexpr int threads = 128;
    const int blocks =
        static_cast<int>((candidate_count + static_cast<std::size_t>(threads) - 1u) /
                         static_cast<std::size_t>(threads));
    alphabet_chi2_linear_period29_remap_kernel<<<blocks, threads, 0, stream>>>(
        cols, device_b0, device_b1, device_counts, candidate_count,
        cipher_minus_ks ? 1u : 0u);
    Status remap =
        CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::linear period29 remap");
    if (!remap.ok()) {
        return remap;
    }
    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
}

Status AlphabetChi2Batch::launch_linear_period29(
    const std::uint8_t* device_in, const std::uint8_t* device_b0, const std::uint8_t* device_b1,
    const double* device_probabilities, std::uint32_t* device_column_scratch,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, bool cipher_minus_ks) {
    Status launched = launch_linear_period29_async(
        device_in, device_b0, device_b1, device_probabilities, device_column_scratch, device_counts,
        device_scores, candidate_count, token_count, cipher_minus_ks, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::linear period29 sync");
}

Status AlphabetChi2Batch::launch_poly_period29_async(
    const std::uint8_t* device_in, const std::uint8_t* device_b0, const std::uint8_t* device_b1,
    const std::uint8_t* device_b2, const double* device_probabilities,
    std::uint32_t* device_column_scratch, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
    cudaStream_t stream) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("AlphabetChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("AlphabetChi2Batch: bad T");
    }
    if (device_in == nullptr || device_b0 == nullptr || device_b1 == nullptr ||
        device_b2 == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("AlphabetChi2Batch: null poly-period29 pointer");
    }

    std::uint32_t* cols = device_column_scratch;
    if (cols == nullptr) {
        Status scratch = ensure_column_scratch(&cols);
        if (!scratch.ok()) {
            return scratch;
        }
    }

    Status cols_ok =
        ColumnHistOnce::launch_async(device_in, cols, token_count, kPeriod29, stream);
    if (!cols_ok.ok()) {
        return cols_ok;
    }

    constexpr int threads = 128;
    const int blocks =
        static_cast<int>((candidate_count + static_cast<std::size_t>(threads) - 1u) /
                         static_cast<std::size_t>(threads));
    alphabet_chi2_poly_period29_remap_kernel<<<blocks, threads, 0, stream>>>(
        cols, device_b0, device_b1, device_b2, device_counts, candidate_count,
        cipher_minus_ks ? 1u : 0u);
    Status remap =
        CudaError::to_status(cudaGetLastError(), "AlphabetChi2Batch::poly period29 remap");
    if (!remap.ok()) {
        return remap;
    }
    return after_hist_finalize(device_probabilities, device_counts, device_scores, candidate_count,
                               token_count, stream);
}

Status AlphabetChi2Batch::launch_poly_period29(
    const std::uint8_t* device_in, const std::uint8_t* device_b0, const std::uint8_t* device_b1,
    const std::uint8_t* device_b2, const double* device_probabilities,
    std::uint32_t* device_column_scratch, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks) {
    Status launched = launch_poly_period29_async(
        device_in, device_b0, device_b1, device_b2, device_probabilities, device_column_scratch,
        device_counts, device_scores, candidate_count, token_count, cipher_minus_ks, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AlphabetChi2Batch::poly period29 sync");
}
