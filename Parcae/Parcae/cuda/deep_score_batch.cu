#include "deep_score_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kHistThreads = 256;
constexpr int kScoreThreads = 256;
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

[[nodiscard]] Status clear_hist(
    unsigned long long* device_counts, std::size_t candidate_count) {
    const std::size_t hist_bytes =
        candidate_count * DeepScoreBatch::alphabet_size * sizeof(unsigned long long);
    return CudaError::to_status(
        cudaMemsetAsync(device_counts, 0, hist_bytes, 0), "DeepScoreBatch::clear");
}

[[nodiscard]] Status zero_scores(double* device_scores, std::size_t candidate_count) {
    return CudaError::to_status(
        cudaMemsetAsync(device_scores, 0, candidate_count * sizeof(double), 0),
        "DeepScoreBatch::zero scores");
}

__global__ void autokey_chi2_hist_kernel(
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
        std::uint8_t key_symbol;
        if (static_cast<std::uint32_t>(t) < len) {
            key_symbol = key_bytes[begin + static_cast<std::uint32_t>(t)];
        } else {
            key_symbol = in[t - static_cast<std::size_t>(len)];
        }
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

__global__ void dynamic_shift_chi2_hist_kernel(
    const std::uint8_t* in,
    const std::uint8_t* bases,
    const std::uint8_t* steps,
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

    const std::uint8_t base = bases[candidate];
    const std::uint8_t step = steps[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        const std::uint8_t shift = static_cast<std::uint8_t>(
            (static_cast<unsigned>(base) +
             static_cast<unsigned>(step) * static_cast<unsigned>(t)) %
            static_cast<unsigned>(kModulus));
        const std::uint8_t y = Z29Device::sub(in[t], shift);
        atomicAdd(&shared[y], 1ull);
    }
    __syncthreads();
    if (threadIdx.x < static_cast<int>(kModulus)) {
        atomicAdd(
            &counts[candidate * kModulus + static_cast<std::size_t>(threadIdx.x)],
            shared[threadIdx.x]);
    }
}

__global__ void caesar_bigram_ll_kernel(
    const std::uint8_t* in,
    const std::uint8_t* shifts,
    const float* bigram_ll,
    double* scores,
    std::size_t token_count) {
    __shared__ double partial[kScoreThreads];
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = shifts[candidate];
    double local = 0.0;
    const std::size_t pairs = token_count > 0 ? token_count - 1 : 0;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < pairs;
         t += stride) {
        const std::uint8_t y0 = Z29Device::sub(in[t], shift);
        const std::uint8_t y1 = Z29Device::sub(in[t + 1], shift);
        local += static_cast<double>(bigram_ll[static_cast<std::size_t>(y0) * kModulus + y1]);
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&scores[candidate], -partial[0]);
    }
}

__global__ void caesar_ngram_dict_kernel(
    const std::uint8_t* in,
    const std::uint8_t* shifts,
    const float* bigram_ll,
    const std::uint8_t* dict_words,
    const std::uint8_t* dict_lens,
    double* scores,
    std::size_t token_count,
    std::size_t dict_word_count) {
    __shared__ double partial[kScoreThreads];
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = shifts[candidate];
    double local = 0.0;
    const std::size_t pairs = token_count > 0 ? token_count - 1 : 0;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < pairs;
         t += stride) {
        const std::uint8_t y0 = Z29Device::sub(in[t], shift);
        const std::uint8_t y1 = Z29Device::sub(in[t + 1], shift);
        local += static_cast<double>(bigram_ll[static_cast<std::size_t>(y0) * kModulus + y1]);

        if ((t & 3u) == 0u) {
            for (std::size_t w = 0; w < dict_word_count; ++w) {
                const std::uint8_t wlen = dict_lens[w];
                if (wlen == 0 || t + static_cast<std::size_t>(wlen) > token_count) {
                    continue;
                }
                bool match = true;
                const std::uint8_t* word = dict_words + w * DeepScoreBatch::kDictWordLen;
                for (std::uint8_t i = 0; i < wlen; ++i) {
                    const std::uint8_t y =
                        Z29Device::sub(in[t + static_cast<std::size_t>(i)], shift);
                    if (y != word[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    local += 8.0;
                }
            }
        }
    }
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&scores[candidate], -partial[0]);
    }
}

}  // namespace

Status DeepScoreBatch::launch_autokey_chi2_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin,
    const std::uint32_t* device_key_len,
    const double* device_probabilities,
    unsigned long long* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || device_in == nullptr || device_key_bytes == nullptr ||
        device_key_begin == nullptr || device_key_len == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("DeepScoreBatch::autokey bad args");
    }
    Status cleared = clear_hist(device_counts, candidate_count);
    if (!cleared.ok()) {
        return cleared;
    }
    const dim3 grid(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    autokey_chi2_hist_kernel<<<grid, kHistThreads>>>(
        device_in,
        device_key_bytes,
        device_key_begin,
        device_key_len,
        device_counts,
        token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::autokey hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status DeepScoreBatch::launch_dynamic_shift_chi2_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_base,
    const std::uint8_t* device_step,
    const double* device_probabilities,
    unsigned long long* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || device_in == nullptr || device_base == nullptr ||
        device_step == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("DeepScoreBatch::dynamic_shift bad args");
    }
    Status cleared = clear_hist(device_counts, candidate_count);
    if (!cleared.ok()) {
        return cleared;
    }
    const dim3 grid(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    dynamic_shift_chi2_hist_kernel<<<grid, kHistThreads>>>(
        device_in, device_base, device_step, device_counts, token_count);
    Status hist =
        CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::dynamic_shift hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
}

Status DeepScoreBatch::launch_caesar_bigram_ll_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const float* device_bigram_ll,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || device_in == nullptr || device_shifts == nullptr ||
        device_bigram_ll == nullptr || device_scores == nullptr) {
        return Status::error("DeepScoreBatch::bigram_ll bad args");
    }
    Status zeroed = zero_scores(device_scores, candidate_count);
    if (!zeroed.ok()) {
        return zeroed;
    }
    const dim3 grid(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    caesar_bigram_ll_kernel<<<grid, kScoreThreads>>>(
        device_in, device_shifts, device_bigram_ll, device_scores, token_count);
    return CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::bigram_ll");
}

Status DeepScoreBatch::launch_caesar_ngram_dict_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const float* device_bigram_ll,
    const std::uint8_t* device_dict_words,
    const std::uint8_t* device_dict_lens,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count,
    std::size_t dict_word_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || dict_word_count == 0 || dict_word_count > kDictWords ||
        device_in == nullptr || device_shifts == nullptr || device_bigram_ll == nullptr ||
        device_dict_words == nullptr || device_dict_lens == nullptr ||
        device_scores == nullptr) {
        return Status::error("DeepScoreBatch::ngram_dict bad args");
    }
    Status zeroed = zero_scores(device_scores, candidate_count);
    if (!zeroed.ok()) {
        return zeroed;
    }
    const dim3 grid(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    caesar_ngram_dict_kernel<<<grid, kScoreThreads>>>(
        device_in,
        device_shifts,
        device_bigram_ll,
        device_dict_words,
        device_dict_lens,
        device_scores,
        token_count,
        dict_word_count);
    return CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::ngram_dict");
}
