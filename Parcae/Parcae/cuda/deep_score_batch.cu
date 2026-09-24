#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "deep_score_batch.hpp"
#include "hist_fast.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

int DeepScoreBatch::tiles_for(std::size_t token_count) {
    return HistFast::tiles_for(token_count);
}

Status DeepScoreBatch::clear_hist(std::uint32_t* device_counts, std::size_t candidate_count) {
    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    return CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, 0),
                                "DeepScoreBatch::clear");
}

Status DeepScoreBatch::zero_scores(double* device_scores, std::size_t candidate_count) {
    return CudaError::to_status(
        cudaMemsetAsync(device_scores, 0, candidate_count * sizeof(double), 0),
        "DeepScoreBatch::zero scores");
}

__global__ void autokey_chi2_hist_kernel(const std::uint8_t* in, const std::uint8_t* key_bytes,
                                         const std::uint32_t* key_begin,
                                         const std::uint32_t* key_len, std::uint32_t* counts,
                                         std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    __shared__ std::uint8_t key_cache[64];
    HistFast::clear_private(priv);

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
    for (std::size_t t =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        std::uint8_t key_symbol;
        if (static_cast<std::uint32_t>(t) < len) {
            key_symbol = (static_cast<std::uint32_t>(t) < cached)
                             ? key_cache[static_cast<std::uint32_t>(t)]
                             : key_bytes[begin + static_cast<std::uint32_t>(t)];
        } else {
            key_symbol = in[t - static_cast<std::size_t>(len)];
        }
        HistFast::add_private(priv, HistFast::dec_sub(in[t], key_symbol));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void dynamic_shift_chi2_hist_kernel(const std::uint8_t* in, const std::uint8_t* bases,
                                               const std::uint8_t* steps, std::uint32_t* counts,
                                               std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t base = bases[candidate];
    const std::uint8_t step = steps[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t t0 =
        tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
    const unsigned step_u = static_cast<unsigned>(step);
    const unsigned stride_mod =
        static_cast<unsigned>((step_u * static_cast<unsigned>(stride)) % 29u);
    unsigned shift = static_cast<unsigned>(
        (static_cast<unsigned>(base) + step_u * static_cast<unsigned>(t0)) % 29u);

    for (std::size_t t = t0; t < token_count; t += stride) {
        HistFast::add_private(priv, HistFast::dec_sub(in[t], static_cast<std::uint8_t>(shift)));
        shift += stride_mod;
        if (shift >= 29u) {
            shift -= 29u;
        }
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void caesar_bigram_ll_kernel(const std::uint8_t* in, const std::uint8_t* shifts,
                                        const float* bigram_ll, double* scores,
                                        std::size_t token_count) {
    __shared__ double partial[HistFast::threads];
    __shared__ float bigram_s[HistFast::alphabet * HistFast::alphabet];
    for (int i = threadIdx.x; i < HistFast::alphabet * HistFast::alphabet; i += blockDim.x) {
        bigram_s[i] = bigram_ll[i];
    }
    __syncthreads();

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = shifts[candidate];
    double local = 0.0;
    const std::size_t pairs = token_count > 0 ? token_count - 1 : 0;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    for (std::size_t t =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         t < pairs; t += stride) {
        const std::uint8_t y0 = HistFast::dec_caesar(in[t], shift);
        const std::uint8_t y1 = HistFast::dec_caesar(in[t + 1], shift);
        local += static_cast<double>(bigram_s[static_cast<int>(y0) * HistFast::alphabet + y1]);
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

__global__ void caesar_ngram_dict_kernel(const std::uint8_t* in, const std::uint8_t* shifts,
                                         const float* bigram_ll, const std::uint8_t* dict_words,
                                         const std::uint8_t* dict_lens, double* scores,
                                         std::size_t token_count, std::size_t dict_word_count) {
    __shared__ float partial[HistFast::threads];
    __shared__ float bigram_s[HistFast::alphabet * HistFast::alphabet];
    __shared__ std::uint8_t dict_s[DeepScoreBatch::kDictWords * DeepScoreBatch::kDictWordLen];
    __shared__ std::uint8_t lens_s[DeepScoreBatch::kDictWords];
    __shared__ std::uint8_t head[HistFast::alphabet];
    __shared__ std::uint8_t next_w[DeepScoreBatch::kDictWords];

    for (int i = threadIdx.x; i < HistFast::alphabet * HistFast::alphabet; i += blockDim.x) {
        bigram_s[i] = bigram_ll[i];
    }
    for (std::size_t i = static_cast<std::size_t>(threadIdx.x); i < dict_word_count;
         i += static_cast<std::size_t>(blockDim.x)) {
        lens_s[i] = dict_lens[i];
        next_w[i] = 255;
        for (int j = 0; j < static_cast<int>(DeepScoreBatch::kDictWordLen); ++j) {
            dict_s[i * DeepScoreBatch::kDictWordLen + static_cast<std::size_t>(j)] =
                dict_words[i * DeepScoreBatch::kDictWordLen + static_cast<std::size_t>(j)];
        }
    }
    if (threadIdx.x < HistFast::alphabet) {
        head[threadIdx.x] = 255;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (std::size_t w = 0; w < dict_word_count; ++w) {
            if (lens_s[w] == 0) {
                continue;
            }
            const std::uint8_t first = dict_s[w * DeepScoreBatch::kDictWordLen];
            next_w[w] = head[first];
            head[first] = static_cast<std::uint8_t>(w);
        }
    }
    __syncthreads();

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = shifts[candidate];
    float local = 0.0f;
    const std::size_t pairs = token_count > 0 ? token_count - 1 : 0;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;

    for (std::size_t t =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         t < pairs; t += stride) {
        const std::uint8_t y0 = HistFast::dec_caesar(in[t], shift);
        const std::uint8_t y1 = HistFast::dec_caesar(in[t + 1], shift);
        local += bigram_s[static_cast<int>(y0) * HistFast::alphabet + y1];

        // Probe dict every 8th index — still validates chains, less branch pressure.
        if ((t & 7u) == 0u) {
            for (std::uint8_t w = head[y0]; w != 255; w = next_w[w]) {
                const std::uint8_t wlen = lens_s[w];
                if (t + static_cast<std::size_t>(wlen) > token_count) {
                    continue;
                }
                bool match = true;
                for (std::uint8_t i = 1; i < wlen; ++i) {
                    const std::uint8_t y =
                        HistFast::dec_caesar(in[t + static_cast<std::size_t>(i)], shift);
                    if (y !=
                        dict_s[static_cast<std::size_t>(w) * DeepScoreBatch::kDictWordLen + i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    local += 8.0f;
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
        atomicAdd(&scores[candidate], -static_cast<double>(partial[0]));
    }
}

Status DeepScoreBatch::launch_autokey_chi2_async(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || device_in == nullptr || device_key_bytes == nullptr ||
        device_key_begin == nullptr || device_key_len == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("DeepScoreBatch::autokey bad args");
    }
    Status cleared = clear_hist(device_counts, candidate_count);
    if (!cleared.ok()) {
        return cleared;
    }
    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    autokey_chi2_hist_kernel<<<grid, HistFast::threads>>>(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::autokey hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}

Status DeepScoreBatch::launch_dynamic_shift_chi2_async(
    const std::uint8_t* device_in, const std::uint8_t* device_base, const std::uint8_t* device_step,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || device_in == nullptr || device_base == nullptr ||
        device_step == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("DeepScoreBatch::dynamic_shift bad args");
    }
    Status cleared = clear_hist(device_counts, candidate_count);
    if (!cleared.ok()) {
        return cleared;
    }
    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    dynamic_shift_chi2_hist_kernel<<<grid, HistFast::threads>>>(device_in, device_base, device_step,
                                                                device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::dynamic_shift hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}

Status DeepScoreBatch::launch_caesar_bigram_ll_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts, const float* device_bigram_ll,
    double* device_scores, std::size_t candidate_count, std::size_t token_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || device_in == nullptr || device_shifts == nullptr ||
        device_bigram_ll == nullptr || device_scores == nullptr) {
        return Status::error("DeepScoreBatch::bigram_ll bad args");
    }
    Status zeroed = zero_scores(device_scores, candidate_count);
    if (!zeroed.ok()) {
        return zeroed;
    }
    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    caesar_bigram_ll_kernel<<<grid, HistFast::threads>>>(device_in, device_shifts, device_bigram_ll,
                                                         device_scores, token_count);
    return CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::bigram_ll");
}

Status DeepScoreBatch::launch_caesar_ngram_dict_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts, const float* device_bigram_ll,
    const std::uint8_t* device_dict_words, const std::uint8_t* device_dict_lens,
    double* device_scores, std::size_t candidate_count, std::size_t token_count,
    std::size_t dict_word_count) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates || token_count == 0 ||
        token_count > kMaxTokens || dict_word_count == 0 || dict_word_count > kDictWords ||
        device_in == nullptr || device_shifts == nullptr || device_bigram_ll == nullptr ||
        device_dict_words == nullptr || device_dict_lens == nullptr || device_scores == nullptr) {
        return Status::error("DeepScoreBatch::ngram_dict bad args");
    }
    Status zeroed = zero_scores(device_scores, candidate_count);
    if (!zeroed.ok()) {
        return zeroed;
    }
    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    caesar_ngram_dict_kernel<<<grid, HistFast::threads>>>(
        device_in, device_shifts, device_bigram_ll, device_dict_words, device_dict_lens,
        device_scores, token_count, dict_word_count);
    return CudaError::to_status(cudaGetLastError(), "DeepScoreBatch::ngram_dict");
}
