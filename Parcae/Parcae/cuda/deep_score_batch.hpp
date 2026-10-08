#ifndef DEEP_SCORE_BATCH_HPP
#define DEEP_SCORE_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// Heavier fused decrypt+score paths for multi-key / autokey / n-gram tiers.
///
/// **CTAK production** (`launch_autokey_chi2_async`): LagDiffHistOnce per unique
/// primer length + O(L) primer-prefix merge (interrupt-free dense CTAK). Legacy
/// decode→hist remains as `launch_autokey_chi2_decode_hist_async`.
///
/// **Caesar bigram-LL production** (`launch_caesar_bigram_ll_async`):
/// `BigramCountOnce` → O(C·841) remap. Canonical score contract (matches
/// `HistAlphabetMap::bigram_ll_dot_rotated`):
///   `score_s = - Σ_{x0,x1=0..28} B[x0,x1] · ll[(x0−s) mod 29][(x1−s) mod 29]`
/// with fixed `x0,x1` loop order (not stream/tile accumulation order).
/// `T < 2` → all scores `0`. Legacy stream path:
/// `launch_caesar_bigram_ll_decode_async` (FP order may differ slightly).
///
/// **`launch_caesar_ngram_dict_async`:** unchanged stream/dict kernel (dictionary
/// matching is not once-remapable without a different index structure).
class DeepScoreBatch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kBigramBins = alphabet_size * alphabet_size;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr std::size_t kMaxKeyLen = 64;
    static constexpr std::size_t kDictWords = 256;
    static constexpr std::size_t kDictWordLen = 8;

    /// Production CTAK χ²: lag-diff once + primer prefix remap.
    [[nodiscard]] static Status launch_autokey_chi2_async(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count);

    /// Legacy per-candidate CTAK decode→hist (`autokey_chi2_hist_kernel`).
    [[nodiscard]] static Status launch_autokey_chi2_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status
    launch_dynamic_shift_chi2_async(const std::uint8_t* device_in, const std::uint8_t* device_base,
                                    const std::uint8_t* device_step,
                                    const double* device_probabilities,
                                    std::uint32_t* device_counts, double* device_scores,
                                    std::size_t candidate_count, std::size_t token_count);

    /// Production Caesar bigram-LL: BigramCountOnce + remap (canonical order).
    [[nodiscard]] static Status
    launch_caesar_bigram_ll_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                                  const float* device_bigram_ll, double* device_scores,
                                  std::size_t candidate_count, std::size_t token_count);

    /// Legacy per-candidate stream bigram-LL (`caesar_bigram_ll_kernel`).
    [[nodiscard]] static Status launch_caesar_bigram_ll_decode_async(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const float* device_bigram_ll, double* device_scores, std::size_t candidate_count,
        std::size_t token_count);

    /// Stream bigram-LL + dictionary probe (not remapped; leave as-is).
    [[nodiscard]] static Status launch_caesar_ngram_dict_async(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const float* device_bigram_ll, const std::uint8_t* device_dict_words,
        const std::uint8_t* device_dict_lens, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, std::size_t dict_word_count);

private:
    DeepScoreBatch() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status clear_hist(std::uint32_t* device_counts,
                                           std::size_t candidate_count);

    [[nodiscard]] static Status zero_scores(double* device_scores, std::size_t candidate_count);

    [[nodiscard]] static Status ensure_lag_diff_scratch(std::uint32_t** out_hist);

    [[nodiscard]] static Status ensure_bigram_scratch(std::uint32_t** out_bigrams);

    [[nodiscard]] static Status launch_autokey_chi2_remap_async(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream);

    [[nodiscard]] static Status launch_caesar_bigram_ll_remap_async(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const float* device_bigram_ll, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream);
};

#endif // DEEP_SCORE_BATCH_HPP
