#ifndef DEEP_SCORE_BATCH_HPP
#define DEEP_SCORE_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Heavier fused decrypt+score paths for multi-key / autokey / n-gram tiers.
///
/// Autokey hist (`launch_autokey_chi2_async`) is dense CTAK decrypt: primer while
/// `t < L`, else key = ciphertext `in[t-L]`, via `AutokeyCtakDevice` — same formula as
/// CPU `CiphertextAutokeyTransform` with empty interrupts.
class DeepScoreBatch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr std::size_t kMaxKeyLen = 64;
    static constexpr std::size_t kDictWords = 256;
    static constexpr std::size_t kDictWordLen = 8;

    [[nodiscard]] static Status launch_autokey_chi2_async(
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

    [[nodiscard]] static Status
    launch_caesar_bigram_ll_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                                  const float* device_bigram_ll, double* device_scores,
                                  std::size_t candidate_count, std::size_t token_count);

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
};

#endif // DEEP_SCORE_BATCH_HPP
