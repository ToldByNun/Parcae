#ifndef DEEP_SCORE_BATCH_HPP
#define DEEP_SCORE_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Heavier fused decrypt+score paths for multi-key / autokey / n-gram tiers.
///
/// Device-resident; scores-only D2H. Throughput counted as C·T rune-evals.
class DeepScoreBatch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr std::size_t kMaxKeyLen = 64;
    static constexpr std::size_t kDictWords = 256;
    static constexpr std::size_t kDictWordLen = 8;

    /// Ciphertext-autokey Vigenère decrypt → χ²:
    /// `p[t] = c[t] - (t < L ? key[t] : c[t-L])`.
    [[nodiscard]] static Status launch_autokey_chi2_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin,
        const std::uint32_t* device_key_len,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Dynamic shift decrypt → χ²: `p[t] = c[t] - (base + t·step) mod 29`.
    [[nodiscard]] static Status launch_dynamic_shift_chi2_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_base,
        const std::uint8_t* device_step,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Caesar decrypt + English-GP bigram log-likelihood (no hist materialize).
    /// `device_bigram_ll` is 29×29 row-major floats (higher = better).
    [[nodiscard]] static Status launch_caesar_bigram_ll_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_shifts,
        const float* device_bigram_ll,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Caesar decrypt + bigram LL + packed dictionary hit count.
    /// Dictionary: `kDictWords` words × `kDictWordLen` bytes (0-padded), plus lengths.
    [[nodiscard]] static Status launch_caesar_ngram_dict_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_shifts,
        const float* device_bigram_ll,
        const std::uint8_t* device_dict_words,
        const std::uint8_t* device_dict_lens,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count,
        std::size_t dict_word_count);

private:
    DeepScoreBatch() = delete;
};

#endif  // DEEP_SCORE_BATCH_HPP
