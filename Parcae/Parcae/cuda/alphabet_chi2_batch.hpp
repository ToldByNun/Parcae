#ifndef ALPHABET_CHI2_BATCH_HPP
#define ALPHABET_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// Alphabet-remap fused χ²: one ciphertext hist, then per-candidate bin remap.
///
/// - Caesar decrypt: `P[b] = H[(b + shift) mod 29]`
/// - Atbash: `P[b] = H[28 - b]` (identical for every occupancy lane)
/// - Atbash∘Caesar-encrypt: `P[b] = H[(28 + shift - b) mod 29]`
/// - Affine decrypt: `P[y] += H[x]` with `y = inv(a)·(x - b)`
///
/// Scores use the same `Chi2BatchScore::finalize_async` path as decode-hist.
class AlphabetChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// `device_cipher_hist` — scratch `uint32[29]` for the once-count.
    /// `device_counts` — `uint32[C * 29]` remapped plaintext hists.
    [[nodiscard]] static Status launch_caesar_decrypt_async(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch_caesar_decrypt(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count);

    [[nodiscard]] static Status launch_atbash_async(
        const std::uint8_t* device_in, const double* device_probabilities,
        std::uint32_t* device_cipher_hist, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch_atbash(
        const std::uint8_t* device_in, const double* device_probabilities,
        std::uint32_t* device_cipher_hist, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status launch_atbash_caesar_async(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch_atbash_caesar(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count);

    [[nodiscard]] static Status launch_affine_decrypt_async(
        const std::uint8_t* device_in, const std::uint8_t* device_a, const std::uint8_t* device_b,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch_affine_decrypt(
        const std::uint8_t* device_in, const std::uint8_t* device_a, const std::uint8_t* device_b,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count);

private:
    AlphabetChi2Batch() = delete;

    [[nodiscard]] static Status validate_common(std::size_t candidate_count, std::size_t token_count,
                                                const std::uint8_t* device_in,
                                                const double* device_probabilities,
                                                const std::uint32_t* device_cipher_hist,
                                                const std::uint32_t* device_counts,
                                                const double* device_scores);

    [[nodiscard]] static Status after_hist_finalize(const double* device_probabilities,
                                                    std::uint32_t* device_counts,
                                                    double* device_scores,
                                                    std::size_t candidate_count,
                                                    std::size_t token_count, cudaStream_t stream);
};

#endif // ALPHABET_CHI2_BATCH_HPP
