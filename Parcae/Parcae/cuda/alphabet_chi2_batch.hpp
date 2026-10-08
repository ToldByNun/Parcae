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
/// - Vigenère decrypt (interrupt-free): column hists +
///   `P[b] = Σ_j Col[j][(b + key[j]) mod 29]` (one ColumnHistOnce per unique L)
/// - Beaufort (interrupt-free): same columns +
///   `P[b] = Σ_j Col[j][(key[j] - b) mod 29]`
/// - LUT-29 decrypt: `P[lut[x]] += H[x]` (row-major `C × 29`)
///
/// Scores use the same `Chi2BatchScore::finalize_async` path as decode-hist.
class AlphabetChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr std::uint32_t kMaxPeriod = 16384;

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

    /// Interrupt-free Vigenère: unique-L ColumnHistOnce passes + remap.
    /// `device_column_scratch` must hold at least `max(key_len) * 29` bins
    /// (pass nullptr to use process-lifetime scratch sized to `kMaxPeriod`).
    [[nodiscard]] static Status launch_vigenere_decrypt_async(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_column_scratch,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch_vigenere_decrypt(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_column_scratch,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count);

    /// Interrupt-free Beaufort: unique-L ColumnHistOnce + `key - cipher` remap.
    [[nodiscard]] static Status launch_beaufort_async(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_column_scratch,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch_beaufort(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_column_scratch,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count);

    /// Per-candidate LUT-29 decrypt remap: `device_luts` row-major `C × 29`,
    /// `lut[c*29 + x] = decrypt(x)`. Uses `P[lut[x]] += H[x]`.
    [[nodiscard]] static Status launch_lut_decrypt_async(
        const std::uint8_t* device_in, const std::uint8_t* device_luts,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream = nullptr);

    [[nodiscard]] static Status launch_lut_decrypt(
        const std::uint8_t* device_in, const std::uint8_t* device_luts,
        const double* device_probabilities, std::uint32_t* device_cipher_hist,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count);

private:
    AlphabetChi2Batch() = delete;

    enum class PeriodicColumnKind : std::uint8_t { Vigenere, Beaufort };

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

    [[nodiscard]] static Status ensure_column_scratch(std::uint32_t** out_cols);

    [[nodiscard]] static Status launch_periodic_column_remap_async(
        const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
        const double* device_probabilities, std::uint32_t* device_column_scratch,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, PeriodicColumnKind kind, cudaStream_t stream);
};

#endif // ALPHABET_CHI2_BATCH_HPP
