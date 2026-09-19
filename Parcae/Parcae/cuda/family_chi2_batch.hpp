#ifndef FAMILY_CHI2_BATCH_HPP
#define FAMILY_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Fused decrypt+χ² histogram for batch families (device-resident, scores only D2H).
///
/// Same throughput shape as `CaesarChi2Batch`: multi-block tiles over T, shared-mem
/// hist, then `Chi2BatchScore::finalize_async`. No `out[C*T]` materialization.
class FamilyChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Atbash: `y = 28 - x` for every candidate lane (typically C=1).
    [[nodiscard]] static Status launch_atbash_async(
        const std::uint8_t* device_in,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Atbash∘Caesar decrypt: `y = (28 - x + shift) % 29`.
    [[nodiscard]] static Status launch_atbash_caesar_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_shifts,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Affine decrypt: `inv(a)·(x-b)`.
    [[nodiscard]] static Status launch_affine_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_a,
        const std::uint8_t* device_b,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Vigenère decrypt, no interrupts: `x - key[t % len]`.
    [[nodiscard]] static Status launch_vigenere_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_key_bytes,
        const std::uint32_t* device_key_begin,
        const std::uint32_t* device_key_len,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

private:
    FamilyChi2Batch() = delete;
};

#endif  // FAMILY_CHI2_BATCH_HPP
