#ifndef FAMILY_CHI2_BATCH_HPP
#define FAMILY_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// Fused decrypt+χ² histogram for batch families (device-resident, scores only D2H).
///
/// `uint32` hist bins + thread-local accumulation + uchar4 where safe.
class FamilyChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    [[nodiscard]] static Status
    launch_atbash_async(const std::uint8_t* device_in, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status
    launch_atbash_caesar_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                               const double* device_probabilities, std::uint32_t* device_counts,
                               double* device_scores, std::size_t candidate_count,
                               std::size_t token_count);

    [[nodiscard]] static Status
    launch_affine_async(const std::uint8_t* device_in, const std::uint8_t* device_a,
                        const std::uint8_t* device_b, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status
    launch_vigenere_async(const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
                          const std::uint32_t* device_key_begin,
                          const std::uint32_t* device_key_len, const double* device_probabilities,
                          std::uint32_t* device_counts, double* device_scores,
                          std::size_t candidate_count, std::size_t token_count);

    /// Beaufort decrypt hist: `out = key[j] - in` (involution; same API as Vigenère).
    [[nodiscard]] static Status
    launch_beaufort_async(const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
                          const std::uint32_t* device_key_begin,
                          const std::uint32_t* device_key_len, const double* device_probabilities,
                          std::uint32_t* device_counts, double* device_scores,
                          std::size_t candidate_count, std::size_t token_count);

    /// Totient / prime−1 stream decrypt hist: `out = in - shifts[begin[c] + t]`.
    /// `device_shifts` must cover `begin[c] + token_count` for every candidate.
    [[nodiscard]] static Status
    launch_totient_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                         const std::uint32_t* device_shift_begin,
                         const double* device_probabilities, std::uint32_t* device_counts,
                         double* device_scores, std::size_t candidate_count,
                         std::size_t token_count);

private:
    FamilyChi2Batch() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status clear_and_grid(std::uint32_t* device_counts,
                                               std::size_t candidate_count, std::size_t token_count,
                                               dim3* grid_out);
};

#endif // FAMILY_CHI2_BATCH_HPP
