#ifndef CAESAR_CHI2_BATCH_HPP
#define CAESAR_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Device-resident Caesar decrypt sweep + χ² histogram/finalize.
///
/// Does **not** materialize `out[C*T]`. Counts are `uint32` (T ≤ 4M).
/// Prefer `launch_decrypt_async` for search sweeps (vec4 + local hist).
class CaesarChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    [[nodiscard]] static Status launch(const std::uint8_t* device_in,
                                       const std::uint8_t* device_shifts,
                                       const std::uint8_t* device_directions,
                                       const double* device_probabilities,
                                       std::uint32_t* device_counts, double* device_scores,
                                       std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status launch_async(const std::uint8_t* device_in,
                                             const std::uint8_t* device_shifts,
                                             const std::uint8_t* device_directions,
                                             const double* device_probabilities,
                                             std::uint32_t* device_counts, double* device_scores,
                                             std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status
    launch_decrypt_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                         const double* device_probabilities, std::uint32_t* device_counts,
                         double* device_scores, std::size_t candidate_count,
                         std::size_t token_count);

private:
    CaesarChi2Batch() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status
    validate(std::size_t candidate_count, std::size_t token_count, const std::uint8_t* device_in,
             const std::uint8_t* device_shifts, const std::uint8_t* device_directions,
             const double* device_probabilities, std::uint32_t* device_counts,
             double* device_scores, bool decrypt_only);

    [[nodiscard]] static Status
    launch_impl(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                const std::uint8_t* device_directions, const double* device_probabilities,
                std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
                std::size_t token_count, bool synchronize, bool decrypt_only);
};

#endif // CAESAR_CHI2_BATCH_HPP
