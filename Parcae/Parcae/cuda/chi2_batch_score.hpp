#ifndef CHI2_BATCH_SCORE_HPP
#define CHI2_BATCH_SCORE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Device-resident batch χ² over candidate-major lanes.
///
/// Hist bins are `uint32` (T ≤ 4M). Used after any transform that leaves
/// `device_out[C * T]` on the GPU, or by fused family kernels that write
/// `device_counts[C * 29]` directly.
class Chi2BatchScore {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    [[nodiscard]] static Status histogram_from_out_async(const std::uint8_t* device_out,
                                                         std::uint32_t* device_counts,
                                                         std::size_t candidate_count,
                                                         std::size_t token_count);

    [[nodiscard]] static Status finalize_async(const std::uint32_t* device_counts,
                                               const double* device_probabilities,
                                               double* device_scores, std::size_t candidate_count,
                                               std::size_t token_count);

    [[nodiscard]] static Status
    score_from_out_async(const std::uint8_t* device_out, const double* device_probabilities,
                         std::uint32_t* device_counts, double* device_scores,
                         std::size_t candidate_count, std::size_t token_count);

private:
    Chi2BatchScore() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // CHI2_BATCH_SCORE_HPP
