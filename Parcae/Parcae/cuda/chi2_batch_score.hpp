#ifndef CHI2_BATCH_SCORE_HPP
#define CHI2_BATCH_SCORE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Device-resident batch χ² over candidate-major lanes.
///
/// Used after any transform that leaves `device_out[C * T]` on the GPU, or by
/// fused family kernels that write `device_counts[C * 29]` directly.
class Chi2BatchScore {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Multi-block histogram of `device_out[c * T + t]` into `device_counts[C * 29]`.
    /// Caller must zero `device_counts` first. Does not synchronize.
    [[nodiscard]] static Status histogram_from_out_async(
        const std::uint8_t* device_out,
        unsigned long long* device_counts,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Fixed-order χ² finalize into `device_scores[C]`. Does not synchronize.
    [[nodiscard]] static Status finalize_async(
        const unsigned long long* device_counts,
        const double* device_probabilities,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// `histogram_from_out_async` + `finalize_async` (counts must be clearable via memset).
    [[nodiscard]] static Status score_from_out_async(
        const std::uint8_t* device_out,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

private:
    Chi2BatchScore() = delete;
};

#endif  // CHI2_BATCH_SCORE_HPP
