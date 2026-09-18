#ifndef HAMMING_AGREEMENT_SCORE_HPP
#define HAMMING_AGREEMENT_SCORE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `HammingAgreement::score` (`score_id` hamming_agreement, v0).
///
/// Integer match count, then one `double(matches)/double(N)`. Equal non-empty
/// lengths required (hard error otherwise). See `cuda-score-reduction.md`.
class HammingAgreementScore {
public:
    /// Device pointers: accumulate match count into `*device_match_count`
    /// (must be zeroed by caller). No-op success if `count == 0`.
    [[nodiscard]] static Status count_matches_device(
        const std::uint8_t* device_candidate,
        const std::uint8_t* device_reference,
        std::size_t count,
        unsigned long long* device_match_count);

    /// H2D → integer match reduce → single FP divide. Empty / length mismatch → error.
    [[nodiscard]] static StatusOr<double> score_host(
        std::span<const std::uint8_t> candidate,
        std::span<const std::uint8_t> reference);

private:
    HammingAgreementScore() = delete;
};

#endif  // HAMMING_AGREEMENT_SCORE_HPP
