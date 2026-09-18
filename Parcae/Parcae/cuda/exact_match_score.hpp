#ifndef EXACT_MATCH_SCORE_HPP
#define EXACT_MATCH_SCORE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `ExactMatch::score` (`score_id` exact_match, v0).
///
/// Returns `1.0` iff lengths match and every Index29 (`uint8_t`) agrees; else
/// `0.0`. Empty≡empty is `1.0` (host short-circuit). Mismatch count uses
/// integer `atomicAdd` (see `cuda-score-reduction.md`).
class ExactMatchScore {
public:
    /// Device pointers: count mismatches into `*device_mismatch_count` (must be
    /// zeroed by caller). No-op success if `count == 0`.
    [[nodiscard]] static Status count_mismatches_device(
        const std::uint8_t* device_candidate,
        const std::uint8_t* device_reference,
        std::size_t count,
        unsigned long long* device_mismatch_count);

    /// H2D → integer mismatch reduce → `1.0` / `0.0`. Length mismatch → `0.0`.
    [[nodiscard]] static StatusOr<double> score_host(
        std::span<const std::uint8_t> candidate,
        std::span<const std::uint8_t> reference);

private:
    ExactMatchScore() = delete;
};

#endif  // EXACT_MATCH_SCORE_HPP
