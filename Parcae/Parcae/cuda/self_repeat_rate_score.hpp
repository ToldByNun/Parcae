#ifndef SELF_REPEAT_RATE_SCORE_HPP
#define SELF_REPEAT_RATE_SCORE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `SelfRepeatRate::score` (`score_id` self_repeat_rate, v0).
///
/// Counts edges `i` where `x[i] == x[i+1]` with an integer reduce (one thread
/// owner per edge index), then one `double(repeats)/double(N-1)`. Requires
/// `N >= 2`. See `cuda-score-reduction.md`.
class SelfRepeatRateScore {
public:
    /// Device: accumulate adjacent-equal count into `*device_repeat_count`
    /// (must be zeroed). `count` is stream length `N` (not edge count).
    /// No-op success if `count < 2`.
    [[nodiscard]] static Status count_repeats_device(
        const std::uint8_t* device_indices,
        std::size_t count,
        unsigned long long* device_repeat_count);

    /// H2D → integer edge reduce → single FP divide.
    [[nodiscard]] static StatusOr<double> score_host(std::span<const std::uint8_t> indices);

private:
    SelfRepeatRateScore() = delete;
};

#endif  // SELF_REPEAT_RATE_SCORE_HPP
