#ifndef IC_MOD29_SCORE_HPP
#define IC_MOD29_SCORE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `IcMod29::score` (`score_id` ic_mod29, v0).
///
/// Histogram with integer `atomicAdd` into 29 bins, then host-side
/// `Σ n_c(n_c-1)` in `c = 0..28` order and one `double` divide.
/// Requires `N >= 2`. See `cuda-score-reduction.md`.
class IcMod29Score {
public:
    static constexpr std::size_t alphabet_size = 29;

    /// Device histogram: `device_counts` must point to 29 zeroed `unsigned long long`.
    /// No-op success if `count == 0`.
    [[nodiscard]] static Status histogram_device(const std::uint8_t* device_indices,
                                                 std::size_t count,
                                                 unsigned long long* device_counts);

    /// H2D → integer histogram → fixed-order numerator → one FP divide.
    [[nodiscard]] static StatusOr<double> score_host(std::span<const std::uint8_t> indices);

private:
    IcMod29Score() = delete;
};

#endif // IC_MOD29_SCORE_HPP
