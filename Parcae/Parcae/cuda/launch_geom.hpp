#ifndef LAUNCH_GEOM_HPP
#define LAUNCH_GEOM_HPP

#include "hist_fast.hpp"

#include <cstddef>

/// Block / grid helpers for stream + fused-hist launches.
///
/// Production default is **256** threads (`HistFast::threads`). A/B values
/// (`k128` / `k512`) are for quiet-plate experiments only — fused-hist shared
/// layouts (`HistFast::warps`, priv/local strides) assume `HistFast::threads`
/// and must not be retargeted without matching shared sizing.
class LaunchGeom {
public:
    static constexpr int kDefaultThreads = 256;

    enum class AbThreads : int {
        k128 = 128,
        k256 = 256,
        k512 = 512,
    };

    [[nodiscard]] static constexpr int threads() noexcept { return kDefaultThreads; }

    [[nodiscard]] static constexpr int threads_ab(AbThreads ab) noexcept {
        return static_cast<int>(ab);
    }

    /// Ceiling division for 1-D grids: `ceil(work / threads)`, at least 1 when
    /// `work > 0`, else 0 (caller skips launch).
    [[nodiscard]] static int blocks_for(std::size_t work, int block_threads = kDefaultThreads) {
        if (work == 0u) {
            return 0;
        }
        const int t = block_threads < 1 ? 1 : block_threads;
        return static_cast<int>((work + static_cast<std::size_t>(t) - 1u) /
                                static_cast<std::size_t>(t));
    }

    static_assert(kDefaultThreads == HistFast::threads,
                  "LaunchGeom default must match HistFast::threads");

private:
    LaunchGeom() = delete;
};

#endif // LAUNCH_GEOM_HPP
