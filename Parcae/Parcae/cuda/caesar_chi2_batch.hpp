#ifndef CAESAR_CHI2_BATCH_HPP
#define CAESAR_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

/// Device-resident Caesar decrypt sweep + χ² histogram/finalize.
///
/// Does **not** materialize `out[C*T]` on host or device for the score path:
/// each candidate lane is decoded on the fly into a 29-bin histogram.
/// Hot-path PCIe is only `scores[C]` (D2H after the timed loop).
class CaesarChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 8192;
    /// Soft cap for a single launch (device memory ≈ T + C*29*8 + …).
    static constexpr std::size_t kMaxTokens = 1u << 22;  // 4M

    /// Clear `device_counts[C * 29]`, fuse decrypt+histogram for all candidates,
    /// finalize χ² into `device_scores[C]` (fixed bin order 0..28). Syncs once.
    ///
    /// `device_probabilities[29]` must already be on device (setup).
    /// Directions: `0` decrypt / `1` encrypt (same as `CudaDir`).
    [[nodiscard]] static Status launch(
        const std::uint8_t* device_in,
        const std::uint8_t* device_shifts,
        const std::uint8_t* device_directions,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

    /// Like `launch`, but queues memset+kernels without synchronizing (caller syncs).
    [[nodiscard]] static Status launch_async(
        const std::uint8_t* device_in,
        const std::uint8_t* device_shifts,
        const std::uint8_t* device_directions,
        const double* device_probabilities,
        unsigned long long* device_counts,
        double* device_scores,
        std::size_t candidate_count,
        std::size_t token_count);

private:
    CaesarChi2Batch() = delete;
};

#endif  // CAESAR_CHI2_BATCH_HPP
