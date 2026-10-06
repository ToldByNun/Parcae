#ifndef HIST_OCCUPANCY_ROOF_HPP
#define HIST_OCCUPANCY_ROOF_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// Shared-cipher **compute** roof calibration for occupancy-padded fused hist.
///
/// Launches identity warp-private hist (`HistFast::add_private` on raw cipher
/// bytes) at catalog Atbash occupancy (`C=512`) with fat-tile + χ² finalize —
/// same traffic shape as `F.atbash`, minimal decrypt ALU. The measured Kernel
/// SLO is the SM/atomic capacity used to freeze
/// `BenchTierSpec::kSharedCipherComputeRoofRps` (Done gate). DRAM occupancy
/// peak (`kDramRooflineSharedCipherOccupancyPeak`) stays diary-only.
///
/// No C++ namespaces.
class HistOccupancyRoof {
public:
    static constexpr std::size_t alphabet_size = 29;
    /// Same occupancy pad as `F.atbash` / `F.totient`.
    static constexpr std::size_t kOccupancyCandidates = 512;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Identity hist + χ² finalize (scores only). Shared `device_in` across C.
    [[nodiscard]] static Status
    launch_identity_async(const std::uint8_t* device_in, const double* device_probabilities,
                          std::uint32_t* device_counts, double* device_scores,
                          std::size_t candidate_count, std::size_t token_count);

private:
    HistOccupancyRoof() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // HIST_OCCUPANCY_ROOF_HPP
