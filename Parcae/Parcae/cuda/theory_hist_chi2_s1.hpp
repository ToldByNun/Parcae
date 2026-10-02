#ifndef THEORY_HIST_CHI2_S1_HPP
#define THEORY_HIST_CHI2_S1_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>

/// S1 LUT-29 fused χ² hist for f(x)-only decrypt (Affine/Caesar-shaped).
///
/// `device_luts` is row-major `C × 29`: `lut[c*29 + x] = decrypt(x; params_c)`.
/// Hist is uchar4 via shared LUT (same pattern as `FamilyChi2Batch` affine).
/// Finalize via `Chi2BatchScore`. No C++ namespaces.
class TheoryHistChi2S1 {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    [[nodiscard]] static Status
    launch_lut_async(const std::uint8_t* device_in, const std::uint8_t* device_luts,
                     const double* device_probabilities, std::uint32_t* device_counts,
                     double* device_scores, std::size_t candidate_count, std::size_t token_count);

private:
    TheoryHistChi2S1() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // THEORY_HIST_CHI2_S1_HPP
