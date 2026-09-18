#ifndef CHI2_ENGLISH_GP_SCORE_HPP
#define CHI2_ENGLISH_GP_SCORE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA twin of `Chi2EnglishGp::score` (`score_id` chi2_english_gp_v0, v0).
///
/// Device: integer 29-bin histogram (same pattern as `IcMod29Score`).
/// Host: χ² = Σ_c (o_c - e_c)² / e_c with e_c = p_c * N, walking **c = 0..28**
/// in that order (non-associative FP; see `cuda-score-reduction.md`).
class Chi2EnglishGpScore {
public:
    static constexpr std::size_t alphabet_size = 29;

    /// H2D → integer histogram → fixed-order FP finalize.
    /// `probabilities` must have size 29 and every entry > 0.
    [[nodiscard]] static StatusOr<double> score_host(
        std::span<const std::uint8_t> indices,
        std::span<const double> probabilities);

private:
    Chi2EnglishGpScore() = delete;
};

#endif  // CHI2_ENGLISH_GP_SCORE_HPP
