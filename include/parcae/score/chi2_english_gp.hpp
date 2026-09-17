#ifndef CHI2_ENGLISH_GP_HPP
#define CHI2_ENGLISH_GP_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

/// Pearson chi-square vs English-GP expected frequencies
/// (score_id `chi2_english_gp_v0`, version v0).
///
/// χ² = Σ_c (o_c - e_c)² / e_c with e_c = p_c * N.
/// Returns raw χ² (lower is closer to the empirical language model;
/// ScoreOrder::Asc). Empty input is an error.
class Chi2EnglishGp {
public:
    static constexpr std::string_view score_id = "chi2_english_gp_v0";
    static constexpr std::string_view score_version = "v0";

    [[nodiscard]] static StatusOr<double> score(
        const std::vector<Index29>& indices,
        const ExpectedFrequencyTable& expected) {
        const std::size_t n = indices.size();
        if (n == 0) {
            return Status::error("chi2_english_gp_v0 requires a non-empty sequence");
        }

        std::array<std::uint64_t, Index29::modulus> observed{};
        for (const Index29 idx : indices) {
            ++observed[idx.value()];
        }

        const double n_d = static_cast<double>(n);
        double chi2 = 0.0;
        for (std::size_t c = 0; c < Index29::modulus; ++c) {
            const double e = expected.probabilities()[c] * n_d;
            if (!(e > 0.0)) {
                return Status::error("chi2_english_gp_v0 expected frequency is zero");
            }
            const double diff = static_cast<double>(observed[c]) - e;
            chi2 += (diff * diff) / e;
        }
        return chi2;
    }
};

#endif // CHI2_ENGLISH_GP_HPP
