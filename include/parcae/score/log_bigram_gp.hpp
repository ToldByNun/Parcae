#ifndef LOG_BIGRAM_GP_HPP
#define LOG_BIGRAM_GP_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/bigram_model_table.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

/// Sum of conditional bigram log-probabilities (score_id `log_bigram_gp_v0`, v0).
///
/// S = Σ_{i=0}^{N-2} L[x_i, x_{i+1}] with L from BigramModelTable (natural log,
/// row-major 29×29). Higher is better (ScoreOrder::Desc). Requires N >= 2.
class LogBigramGp {
public:
    static constexpr std::string_view score_id = "log_bigram_gp_v0";
    static constexpr std::string_view score_version = "v0";

    [[nodiscard]] static StatusOr<double> score(const std::vector<Index29>& indices,
                                                const BigramModelTable& model) {
        const std::size_t n = indices.size();
        if (n < 2) {
            return Status::error("log_bigram_gp_v0 requires at least 2 symbols");
        }

        double sum = 0.0;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            sum += model.log_prob(indices[i], indices[i + 1]);
        }
        return sum;
    }
};

#endif // LOG_BIGRAM_GP_HPP
