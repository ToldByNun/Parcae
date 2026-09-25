#ifndef LOG_BIGRAM_GP_SCORE_HPP
#define LOG_BIGRAM_GP_SCORE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA-catalog twin of `LogBigramGp::score` (`score_id` log_bigram_gp_v0, v0).
///
/// Host path: S = Σ L[x_i, x_{i+1}] over a flat 841-entry row-major natural-log
/// table (same layout as `BigramModelTable`). Requires N >= 2. Bit-identical to
/// the CPU registry score when given the same table entries.
class LogBigramGpScore {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t cell_count = alphabet_size * alphabet_size;

    /// Host twin: sum conditional log-probs from `log_probs` (length 841).
    [[nodiscard]] static StatusOr<double> score_host(std::span<const std::uint8_t> indices,
                                                     std::span<const double> log_probs) {
        const std::size_t n = indices.size();
        if (n < 2) {
            return Status::error("log_bigram_gp_v0 requires at least 2 symbols");
        }
        if (log_probs.size() != cell_count) {
            return Status::error("log_bigram_gp_v0 log_probs must have length 841");
        }

        double sum = 0.0;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            const std::uint8_t a = indices[i];
            const std::uint8_t b = indices[i + 1];
            if (a >= alphabet_size || b >= alphabet_size) {
                return Status::error("log_bigram_gp_v0 index out of range [0,28]");
            }
            sum += log_probs[static_cast<std::size_t>(a) * alphabet_size +
                             static_cast<std::size_t>(b)];
        }
        return sum;
    }

private:
    LogBigramGpScore() = delete;
};

#endif // LOG_BIGRAM_GP_SCORE_HPP
