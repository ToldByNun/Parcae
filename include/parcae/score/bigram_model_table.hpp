#ifndef BIGRAM_MODEL_TABLE_HPP
#define BIGRAM_MODEL_TABLE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

/// Conditional Index29 bigram log-probability table for `log_bigram_gp_v0`.
/// Layout is row-major: entry (a, b) lives at index `a * 29 + b` (841 cells).
/// Entries are natural-log probabilities after add-one smoothing per source row.
class BigramModelTable {
public:
    static constexpr std::size_t alphabet_size = Index29::modulus;
    static constexpr std::size_t cell_count = alphabet_size * alphabet_size;

    BigramModelTable(std::string id, std::array<double, cell_count> log_probs,
                     std::array<std::uint64_t, cell_count> raw_counts, std::string smoothing,
                     std::string log_base, std::vector<std::string> source_fixture_ids)
        : id_(std::move(id)), log_probs_(log_probs), raw_counts_(raw_counts),
          smoothing_(std::move(smoothing)), log_base_(std::move(log_base)),
          source_fixture_ids_(std::move(source_fixture_ids)) {}

    [[nodiscard]] const std::string& id() const noexcept { return id_; }

    [[nodiscard]] const std::array<double, cell_count>& log_probs() const noexcept {
        return log_probs_;
    }

    [[nodiscard]] const std::array<std::uint64_t, cell_count>& raw_counts() const noexcept {
        return raw_counts_;
    }

    [[nodiscard]] const std::string& smoothing() const noexcept { return smoothing_; }

    [[nodiscard]] const std::string& log_base() const noexcept { return log_base_; }

    [[nodiscard]] const std::vector<std::string>& source_fixture_ids() const noexcept {
        return source_fixture_ids_;
    }

    [[nodiscard]] static constexpr std::size_t flat_index(Index29 prev, Index29 next) noexcept {
        return static_cast<std::size_t>(prev.value()) * alphabet_size +
               static_cast<std::size_t>(next.value());
    }

    [[nodiscard]] double log_prob(Index29 prev, Index29 next) const noexcept {
        return log_probs_[flat_index(prev, next)];
    }

    [[nodiscard]] std::uint64_t raw_count(Index29 prev, Index29 next) const noexcept {
        return raw_counts_[flat_index(prev, next)];
    }

    /// Build add-one–smoothed conditional log-probs from raw bigram counts.
    /// Row \(a\): \(\tilde Z_a = \sum_b (c[a][b] + 1)\),
    /// \(L[a,b] = \ln((c[a][b] + 1) / \tilde Z_a)\).
    [[nodiscard]] static StatusOr<BigramModelTable>
    from_raw_counts(std::string id, const std::array<std::uint64_t, cell_count>& raw_counts,
                    std::vector<std::string> source_fixture_ids) {
        std::array<double, cell_count> logs{};
        for (std::size_t a = 0; a < alphabet_size; ++a) {
            double row_sum = 0.0;
            for (std::size_t b = 0; b < alphabet_size; ++b) {
                row_sum += static_cast<double>(raw_counts[a * alphabet_size + b]) + 1.0;
            }
            if (!(row_sum > 0.0)) {
                return Status::error("bigram model table row sum must be positive");
            }
            for (std::size_t b = 0; b < alphabet_size; ++b) {
                const double smoothed =
                    static_cast<double>(raw_counts[a * alphabet_size + b]) + 1.0;
                logs[a * alphabet_size + b] = std::log(smoothed / row_sum);
            }
        }

        return BigramModelTable(std::move(id), logs, raw_counts, "add_one", "ln",
                                std::move(source_fixture_ids));
    }

private:
    std::string id_;
    std::array<double, cell_count> log_probs_{};
    std::array<std::uint64_t, cell_count> raw_counts_{};
    std::string smoothing_;
    std::string log_base_;
    std::vector<std::string> source_fixture_ids_;
};

#endif // BIGRAM_MODEL_TABLE_HPP
