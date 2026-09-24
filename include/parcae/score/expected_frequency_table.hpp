#ifndef EXPECTED_FREQUENCY_TABLE_HPP
#define EXPECTED_FREQUENCY_TABLE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

/// Empirical Index29 frequency table for `chi2_english_gp_v0`.
/// Probabilities MUST be positive and sum (approximately) to 1.
class ExpectedFrequencyTable {
public:
    static constexpr std::size_t alphabet_size = Index29::modulus;

    ExpectedFrequencyTable(std::string id, std::array<double, alphabet_size> probabilities,
                           std::array<std::uint64_t, alphabet_size> raw_counts,
                           std::uint64_t raw_total, std::string smoothing,
                           std::vector<std::string> source_fixture_ids)
        : id_(std::move(id)), probabilities_(probabilities), raw_counts_(raw_counts),
          raw_total_(raw_total), smoothing_(std::move(smoothing)),
          source_fixture_ids_(std::move(source_fixture_ids)) {}

    [[nodiscard]] const std::string& id() const noexcept { return id_; }

    [[nodiscard]] const std::array<double, alphabet_size>& probabilities() const noexcept {
        return probabilities_;
    }

    [[nodiscard]] const std::array<std::uint64_t, alphabet_size>& raw_counts() const noexcept {
        return raw_counts_;
    }

    [[nodiscard]] std::uint64_t raw_total() const noexcept { return raw_total_; }

    [[nodiscard]] const std::string& smoothing() const noexcept { return smoothing_; }

    [[nodiscard]] const std::vector<std::string>& source_fixture_ids() const noexcept {
        return source_fixture_ids_;
    }

    [[nodiscard]] double probability(Index29 index) const noexcept {
        return probabilities_[index.value()];
    }

    /// Build add-one–smoothed probabilities from raw Index29 counts.
    [[nodiscard]] static StatusOr<ExpectedFrequencyTable>
    from_raw_counts(std::string id, const std::array<std::uint64_t, alphabet_size>& raw_counts,
                    std::vector<std::string> source_fixture_ids) {
        std::uint64_t total = 0;
        for (std::uint64_t c : raw_counts) {
            total += c;
        }
        if (total == 0) {
            return Status::error("expected frequency table requires non-empty counts");
        }

        const double denom = static_cast<double>(total) + static_cast<double>(alphabet_size);
        std::array<double, alphabet_size> probs{};
        for (std::size_t i = 0; i < alphabet_size; ++i) {
            probs[i] = (static_cast<double>(raw_counts[i]) + 1.0) / denom;
        }

        return ExpectedFrequencyTable(std::move(id), probs, raw_counts, total, "add_one",
                                      std::move(source_fixture_ids));
    }

private:
    std::string id_;
    std::array<double, alphabet_size> probabilities_{};
    std::array<std::uint64_t, alphabet_size> raw_counts_{};
    std::uint64_t raw_total_ = 0;
    std::string smoothing_;
    std::vector<std::string> source_fixture_ids_;
};

#endif // EXPECTED_FREQUENCY_TABLE_HPP
