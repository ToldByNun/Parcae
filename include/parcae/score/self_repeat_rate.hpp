#ifndef SELF_REPEAT_RATE_HPP
#define SELF_REPEAT_RATE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

/// Adjacent equal-rune (diagonal bigram) rate — score_id `self_repeat_rate`, v0.
///
/// R = |{ i : x_i == x_{i+1} }| / (N - 1) for N >= 2.
/// This is a generic exploration statistic (negative controls, batch reports).
/// It does **not** hard-code or endorse any external Liber Primus “self-follow”
/// target (see scores.md Tier C).
class SelfRepeatRate {
public:
    static constexpr std::string_view score_id = "self_repeat_rate";
    static constexpr std::string_view score_version = "v0";

    [[nodiscard]] static StatusOr<double> score(const std::vector<Index29>& indices) {
        const std::size_t n = indices.size();
        if (n < 2) {
            return Status::error("self_repeat_rate requires at least 2 symbols");
        }

        std::uint64_t repeats = 0;
        for (std::size_t i = 0; i + 1 < n; ++i) {
            if (indices[i] == indices[i + 1]) {
                ++repeats;
            }
        }

        return static_cast<double>(repeats) / static_cast<double>(n - 1);
    }
};

#endif // SELF_REPEAT_RATE_HPP
