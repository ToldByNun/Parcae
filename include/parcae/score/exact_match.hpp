#ifndef EXACT_MATCH_HPP
#define EXACT_MATCH_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

/// Binary equality score — score_id `exact_match`, v0.
/// Returns 1.0 iff lengths match and every Index29 agrees; otherwise 0.0.
/// Never errors (empty≡empty is 1.0). ScoreOrder::Desc.
class ExactMatch {
public:
    static constexpr std::string_view score_id = "exact_match";
    static constexpr std::string_view score_version = "v0";

    [[nodiscard]] static StatusOr<double> score(const std::vector<Index29>& candidate,
                                                const std::vector<Index29>& reference) {
        if (candidate.size() != reference.size()) {
            return 0.0;
        }
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            if (candidate[i] != reference[i]) {
                return 0.0;
            }
        }
        return 1.0;
    }
};

#endif // EXACT_MATCH_HPP
