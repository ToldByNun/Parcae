#ifndef SCORE_ORDER_HPP
#define SCORE_ORDER_HPP

#include "parcae/score/score_id.hpp"

#include <string_view>

/// Ranking direction for batch / top-k (`asc` = lower is better).
enum class ScoreOrder {
    Asc,
    Desc,
};

class ScoreOrderUtil {
public:
    [[nodiscard]] static constexpr std::string_view to_string(ScoreOrder order) noexcept {
        switch (order) {
        case ScoreOrder::Asc:
            return "asc";
        case ScoreOrder::Desc:
            return "desc";
        }
        return "asc";
    }

    /// Spec table: exact_match/hamming/ic desc; chi2 asc. Unknown → Asc.
    [[nodiscard]] static ScoreOrder for_score_id(const ScoreId& id) noexcept {
        const std::string& s = id.str();
        if (s == "exact_match" || s == "hamming_agreement" || s == "ic_mod29") {
            return ScoreOrder::Desc;
        }
        if (s == "chi2_english_gp_v0") {
            return ScoreOrder::Asc;
        }
        // self_repeat_rate: neither assumed; report raw (treat as Asc for ties only).
        return ScoreOrder::Asc;
    }
};

#endif // SCORE_ORDER_HPP
