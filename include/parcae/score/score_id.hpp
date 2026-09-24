#ifndef SCORE_ID_HPP
#define SCORE_ID_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <string>
#include <string_view>
#include <utility>

/// Stable string id for a score (`ic_mod29`, `chi2_english_gp_v0`, …).
class ScoreId {
public:
    [[nodiscard]] static ScoreId exact_match() { return ScoreId{"exact_match"}; }

    [[nodiscard]] static ScoreId hamming_agreement() { return ScoreId{"hamming_agreement"}; }

    [[nodiscard]] static ScoreId ic_mod29() { return ScoreId{"ic_mod29"}; }

    [[nodiscard]] static ScoreId chi2_english_gp_v0() { return ScoreId{"chi2_english_gp_v0"}; }

    [[nodiscard]] static ScoreId self_repeat_rate() { return ScoreId{"self_repeat_rate"}; }

    [[nodiscard]] static StatusOr<ScoreId> from_string(std::string_view text) {
        if (text == "exact_match") {
            return exact_match();
        }
        if (text == "hamming_agreement") {
            return hamming_agreement();
        }
        if (text == "ic_mod29") {
            return ic_mod29();
        }
        if (text == "chi2_english_gp_v0") {
            return chi2_english_gp_v0();
        }
        if (text == "self_repeat_rate") {
            return self_repeat_rate();
        }
        return Status::error("Unknown score_id");
    }

    [[nodiscard]] const std::string& str() const noexcept { return value_; }

    [[nodiscard]] bool operator==(const ScoreId&) const noexcept = default;

private:
    explicit ScoreId(std::string value) : value_(std::move(value)) {}

    std::string value_;
};

#endif // SCORE_ID_HPP
