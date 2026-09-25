#ifndef SCORE_REQUEST_HPP
#define SCORE_REQUEST_HPP

#include "parcae/core/index29.hpp"
#include "parcae/score/bigram_model_table.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <optional>
#include <span>

/// Optional side inputs for scores that need more than a candidate span.
/// Tools populate this from CLI flags / params_json.
struct ScoreRequest {
    /// Reference stream for `exact_match` / `hamming_agreement`.
    std::optional<std::span<const Index29>> reference;

    /// Empirical table for `chi2_english_gp_v0` (required for that id).
    const ExpectedFrequencyTable* expected_frequencies = nullptr;

    /// Conditional bigram log table for `log_bigram_gp_v0` (required for that id).
    const BigramModelTable* bigram_model = nullptr;
};

#endif // SCORE_REQUEST_HPP
