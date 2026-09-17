#ifndef SCORE_REGISTRY_HPP
#define SCORE_REGISTRY_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/chi2_english_gp.hpp"
#include "parcae/score/exact_match.hpp"
#include "parcae/score/hamming_agreement.hpp"
#include "parcae/score/ic_mod29.hpp"
#include "parcae/score/score_catalog_entry.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/score/self_repeat_rate.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

/// String-id dispatch for the tool / agent `score` primitive.
///
/// `score(score_id, candidate, score_version="v0", params_json?, request?)`
/// Pairwise scores take `request.reference` (or `params.reference` as int[]).
/// `chi2_english_gp_v0` requires `request.expected_frequencies`.
class ScoreRegistry {
public:
    [[nodiscard]] static std::vector<ScoreCatalogEntry> catalog() {
        return {
            {ScoreId::exact_match().str(),
             "v0",
             ScoreOrder::Desc,
             ScoreCatalogEntry::Arity::Pairwise},
            {ScoreId::hamming_agreement().str(),
             "v0",
             ScoreOrder::Desc,
             ScoreCatalogEntry::Arity::Pairwise},
            {ScoreId::ic_mod29().str(),
             "v0",
             ScoreOrder::Desc,
             ScoreCatalogEntry::Arity::Unary},
            {ScoreId::chi2_english_gp_v0().str(),
             "v0",
             ScoreOrder::Asc,
             ScoreCatalogEntry::Arity::UnaryWithTable},
            {ScoreId::self_repeat_rate().str(),
             "v0",
             ScoreOrder::Asc,
             ScoreCatalogEntry::Arity::Unary},
        };
    }

    [[nodiscard]] static std::vector<std::string> known_ids() {
        std::vector<std::string> ids;
        for (const ScoreCatalogEntry& entry : catalog()) {
            ids.push_back(entry.id());
        }
        return ids;
    }

    [[nodiscard]] static bool is_known(std::string_view score_id) {
        return ScoreId::from_string(score_id).ok();
    }

    [[nodiscard]] static StatusOr<ScoreOrder> order_of(std::string_view score_id) {
        StatusOr<ScoreId> id = ScoreId::from_string(score_id);
        if (!id.ok()) {
            return id.status();
        }
        return ScoreOrderUtil::for_score_id(id.value());
    }

    [[nodiscard]] static StatusOr<double> score(
        std::string_view score_id,
        std::span<const Index29> candidate,
        std::string_view score_version = "v0",
        const nlohmann::json& params = nlohmann::json::object(),
        const ScoreRequest& request = {}) {
        if (score_version != "v0") {
            return Status::error("Unsupported score_version (only v0 is registered)");
        }

        StatusOr<ScoreId> id = ScoreId::from_string(score_id);
        if (!id.ok()) {
            return id.status();
        }

        const std::vector<Index29> candidate_vec(candidate.begin(), candidate.end());

        if (id.value() == ScoreId::ic_mod29()) {
            return IcMod29::score(candidate_vec);
        }
        if (id.value() == ScoreId::self_repeat_rate()) {
            return SelfRepeatRate::score(candidate_vec);
        }
        if (id.value() == ScoreId::chi2_english_gp_v0()) {
            if (request.expected_frequencies == nullptr) {
                return Status::error(
                    "chi2_english_gp_v0 requires ScoreRequest.expected_frequencies");
            }
            return Chi2EnglishGp::score(candidate_vec, *request.expected_frequencies);
        }
        if (id.value() == ScoreId::exact_match()) {
            StatusOr<std::vector<Index29>> reference =
                resolve_reference(request, params, "exact_match");
            if (!reference.ok()) {
                return reference.status();
            }
            return ExactMatch::score(candidate_vec, reference.value());
        }
        if (id.value() == ScoreId::hamming_agreement()) {
            StatusOr<std::vector<Index29>> reference =
                resolve_reference(request, params, "hamming_agreement");
            if (!reference.ok()) {
                return reference.status();
            }
            return HammingAgreement::score(candidate_vec, reference.value());
        }

        return Status::error("score_id is not supported by ScoreRegistry");
    }

private:
    ScoreRegistry() = delete;

    [[nodiscard]] static StatusOr<std::vector<Index29>> resolve_reference(
        const ScoreRequest& request,
        const nlohmann::json& params,
        std::string_view score_name) {
        if (request.reference.has_value()) {
            const std::span<const Index29> ref = request.reference.value();
            return std::vector<Index29>(ref.begin(), ref.end());
        }

        if (!params.contains("reference")) {
            return Status::error(
                std::string(score_name) +
                " requires ScoreRequest.reference or params.reference");
        }
        if (!params.at("reference").is_array()) {
            return Status::error(std::string(score_name) + " params.reference must be an array");
        }

        std::vector<Index29> out;
        out.reserve(params.at("reference").size());
        for (const auto& item : params.at("reference")) {
            if (!item.is_number_integer()) {
                return Status::error(
                    std::string(score_name) + " params.reference entries must be integers");
            }
            const int value = item.get<int>();
            if (value < 0 || value >= static_cast<int>(Index29::modulus)) {
                return Status::error(
                    std::string(score_name) + " params.reference entry out of range [0,28]");
            }
            out.push_back(Index29{static_cast<std::uint8_t>(value)});
        }
        return out;
    }
};

#endif // SCORE_REGISTRY_HPP
