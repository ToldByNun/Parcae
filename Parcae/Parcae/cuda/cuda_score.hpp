#ifndef CUDA_SCORE_HPP
#define CUDA_SCORE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/score_catalog_entry.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"

#if defined(PARCAE_HAS_CUDA)
#include "chi2_english_gp_score.hpp"
#include "exact_match_score.hpp"
#include "hamming_agreement_score.hpp"
#include "ic_mod29_score.hpp"
#include "parcae_cuda.hpp"
#include "self_repeat_rate_score.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// CUDA twin of `ScoreRegistry::score` — same `score_id` catalog, device kernels.
///
/// Unary / pairwise / table scores dispatch to `ExactMatchScore`,
/// `HammingAgreementScore`, `IcMod29Score`, `SelfRepeatRateScore`,
/// `Chi2EnglishGpScore`. Catalog metadata matches CPU `ScoreRegistry`.
class CudaScore {
public:
    [[nodiscard]] static bool available() noexcept {
#if defined(PARCAE_HAS_CUDA)
        return ParcaeCuda::available();
#else
        return false;
#endif
    }

    [[nodiscard]] static std::vector<ScoreCatalogEntry> catalog() {
        return ScoreRegistry::catalog();
    }

    [[nodiscard]] static std::vector<std::string> known_ids() { return ScoreRegistry::known_ids(); }

    [[nodiscard]] static bool is_known(std::string_view score_id) {
        return ScoreRegistry::is_known(score_id);
    }

    [[nodiscard]] static StatusOr<ScoreOrder> order_of(std::string_view score_id) {
        return ScoreRegistry::order_of(score_id);
    }

    [[nodiscard]] static StatusOr<double>
    score(std::string_view score_id, [[maybe_unused]] std::span<const Index29> candidate,
          std::string_view score_version = "v0",
          [[maybe_unused]] const nlohmann::json& params = nlohmann::json::object(),
          const ScoreRequest& request = ScoreRequest()) {
        if (score_version != "v0") {
            return Status::error("Unsupported score_version (only v0 is registered)");
        }

        StatusOr<ScoreId> id = ScoreId::from_string(score_id);
        if (!id.ok()) {
            return id.status();
        }

        if (!available()) {
            return Status::error("CudaScore: CUDA not available");
        }

#if !defined(PARCAE_HAS_CUDA)
        return Status::error("CudaScore: CUDA not available");
#else
        const std::vector<std::uint8_t> cand_bytes = to_bytes(candidate);

        if (id.value() == ScoreId::ic_mod29()) {
            return IcMod29Score::score_host(cand_bytes);
        }
        if (id.value() == ScoreId::self_repeat_rate()) {
            return SelfRepeatRateScore::score_host(cand_bytes);
        }
        if (id.value() == ScoreId::chi2_english_gp_v0()) {
            if (request.expected_frequencies == nullptr) {
                return Status::error(
                    "chi2_english_gp_v0 requires ScoreRequest.expected_frequencies");
            }
            return Chi2EnglishGpScore::score_host(cand_bytes,
                                                  request.expected_frequencies->probabilities());
        }
        if (id.value() == ScoreId::exact_match()) {
            StatusOr<std::vector<Index29>> reference =
                resolve_reference(request, params, "exact_match");
            if (!reference.ok()) {
                return reference.status();
            }
            return ExactMatchScore::score_host(cand_bytes, to_bytes(reference.value()));
        }
        if (id.value() == ScoreId::hamming_agreement()) {
            StatusOr<std::vector<Index29>> reference =
                resolve_reference(request, params, "hamming_agreement");
            if (!reference.ok()) {
                return reference.status();
            }
            return HammingAgreementScore::score_host(cand_bytes, to_bytes(reference.value()));
        }

        return Status::error("score_id is not supported by CudaScore");
#endif
    }

private:
    CudaScore() = delete;

    [[nodiscard]] static std::vector<std::uint8_t> to_bytes(std::span<const Index29> indices) {
        std::vector<std::uint8_t> bytes(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            bytes[i] = indices[i].value();
        }
        return bytes;
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>>
    resolve_reference(const ScoreRequest& request, const nlohmann::json& params,
                      std::string_view score_name) {
        if (request.reference.has_value()) {
            const std::span<const Index29> ref = request.reference.value();
            return std::vector<Index29>(ref.begin(), ref.end());
        }

        if (!params.contains("reference")) {
            return Status::error(std::string(score_name) +
                                 " requires ScoreRequest.reference or params.reference");
        }
        if (!params.at("reference").is_array()) {
            return Status::error(std::string(score_name) + " params.reference must be an array");
        }

        std::vector<Index29> out;
        out.reserve(params.at("reference").size());
        for (const auto& item : params.at("reference")) {
            if (!item.is_number_integer()) {
                return Status::error(std::string(score_name) +
                                     " params.reference entries must be integers");
            }
            const int value = item.get<int>();
            if (value < 0 || value >= static_cast<int>(Index29::modulus)) {
                return Status::error(std::string(score_name) +
                                     " params.reference entry out of range [0,28]");
            }
            out.push_back(Index29{static_cast<std::uint8_t>(value)});
        }
        return out;
    }
};

#endif // CUDA_SCORE_HPP
