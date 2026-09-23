#ifndef THEORY_EXPLICIT_PARAMS_CANDIDATE_GENERATOR_HPP
#define THEORY_EXPLICIT_PARAMS_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_dispatch.hpp"
#include "parcae/dsl/theory_envelope_bridge.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Bounded generator `gen_theory_explicit_params`.
///
/// Enumerates a **caller-supplied** `params_list` for one `theory_uri` only
/// (cost = |params_list|). Does **not** expand TheorySweep grids. Opt-in search
/// family `theory` requires `SearchJob.allow_theory_uri`.
class TheoryExplicitParamsCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_theory_explicit_params";

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> generate(
        std::span<const Index29> ciphertext,
        const std::filesystem::path& theories_root,
        std::string_view theory_uri_text,
        const std::vector<nlohmann::json>& params_list,
        TransformDirection direction = TransformDirection::Decrypt,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (params_list.empty()) {
            return Status::error(
                "gen_theory_explicit_params requires a non-empty params_list");
        }
        StatusOr<TheoryUri> uri = TheoryUri::parse(theory_uri_text);
        if (!uri.ok()) {
            return uri.status();
        }
        const std::string uri_str = uri.value().to_string();
        const TransformId tid = TransformId::unchecked(uri_str);

        std::optional<nlohmann::json> interrupt_json;
        if (!interrupt.skip_indices().empty()) {
            interrupt_json = interrupt.to_json();
        }

        std::vector<TransformCandidate> out;
        out.reserve(params_list.size());
        for (std::size_t i = 0; i < params_list.size(); ++i) {
            const nlohmann::json& params = params_list[i];
            if (!params.is_object()) {
                return Status::error(
                    "gen_theory_explicit_params params_list entries must be objects");
            }
            TheoryEnvelopeBridge::Envelope envelope{
                TheoryEnvelopeBridge::Kind::Theory,
                uri_str,
                direction,
                params,
                interrupt,
                uri.value()};
            StatusOr<std::vector<Index29>> plain =
                TheoryDispatch::apply(theories_root, envelope, ciphertext);
            if (!plain.ok()) {
                return Status::error(
                    "gen_theory_explicit_params apply failed for index " +
                    std::to_string(i) + ": " + plain.status().message());
            }
            out.emplace_back(
                make_candidate_id(uri_str, i),
                tid,
                direction,
                params,
                std::move(plain.value()),
                interrupt_json);
        }
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id(
        std::string_view theory_uri,
        std::size_t list_index) {
        return "theory:i=" + std::to_string(list_index) + ":uri=" + std::string(theory_uri);
    }
};

#endif  // THEORY_EXPLICIT_PARAMS_CANDIDATE_GENERATOR_HPP
