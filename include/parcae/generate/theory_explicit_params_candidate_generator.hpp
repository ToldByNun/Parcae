#ifndef THEORY_EXPLICIT_PARAMS_CANDIDATE_GENERATOR_HPP
#define THEORY_EXPLICIT_PARAMS_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_dispatch.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstddef>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Bounded generator `gen_theory_explicit_params`.
///
/// Enumerates a **caller-supplied** `params_list` for one `theory_uri` only
/// (cost = |params_list|). Does **not** expand TheorySweep grids. Opt-in search
/// family `theory` requires `SearchJob.allow_theory_uri`.
///
/// Hot path: load `apply_ir` **once**, compile `Z29Bytecode` **once**, then
/// apply each params object via the bytecode host evaluator (oracle-aligned
/// with `DslIrApplicator` for pure HotLoops). Avoids reloading the artifact and
/// re-walking the IR tree per candidate.
class TheoryExplicitParamsCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_theory_explicit_params";

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext, const std::filesystem::path& theories_root,
             std::string_view theory_uri_text, const std::vector<nlohmann::json>& params_list,
             TransformDirection direction = TransformDirection::Decrypt,
             const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (params_list.empty()) {
            return Status::error("gen_theory_explicit_params requires a non-empty params_list");
        }
        StatusOr<TheoryUri> uri = TheoryUri::parse(theory_uri_text);
        if (!uri.ok()) {
            return uri.status();
        }
        const std::string uri_str = uri.value().to_string();
        const TransformId tid = TransformId::unchecked(uri_str);

        StatusOr<PreparedTheory> prepared = prepare(theories_root, uri.value(), direction);
        if (!prepared.ok()) {
            return prepared.status();
        }

        std::optional<nlohmann::json> interrupt_json;
        if (!interrupt.skip_indices().empty()) {
            interrupt_json = interrupt.to_json();
        }

        std::vector<TransformCandidate> out;
        out.reserve(params_list.size());
        std::vector<Index29> plain(ciphertext.size());
        for (std::size_t i = 0; i < params_list.size(); ++i) {
            const nlohmann::json& params = params_list[i];
            if (!params.is_object()) {
                return Status::error(
                    "gen_theory_explicit_params params_list entries must be objects");
            }
            Status st = Z29Bytecode::apply_into_theory(prepared.value().program,
                                                       prepared.value().theory, params, ciphertext,
                                                       plain, interrupt);
            if (!st.ok()) {
                return Status::error("gen_theory_explicit_params apply failed for index " +
                                     std::to_string(i) + ": " + st.message());
            }
            out.emplace_back(make_candidate_id(uri_str, i), tid, direction, params, plain,
                             interrupt_json);
        }
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id(std::string_view theory_uri,
                                                       std::size_t list_index) {
        return "theory:i=" + std::to_string(list_index) + ":uri=" + std::string(theory_uri);
    }

private:
    struct PreparedTheory {
        TheoryIr theory;
        Z29Bytecode::Program program;
    };

    [[nodiscard]] static StatusOr<PreparedTheory>
    prepare(const std::filesystem::path& theories_root, const TheoryUri& uri,
            TransformDirection direction) {
        StatusOr<TheoryIr> theory = TheoryDispatch::load_apply_ir(theories_root, uri);
        if (!theory.ok()) {
            return theory.status();
        }
        // Prefer cipher_var from apply_ir.json when present (TheoryDispatch path).
        std::string cipher_var = "x";
        {
            StatusOr<TheoryArtifact> art =
                TheoryRegistry::load(theories_root, uri.name(), uri.version());
            if (art.ok() && art.value().paths().apply_ir().has_value()) {
                const std::filesystem::path path =
                    art.value().artifact_dir(theories_root) / *art.value().paths().apply_ir();
                StatusOr<std::string> cv = TheoryApplyIr::load_cipher_var(path);
                if (cv.ok()) {
                    cipher_var = std::move(cv.value());
                }
            }
        }
        StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory.value(), direction, cipher_var);
        if (!prog.ok()) {
            return Status::error("gen_theory_explicit_params bytecode compile failed for " +
                                 uri.to_string() + ": " + prog.status().message());
        }
        return PreparedTheory{std::move(theory.value()), std::move(prog.value())};
    }
};

#endif // THEORY_EXPLICIT_PARAMS_CANDIDATE_GENERATOR_HPP
