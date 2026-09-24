#ifndef THEORY_DISPATCH_HPP
#define THEORY_DISPATCH_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_ir_applicator.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_envelope_bridge.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/tool/transform_envelope.hpp"
#include "parcae/transform/apply_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// ApplyTransform hook for theory URIs + frozen catalog envelopes.
///
/// - Catalog `transform_id` → `ApplyTransform` (same semantics as tools).
/// - `parcae://theories/<name>@<ver>` → load artifact `apply_ir.json` via
///   TheoryRegistry (stale dsl_spec_version → hard fail) + `DslIrApplicator`.
///
/// Do not pass theory URIs to `ApplyTransform` / `TransformEnvelope` alone —
/// they reject unknown catalog ids. Use this class (or tool overloads that call it).
class TheoryDispatch {
public:
    /// Apply a theory-aware envelope under `theories_root` (`data/theories`).
    [[nodiscard]] static Status apply_into(
        const std::filesystem::path& theories_root,
        const TheoryEnvelopeBridge::Envelope& envelope,
        std::span<const Index29> input,
        std::span<Index29> output) {
        if (envelope.is_catalog()) {
            StatusOr<TransformEnvelope> catalog =
                envelope.to_catalog_envelope();
            if (!catalog.ok()) {
                return catalog.status();
            }
            return ApplyTransform::apply_into(
                catalog.value().transform_id(),
                input,
                output,
                catalog.value().params(),
                catalog.value().direction(),
                catalog.value().interrupt());
        }
        if (!envelope.theory_uri().has_value()) {
            return Status::error("TheoryDispatch: theory envelope missing TheoryUri");
        }
        return apply_theory_uri(
            theories_root,
            *envelope.theory_uri(),
            input,
            output,
            envelope.params(),
            envelope.direction(),
            envelope.interrupt());
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply(
        const std::filesystem::path& theories_root,
        const TheoryEnvelopeBridge::Envelope& envelope,
        std::span<const Index29> input) {
        std::vector<Index29> out(input.size());
        Status st = apply_into(theories_root, envelope, input, out);
        if (!st.ok()) {
            return st;
        }
        return out;
    }

    /// Apply a loaded TheoryIr (session / test helper — no artifact I/O).
    [[nodiscard]] static Status apply_into(
        const TheoryIr& theory,
        std::span<const Index29> input,
        std::span<Index29> output,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none(),
        std::string_view cipher_var = "x") {
        return DslIrApplicator::apply_into(
            theory, input, output, params, direction, interrupt, cipher_var);
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply(
        const TheoryIr& theory,
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none(),
        std::string_view cipher_var = "x") {
        return DslIrApplicator::apply(
            theory, input, params, direction, interrupt, cipher_var);
    }

    /// Load `paths.apply_ir` for a registered artifact and return runnable TheoryIr.
    [[nodiscard]] static StatusOr<TheoryIr> load_apply_ir(
        const std::filesystem::path& theories_root,
        const TheoryUri& uri) {
        StatusOr<TheoryArtifact> art =
            TheoryRegistry::load(theories_root, uri.name(), uri.version());
        if (!art.ok()) {
            return art.status();
        }
        if (!art.value().paths().apply_ir().has_value()) {
            return Status::error(
                "TheoryDispatch: artifact missing paths.apply_ir for " + uri.to_string() +
                " (recompile with parcae-compile)");
        }
        const std::filesystem::path path =
            art.value().artifact_dir(theories_root) / *art.value().paths().apply_ir();
        return TheoryApplyIr::load(path);
    }

private:
    TheoryDispatch() = delete;

    [[nodiscard]] static Status apply_theory_uri(
        const std::filesystem::path& theories_root,
        const TheoryUri& uri,
        std::span<const Index29> input,
        std::span<Index29> output,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt) {
        StatusOr<TheoryArtifact> art =
            TheoryRegistry::load(theories_root, uri.name(), uri.version());
        if (!art.ok()) {
            return art.status();
        }
        if (!art.value().paths().apply_ir().has_value()) {
            return Status::error(
                "TheoryDispatch: artifact missing paths.apply_ir for " + uri.to_string() +
                " (recompile with parcae-compile)");
        }
        const std::filesystem::path path =
            art.value().artifact_dir(theories_root) / *art.value().paths().apply_ir();
        StatusOr<TheoryIr> theory = TheoryApplyIr::load(path);
        if (!theory.ok()) {
            return theory.status();
        }
        StatusOr<std::string> cipher = TheoryApplyIr::load_cipher_var(path);
        if (!cipher.ok()) {
            return cipher.status();
        }
        return DslIrApplicator::apply_into(
            theory.value(),
            input,
            output,
            params,
            direction,
            interrupt,
            cipher.value());
    }
};

#endif // THEORY_DISPATCH_HPP
