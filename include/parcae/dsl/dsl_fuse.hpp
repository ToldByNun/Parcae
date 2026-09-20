#ifndef DSL_FUSE_HPP
#define DSL_FUSE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/compose_ir.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_emit_cpu.hpp"
#include "parcae/dsl/dsl_emit_cuda.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Inline `ComposedTheory` chains + emit fused / staged ComposeDriver fallback
/// (docs/spec/dsl.md § Compose fusion). Bench selection of fused vs staged is F26;
/// callers pass `FusionStatus` explicitly until then.
class DslFuse {
public:
    /// Match `ComposeTransform::max_depth`.
    static constexpr std::size_t max_depth = 8;

    enum class FusionStatus : std::uint8_t {
        Fused = 0,
        FallbackStaged,
    };

    [[nodiscard]] static constexpr std::string_view fusion_status_str(FusionStatus s) noexcept {
        switch (s) {
        case FusionStatus::Fused:
            return "fused";
        case FusionStatus::FallbackStaged:
            return "fallback_staged";
        }
        return "n/a";
    }

    class Result {
    public:
        Result(
            TheoryIr theory,
            std::vector<std::string> flattened_steps,
            bool nested_flattened,
            std::string cipher_var)
            : theory_(std::move(theory)),
              flattened_steps_(std::move(flattened_steps)),
              nested_flattened_(nested_flattened),
              cipher_var_(std::move(cipher_var)) {}

        [[nodiscard]] const TheoryIr& theory() const noexcept {
            return theory_;
        }

        [[nodiscard]] const std::vector<std::string>& flattened_steps() const noexcept {
            return flattened_steps_;
        }

        [[nodiscard]] bool nested_flattened() const noexcept {
            return nested_flattened_;
        }

        [[nodiscard]] const std::string& cipher_var() const noexcept {
            return cipher_var_;
        }

    private:
        TheoryIr theory_;
        std::vector<std::string> flattened_steps_;
        bool nested_flattened_ = false;
        std::string cipher_var_;
    };

    /// Concrete \(\mathbb{Z}_{29}\) bindings for staged recipe materialization
    /// (compose param name → value). Missing keys fall back to `ParamIr::min()`.
    using ParamValues = std::unordered_map<std::string, std::uint8_t>;

    class EmitBundle {
    public:
        EmitBundle(
            FusionStatus status,
            Result fused,
            std::string fused_cpu_header,
            std::string fused_cuda_header,
            std::string fused_cuda_cu,
            std::string staged_recipe_json,
            std::string staged_cpu_header,
            std::string staged_cuda_header)
            : status_(status),
              fused_(std::move(fused)),
              fused_cpu_header_(std::move(fused_cpu_header)),
              fused_cuda_header_(std::move(fused_cuda_header)),
              fused_cuda_cu_(std::move(fused_cuda_cu)),
              staged_recipe_json_(std::move(staged_recipe_json)),
              staged_cpu_header_(std::move(staged_cpu_header)),
              staged_cuda_header_(std::move(staged_cuda_header)) {}

        [[nodiscard]] FusionStatus status() const noexcept {
            return status_;
        }

        [[nodiscard]] std::string_view status_str() const noexcept {
            return fusion_status_str(status_);
        }

        [[nodiscard]] const Result& fused() const noexcept {
            return fused_;
        }

        [[nodiscard]] const std::string& fused_cpu_header() const noexcept {
            return fused_cpu_header_;
        }

        [[nodiscard]] const std::string& fused_cuda_header() const noexcept {
            return fused_cuda_header_;
        }

        [[nodiscard]] const std::string& fused_cuda_cu() const noexcept {
            return fused_cuda_cu_;
        }

        [[nodiscard]] const std::string& staged_recipe_json() const noexcept {
            return staged_recipe_json_;
        }

        [[nodiscard]] const std::string& staged_cpu_header() const noexcept {
            return staged_cpu_header_;
        }

        [[nodiscard]] const std::string& staged_cuda_header() const noexcept {
            return staged_cuda_header_;
        }

        /// Artifact primary CPU text according to `status`.
        [[nodiscard]] const std::string& selected_cpu_header() const noexcept {
            return status_ == FusionStatus::FallbackStaged ? staged_cpu_header_
                                                           : fused_cpu_header_;
        }

        /// Artifact primary CUDA façade according to `status`.
        [[nodiscard]] const std::string& selected_cuda_header() const noexcept {
            return status_ == FusionStatus::FallbackStaged ? staged_cuda_header_
                                                           : fused_cuda_header_;
        }

    private:
        FusionStatus status_ = FusionStatus::Fused;
        Result fused_;
        std::string fused_cpu_header_;
        std::string fused_cuda_header_;
        std::string fused_cuda_cu_;
        std::string staged_recipe_json_;
        std::string staged_cpu_header_;
        std::string staged_cuda_header_;
    };

    /// Flatten nested compose step ids, then inline encrypt/decrypt expressions.
    [[nodiscard]] static StatusOr<Result> fuse_inline(
        const ComposeIr& compose,
        std::span<const TheoryIr> theories,
        std::span<const ComposeIr> composes = {},
        std::string_view cipher_var = "x") {
        if (cipher_var.empty()) {
            return fail(compose, "cipher_var must be non-empty");
        }

        Status st = compose.validate();
        if (!st.ok()) {
            return st;
        }

        std::unordered_map<std::string, const TheoryIr*> theory_by_name;
        for (const TheoryIr& t : theories) {
            theory_by_name.emplace(t.name(), &t);
        }
        std::unordered_map<std::string, const ComposeIr*> compose_by_name;
        for (const ComposeIr& c : composes) {
            compose_by_name.emplace(c.name(), &c);
        }

        std::vector<std::string> flat_steps;
        std::vector<ComposeIr::StepParamBinding> flat_bindings;
        bool nested = false;
        Status flatten_st = flatten_steps(
            compose,
            compose_by_name,
            flat_steps,
            flat_bindings,
            nested,
            /*depth=*/0);
        if (!flatten_st.ok()) {
            return flatten_st;
        }

        for (const ComposeIr::StepParamBinding& b : compose.step_params()) {
            flat_bindings.push_back(b);
        }

        Z29Expr::Ptr decrypt = Z29Expr::var(std::string(cipher_var));
        Z29Expr::Ptr encrypt = Z29Expr::var(std::string(cipher_var));

        for (const std::string& step_id : flat_steps) {
            const TheoryIr* th = find_theory(theory_by_name, step_id);
            if (th == nullptr) {
                return fail(
                    compose,
                    "fuse step '" + step_id + "' is not a TheoryIr in the catalog");
            }
            StatusOr<Z29Expr::Ptr> next = inline_stage(
                *th, step_id, flat_bindings, decrypt, cipher_var, /*use_encrypt=*/false);
            if (!next.ok()) {
                return next.status();
            }
            decrypt = next.value();
        }

        for (std::size_t i = flat_steps.size(); i > 0; --i) {
            const std::string& step_id = flat_steps[i - 1];
            const TheoryIr* th = find_theory(theory_by_name, step_id);
            if (th == nullptr) {
                return fail(
                    compose,
                    "fuse step '" + step_id + "' is not a TheoryIr in the catalog");
            }
            StatusOr<Z29Expr::Ptr> next = inline_stage(
                *th, step_id, flat_bindings, encrypt, cipher_var, /*use_encrypt=*/true);
            if (!next.ok()) {
                return next.status();
            }
            encrypt = next.value();
        }

        StatusOr<TheoryIr> fused = TheoryIr::make(
            compose.name(),
            TheoryIr::Family::Elementwise,
            compose.tier(),
            TheoryIr::InterruptMode::ElementwiseDefault,
            compose.params(),
            encrypt,
            decrypt,
            compose.structural_claim());
        if (!fused.ok()) {
            return fused.status();
        }

        return Result{
            fused.value(),
            std::move(flat_steps),
            nested,
            std::string(cipher_var)};
    }

    /// ComposeTransform / ComposeDriver recipe JSON (`{"stages":[…]}`).
    [[nodiscard]] static StatusOr<std::string> emit_staged_recipe_json(
        const ComposeIr& compose,
        std::span<const TheoryIr> theories,
        std::span<const ComposeIr> composes = {},
        const ParamValues& param_values = {}) {
        StatusOr<Flattened> flat = flatten_only(compose, theories, composes);
        if (!flat.ok()) {
            return flat.status();
        }

        nlohmann::json stages = nlohmann::json::array();
        for (const std::string& step_id : flat.value().steps) {
            const TheoryIr* th = flat.value().theory_by_name.at(step_id);
            StatusOr<std::string> family = catalog_transform_id(step_id);
            if (!family.ok()) {
                return fail(
                    compose,
                    "staged recipe: step '" + step_id +
                        "' is not a ComposeDriver/ComposeTransform catalog id");
            }

            nlohmann::json stage;
            stage["transform_id"] = family.value();
            stage["params"] = stage_params_json(*th, step_id, flat.value().bindings, param_values);
            stages.push_back(std::move(stage));
        }

        nlohmann::json root;
        root["stages"] = std::move(stages);
        return root.dump(2);
    }

    /// Emit fused CPU/CUDA twins + staged ComposeTransform/ComposeDriver façades.
    /// `status` selects which texts `selected_*` return (bench gate is F26).
    [[nodiscard]] static StatusOr<EmitBundle> emit_compose(
        const ComposeIr& compose,
        std::span<const TheoryIr> theories,
        std::span<const ComposeIr> composes = {},
        FusionStatus status = FusionStatus::Fused,
        const ParamValues& staged_param_values = {},
        std::string_view cipher_var = "x") {
        StatusOr<Result> fused = fuse_inline(compose, theories, composes, cipher_var);
        if (!fused.ok()) {
            return fused.status();
        }

        StatusOr<std::string> cpu = DslEmitCpu::emit_theory_header(fused.value().theory(), cipher_var);
        if (!cpu.ok()) {
            return cpu.status();
        }
        StatusOr<std::string> cuda_h =
            DslEmitCuda::emit_theory_header(fused.value().theory(), cipher_var);
        if (!cuda_h.ok()) {
            return cuda_h.status();
        }
        StatusOr<std::string> cuda_cu =
            DslEmitCuda::emit_theory_cu(fused.value().theory(), cipher_var);
        if (!cuda_cu.ok()) {
            return cuda_cu.status();
        }

        StatusOr<std::string> recipe =
            emit_staged_recipe_json(compose, theories, composes, staged_param_values);
        if (!recipe.ok()) {
            return recipe.status();
        }

        StatusOr<Flattened> flat = flatten_only(compose, theories, composes);
        if (!flat.ok()) {
            return flat.status();
        }

        StatusOr<std::string> staged_cpu =
            emit_staged_cpu_header(compose, flat.value(), recipe.value());
        if (!staged_cpu.ok()) {
            return staged_cpu.status();
        }
        StatusOr<std::string> staged_cuda =
            emit_staged_cuda_header(compose, flat.value(), staged_param_values);
        if (!staged_cuda.ok()) {
            return staged_cuda.status();
        }

        return EmitBundle{
            status,
            fused.value(),
            std::move(cpu.value()),
            std::move(cuda_h.value()),
            std::move(cuda_cu.value()),
            std::move(recipe.value()),
            std::move(staged_cpu.value()),
            std::move(staged_cuda.value())};
    }

private:
    DslFuse() = delete;

    class Flattened {
    public:
        std::vector<std::string> steps;
        std::vector<ComposeIr::StepParamBinding> bindings;
        std::unordered_map<std::string, const TheoryIr*> theory_by_name;
        bool nested = false;
    };

    [[nodiscard]] static Status fail(const ComposeIr& compose, std::string message) {
        return DslDiag::make(
                   DslRuleId::E032_primitive_body,
                   "compose '" + compose.name() + "': " + std::move(message))
            .to_status();
    }

    [[nodiscard]] static const TheoryIr* find_theory(
        const std::unordered_map<std::string, const TheoryIr*>& map,
        const std::string& name) {
        const auto it = map.find(name);
        if (it == map.end()) {
            return nullptr;
        }
        return it->second;
    }

    [[nodiscard]] static StatusOr<std::string> catalog_transform_id(std::string_view step) {
        if (step == "identity" || step == "atbash" || step == "caesar" || step == "affine" ||
            step == "vigenere_key" || step == "beaufort_key" ||
            step == "totient_prime_stream") {
            return std::string(step);
        }
        return Status::error("not a catalog transform_id");
    }

    [[nodiscard]] static StatusOr<std::string> cuda_family_enum(std::string_view step) {
        if (step == "identity") {
            return std::string("CudaFamilyId::Identity");
        }
        if (step == "atbash") {
            return std::string("CudaFamilyId::Atbash");
        }
        if (step == "caesar") {
            return std::string("CudaFamilyId::Caesar");
        }
        if (step == "affine") {
            return std::string("CudaFamilyId::Affine");
        }
        if (step == "vigenere_key") {
            return std::string("CudaFamilyId::VigenereKey");
        }
        if (step == "beaufort_key") {
            return std::string("CudaFamilyId::BeaufortKey");
        }
        if (step == "totient_prime_stream") {
            return std::string("CudaFamilyId::TotientPrimeStream");
        }
        return Status::error("not a CudaFamilyId");
    }

    [[nodiscard]] static std::string bound_compose_param(
        const std::string& step_id,
        const std::string& theory_param,
        const std::vector<ComposeIr::StepParamBinding>& bindings) {
        for (const ComposeIr::StepParamBinding& b : bindings) {
            if (b.step_id() == step_id && b.param_name() == theory_param) {
                return b.value_ref();
            }
        }
        return theory_param;
    }

    [[nodiscard]] static nlohmann::json stage_params_json(
        const TheoryIr& theory,
        const std::string& step_id,
        const std::vector<ComposeIr::StepParamBinding>& bindings,
        const ParamValues& values) {
        nlohmann::json params = nlohmann::json::object();
        for (const ParamIr& p : theory.params()) {
            const std::string compose_name = bound_compose_param(step_id, p.name(), bindings);
            const auto it = values.find(compose_name);
            const std::uint8_t v = it != values.end() ? it->second : p.min();
            params[p.name()] = v;
        }
        return params;
    }

    [[nodiscard]] static StatusOr<Flattened> flatten_only(
        const ComposeIr& compose,
        std::span<const TheoryIr> theories,
        std::span<const ComposeIr> composes) {
        Status st = compose.validate();
        if (!st.ok()) {
            return st;
        }

        Flattened out;
        for (const TheoryIr& t : theories) {
            out.theory_by_name.emplace(t.name(), &t);
        }
        std::unordered_map<std::string, const ComposeIr*> compose_by_name;
        for (const ComposeIr& c : composes) {
            compose_by_name.emplace(c.name(), &c);
        }

        st = flatten_steps(
            compose,
            compose_by_name,
            out.steps,
            out.bindings,
            out.nested,
            /*depth=*/0);
        if (!st.ok()) {
            return st;
        }
        for (const ComposeIr::StepParamBinding& b : compose.step_params()) {
            out.bindings.push_back(b);
        }
        for (const std::string& step_id : out.steps) {
            if (out.theory_by_name.find(step_id) == out.theory_by_name.end()) {
                return fail(
                    compose,
                    "fuse step '" + step_id + "' is not a TheoryIr in the catalog");
            }
        }
        return out;
    }

    [[nodiscard]] static Status flatten_steps(
        const ComposeIr& compose,
        const std::unordered_map<std::string, const ComposeIr*>& compose_by_name,
        std::vector<std::string>& out_steps,
        std::vector<ComposeIr::StepParamBinding>& out_bindings,
        bool& nested_flattened,
        std::size_t depth) {
        if (depth > max_depth) {
            return fail(
                compose,
                "compose nesting exceeds max_depth " + std::to_string(max_depth));
        }
        for (const std::string& step_id : compose.steps()) {
            const auto it = compose_by_name.find(step_id);
            if (it != compose_by_name.end()) {
                nested_flattened = true;
                const ComposeIr& inner = *it->second;
                Status st = inner.validate();
                if (!st.ok()) {
                    return st;
                }
                for (const ComposeIr::StepParamBinding& b : inner.step_params()) {
                    out_bindings.push_back(b);
                }
                st = flatten_steps(
                    inner,
                    compose_by_name,
                    out_steps,
                    out_bindings,
                    nested_flattened,
                    depth + 1);
                if (!st.ok()) {
                    return st;
                }
                continue;
            }
            out_steps.push_back(step_id);
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<Z29Expr::Ptr> inline_stage(
        const TheoryIr& theory,
        const std::string& step_id,
        const std::vector<ComposeIr::StepParamBinding>& bindings,
        const Z29Expr::Ptr& cipher_expr,
        std::string_view cipher_var,
        bool use_encrypt) {
        const Z29Expr::Ptr& step =
            use_encrypt ? theory.encrypt_step() : theory.decrypt_step();
        if (!step) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "theory '" + theory.name() + "' missing " +
                           (use_encrypt ? "encrypt_step" : "decrypt_step") +
                           " for fuse step '" + step_id + "'")
                .to_status();
        }
        if (theory.family() != TheoryIr::Family::Elementwise) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "fuse step '" + step_id + "' theory '" + theory.name() +
                           "' must be family elementwise (v0)")
                .to_status();
        }

        std::unordered_map<std::string, Z29Expr::Ptr> mapping;
        mapping.emplace(std::string(cipher_var), cipher_expr);

        for (const ParamIr& p : theory.params()) {
            mapping.emplace(p.name(), Z29Expr::var(p.name()));
        }
        for (const ComposeIr::StepParamBinding& b : bindings) {
            if (b.step_id() != step_id) {
                continue;
            }
            mapping[b.param_name()] = Z29Expr::var(b.value_ref());
        }

        return step->remap(mapping);
    }

    [[nodiscard]] static StatusOr<std::string> emit_staged_cpu_header(
        const ComposeIr& compose,
        const Flattened& flat,
        const std::string& recipe_json) {
        (void)flat;
        const std::string class_name = DslEmitCpu::to_pascal(compose.name()) + "StagedTransform";
        std::string guard;
        for (char ch : class_name) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if (std::isalnum(c)) {
                guard.push_back(static_cast<char>(std::toupper(c)));
            } else {
                guard.push_back('_');
            }
        }
        guard += "_HPP";

        // Escape recipe for a raw string delimiter that is unlikely in JSON.
        const std::string delim = "parcae_staged_recipe";

        std::ostringstream out;
        out << "// Generated by parcae-compile (DslFuse staged) — do not hand-edit.\n";
        out << "// fusion.status fallback path: ComposeTransform ping-pong.\n";
        out << "#ifndef " << guard << "\n";
        out << "#define " << guard << "\n\n";
        out << "#include \"parcae/core/index29.hpp\"\n";
        out << "#include \"parcae/core/status.hpp\"\n";
        out << "#include \"parcae/core/status_or.hpp\"\n";
        out << "#include \"parcae/interrupt/policy.hpp\"\n";
        out << "#include \"parcae/transform/compose_transform.hpp\"\n";
        out << "#include \"parcae/transform/transform_direction.hpp\"\n\n";
        out << "#include <span>\n";
        out << "#include <string_view>\n";
        out << "#include <vector>\n\n";
        out << "#include <nlohmann/json.hpp>\n\n";
        out << "class " << class_name << " {\n";
        out << "public:\n";
        out << "    static constexpr std::string_view theory_id = \"" << compose.name() << "\";\n";
        out << "    static constexpr std::string_view fusion_status = \"fallback_staged\";\n\n";
        out << "    [[nodiscard]] static nlohmann::json recipe() {\n";
        out << "        return nlohmann::json::parse(R\"" << delim << "(\n";
        out << recipe_json << "\n";
        out << ")" << delim << "\");\n";
        out << "    }\n\n";
        out << "    [[nodiscard]] static Status apply_into(\n";
        out << "        std::span<const Index29> input,\n";
        out << "        std::span<Index29> output,\n";
        out << "        TransformDirection direction,\n";
        out << "        const InterruptPolicy& interrupt = InterruptPolicy::none()) {\n";
        out << "        return ComposeTransform{}.apply_into(\n";
        out << "            input, output, recipe(), direction, interrupt);\n";
        out << "    }\n\n";
        out << "private:\n";
        out << "    " << class_name << "() = delete;\n";
        out << "};\n\n";
        out << "#endif // " << guard << "\n";
        return out.str();
    }

    [[nodiscard]] static StatusOr<std::string> emit_staged_cuda_header(
        const ComposeIr& compose,
        const Flattened& flat,
        const ParamValues& param_values) {
        (void)param_values;
        for (const std::string& step_id : flat.steps) {
            StatusOr<std::string> fam = cuda_family_enum(step_id);
            if (!fam.ok()) {
                return fail(
                    compose,
                    "staged ComposeDriver emit: step '" + step_id +
                        "' is not a CUDA catalog family");
            }
        }

        const std::string class_name = DslEmitCpu::to_pascal(compose.name()) + "StagedKernel";
        std::string guard;
        for (char ch : class_name) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if (std::isalnum(c)) {
                guard.push_back(static_cast<char>(std::toupper(c)));
            } else {
                guard.push_back('_');
            }
        }
        guard += "_HPP";

        std::ostringstream out;
        out << "// Generated by parcae-compile (DslFuse staged) — do not hand-edit.\n";
        out << "// fusion.status fallback path: ComposeDriver ping-pong.\n";
        out << "#ifndef " << guard << "\n";
        out << "#define " << guard << "\n\n";
        out << "#include \"../compose_driver.hpp\"\n";
        out << "#include \"../interrupt_device_view.hpp\"\n";
        out << "#include \"../params.hpp\"\n\n";
        out << "#include \"parcae/core/status.hpp\"\n";
        out << "#include \"parcae/interrupt/policy.hpp\"\n\n";
        out << "#include <cstddef>\n";
        out << "#include <cstdint>\n";
        out << "#include <span>\n";
        out << "#include <string_view>\n\n";
        out << "class " << class_name << " {\n";
        out << "public:\n";
        out << "    static constexpr std::string_view theory_id = \"" << compose.name() << "\";\n";
        out << "    static constexpr std::string_view fusion_status = \"fallback_staged\";\n\n";

        out << "    [[nodiscard]] static Status apply_host(\n";
        out << "        std::span<const std::uint8_t> host_in,\n";
        out << "        std::span<std::uint8_t> host_out,\n";
        for (const ParamIr& p : compose.params()) {
            out << "        std::uint8_t " << p.name() << ",\n";
        }
        out << "        CudaDir direction,\n";
        out << "        const InterruptPolicy& interrupt = InterruptPolicy::none()) {\n";
        out << "        if (host_in.size() != host_out.size()) {\n";
        out << "            return Status::error(\"" << class_name
            << "::apply_host size mismatch\");\n";
        out << "        }\n";
        out << "        ComposeParamsHost recipe;\n";
        out << "        recipe.stages.resize(" << flat.steps.size() << ");\n";

        for (std::size_t i = 0; i < flat.steps.size(); ++i) {
            const std::string& step_id = flat.steps[i];
            const TheoryIr* th = flat.theory_by_name.at(step_id);
            StatusOr<std::string> fam = cuda_family_enum(step_id);
            out << "        recipe.stages[" << i << "].family = " << fam.value() << ";\n";
            out << "        recipe.stages[" << i << "].direction = CudaDir::Decrypt;\n";
            if (step_id == "caesar") {
                const std::string compose_param =
                    bound_compose_param(step_id, "shift", flat.bindings);
                out << "        recipe.stages[" << i << "].caesar.shift = " << compose_param
                    << ";\n";
            } else if (step_id == "affine") {
                const std::string a = bound_compose_param(step_id, "a", flat.bindings);
                const std::string b = bound_compose_param(step_id, "b", flat.bindings);
                out << "        recipe.stages[" << i << "].affine.a = " << a << ";\n";
                out << "        recipe.stages[" << i << "].affine.b = " << b << ";\n";
            } else if (step_id == "totient_prime_stream") {
                const std::string idx =
                    bound_compose_param(step_id, "prime_start_index", flat.bindings);
                out << "        recipe.stages[" << i
                    << "].totient.prime_start_index = " << idx << ";\n";
            }
            (void)th;
        }

        out << "        StatusOr<InterruptDeviceView> view =\n";
        out << "            InterruptDeviceView::from_policy(interrupt, host_in.size());\n";
        out << "        if (!view.ok()) {\n";
        out << "            return view.status();\n";
        out << "        }\n";
        out << "        return ComposeDriver::apply_host(\n";
        out << "            host_in, host_out, recipe, view.value(), direction);\n";
        out << "    }\n\n";
        out << "private:\n";
        out << "    " << class_name << "() = delete;\n";
        out << "};\n\n";
        out << "#endif // " << guard << "\n";
        return out.str();
    }
};

#endif // DSL_FUSE_HPP
