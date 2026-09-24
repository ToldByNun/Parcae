#ifndef DSL_IR_APPLICATOR_HPP
#define DSL_IR_APPLICATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/primitive_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_buffer.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// CPU reference applicator: evaluate DSL IR elementwise into an output span.
/// Matches Transform hot-path shape (`apply_into`, interrupt pass-through).
/// Emitted `…Transform` sources (DslEmitCpu) are optional; this is the IR path.
class DslIrApplicator {
public:
    /// Bind `cipher_var` to each consumable rune; `params` hold remaining env.
    /// Skipped interrupt indices copy input→output unchanged.
    [[nodiscard]] static Status apply_into(
        const Z29Expr::Ptr& step,
        std::string_view cipher_var,
        const Z29Expr::Env& params,
        std::span<const Index29> input,
        std::span<Index29> output,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (!step) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "step expression is null")
                .to_status();
        }
        if (cipher_var.empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "cipher_var must be non-empty")
                .to_status();
        }
        if (params.find(std::string(cipher_var)) != params.end()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "cipher_var '" + std::string(cipher_var) +
                           "' collides with a fixed param binding")
                .to_status();
        }
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        Status range = TransformBuffer::validate_interrupt_range(
            interrupt, input.size(), "DslIrApplicator");
        if (!range.ok()) {
            return range;
        }

        const auto skips = TransformBuffer::skip_span(interrupt);
        Z29Expr::Env env = params;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (TransformBuffer::should_skip(skips, i)) {
                output[i] = input[i];
                continue;
            }
            env[std::string(cipher_var)] = input[i];
            StatusOr<Index29> out = step->eval(env);
            if (!out.ok()) {
                return out.status();
            }
            output[i] = out.value();
        }
        return Status::success();
    }

    /// Primitive stream apply: `param_names[0]` binds to each input rune;
    /// `param_tail` supplies the remaining arity-1 values in signature order.
    [[nodiscard]] static Status apply_into(
        const PrimitiveIr& primitive,
        std::span<const Index29> param_tail,
        std::span<const Index29> input,
        std::span<Index29> output,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        Status st = primitive.validate();
        if (!st.ok()) {
            return st;
        }
        if (primitive.arity() < 1) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "primitive '" + primitive.name() +
                           "' arity must be >= 1 for stream apply_into")
                .to_status();
        }
        if (param_tail.size() + 1 != primitive.arity()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "primitive '" + primitive.name() + "' expects " +
                           std::to_string(primitive.arity() - 1) +
                           " trailing params, got " + std::to_string(param_tail.size()))
                .to_status();
        }
        Z29Expr::Env fixed;
        for (std::size_t i = 0; i < param_tail.size(); ++i) {
            fixed.emplace(primitive.param_names()[i + 1], param_tail[i]);
        }
        return apply_into(
            primitive.body(), primitive.param_names()[0], fixed, input, output, interrupt);
    }

    /// TheoryIr: selects encrypt_step / decrypt_step; JSON keys match ParamIr names.
    /// Default cipher variable is `"x"` (must match the step expression).
    [[nodiscard]] static Status apply_into(
        const TheoryIr& theory,
        std::span<const Index29> input,
        std::span<Index29> output,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none(),
        std::string_view cipher_var = "x") {
        StatusOr<Z29Expr::Env> env = bind_theory_params(theory, params);
        if (!env.ok()) {
            return env.status();
        }
        // Keyed-stream position: emit maps var `i` to the loop index; bind the same
        // unless `i` is already a declared theory param.
        bool i_is_param = false;
        for (const ParamIr& p : theory.params()) {
            if (p.name() == "i") {
                i_is_param = true;
                break;
            }
        }
        const Z29Expr::Ptr& step = (direction == TransformDirection::Encrypt)
                                       ? theory.encrypt_step()
                                       : theory.decrypt_step();
        if (!step) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       std::string("theory '") + theory.name() + "' missing " +
                           (direction == TransformDirection::Encrypt ? "encrypt_step"
                                                                     : "decrypt_step"))
                .to_status();
        }
        if (theory.interrupt_mode() == TheoryIr::InterruptMode::NoneByDesign &&
            !interrupt.empty()) {
            return DslDiag::make(
                       DslRuleId::E030_interrupt_policy,
                       "theory '" + theory.name() +
                           "' declares interrupts=none_by_design; non-empty "
                           "InterruptPolicy rejected (CPU↔CUDA parity)")
                .to_status();
        }
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        Status range = TransformBuffer::validate_interrupt_range(
            interrupt, input.size(), "DslIrApplicator");
        if (!range.ok()) {
            return range;
        }

        const auto skips = TransformBuffer::skip_span(interrupt);
        Z29Expr::Env run = env.value();
        for (std::size_t idx = 0; idx < input.size(); ++idx) {
            if (TransformBuffer::should_skip(skips, idx)) {
                output[idx] = input[idx];
                continue;
            }
            run[std::string(cipher_var)] = input[idx];
            if (!i_is_param) {
                run["i"] = Index29{static_cast<std::uint8_t>(idx % Index29::modulus)};
            }
            StatusOr<Index29> out = step->eval(run);
            if (!out.ok()) {
                return out.status();
            }
            output[idx] = out.value();
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply(
        const Z29Expr::Ptr& step,
        std::string_view cipher_var,
        const Z29Expr::Env& params,
        std::span<const Index29> input,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        std::vector<Index29> out(input.size());
        Status st = apply_into(step, cipher_var, params, input, out, interrupt);
        if (!st.ok()) {
            return st;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply(
        const PrimitiveIr& primitive,
        std::span<const Index29> param_tail,
        std::span<const Index29> input,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        std::vector<Index29> out(input.size());
        Status st = apply_into(primitive, param_tail, input, out, interrupt);
        if (!st.ok()) {
            return st;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply(
        const TheoryIr& theory,
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none(),
        std::string_view cipher_var = "x") {
        std::vector<Index29> out(input.size());
        Status st =
            apply_into(theory, input, out, params, direction, interrupt, cipher_var);
        if (!st.ok()) {
            return st;
        }
        return out;
    }

private:
    DslIrApplicator() = delete;

    [[nodiscard]] static StatusOr<Z29Expr::Env> bind_theory_params(
        const TheoryIr& theory,
        const nlohmann::json& params) {
        if (!params.is_object()) {
            return DslDiag::make(
                       DslRuleId::E040_param_domain,
                       "theory params must be a JSON object")
                .to_status();
        }
        if (params.size() != theory.params().size()) {
            return DslDiag::make(
                       DslRuleId::E040_param_domain,
                       "theory '" + theory.name() + "' expects " +
                           std::to_string(theory.params().size()) + " params, got " +
                           std::to_string(params.size()))
                .to_status();
        }
        Z29Expr::Env env;
        for (const ParamIr& p : theory.params()) {
            if (!params.contains(p.name())) {
                return DslDiag::make(
                           DslRuleId::E040_param_domain,
                           "missing theory param '" + p.name() + "'")
                    .to_status();
            }
            const nlohmann::json& v = params.at(p.name());
            if (!v.is_number_integer()) {
                return DslDiag::make(
                           DslRuleId::E040_param_domain,
                           "theory param '" + p.name() + "' must be an integer")
                    .to_status();
            }
            const std::int64_t raw = v.get<std::int64_t>();
            if (raw < 0 || raw >= Index29::modulus) {
                return DslDiag::make(
                           DslRuleId::E040_param_domain,
                           "theory param '" + p.name() + "' outside Index29 domain")
                    .to_status();
            }
            const Index29 idx{static_cast<std::uint8_t>(raw)};
            Status domain = p.check_value(idx);
            if (!domain.ok()) {
                return domain;
            }
            env.emplace(p.name(), idx);
        }
        return env;
    }
};

#endif // DSL_IR_APPLICATOR_HPP
