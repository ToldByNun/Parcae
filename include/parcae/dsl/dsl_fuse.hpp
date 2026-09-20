#ifndef DSL_FUSE_HPP
#define DSL_FUSE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/compose_ir.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

/// Inline `ComposedTheory` chains into one elementwise `TheoryIr` (docs/spec/dsl.md).
/// Flattens nested compose (depth ≤ `max_depth`); substitutes cipher + step_params.
/// Benchmark fused-vs-staged selection lands in later commits (`fusion.status`).
class DslFuse {
public:
    /// Match `ComposeTransform::max_depth`.
    static constexpr std::size_t max_depth = 8;

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

    /// Flatten nested compose step ids, then inline encrypt/decrypt expressions.
    /// `theories` / `composes` are looked up by `TheoryIr::name` / `ComposeIr::name`.
    /// Decrypt path: stages in `steps` order using each theory's `decrypt_step`.
    /// Encrypt path: reverse order using each theory's `encrypt_step`.
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

        // Outer compose step_params win over nested for the same (step, param).
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

private:
    DslFuse() = delete;

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

        // Default: keep theory param names; override via step_params value_ref.
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
};

#endif // DSL_FUSE_HPP
