#ifndef DSL_OPTIMIZE_HPP
#define DSL_OPTIMIZE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// IR peephole passes for theory DSL expressions (docs/architecture/python-transpiler.md).
/// - Const-fold pure \(\mathbb{Z}_{29}\) subtrees (incl. `inv` of nonzero constants).
/// - Hoist `inv` of cipher-independent args to named temps (`__parcae_inv_N`) for emit.
class DslOptimize {
public:
    class Hoist {
    public:
        Hoist(std::string name, Z29Expr::Ptr inv_arg)
            : name_(std::move(name)), inv_arg_(std::move(inv_arg)) {}

        [[nodiscard]] const std::string& name() const noexcept {
            return name_;
        }

        /// Argument of `inv` — emit as `Z29::inv(<emit inv_arg>)` once per kernel.
        [[nodiscard]] const Z29Expr::Ptr& inv_arg() const noexcept {
            return inv_arg_;
        }

    private:
        std::string name_;
        Z29Expr::Ptr inv_arg_;
    };

    class Result {
    public:
        Result(
            Z29Expr::Ptr expr,
            std::vector<Hoist> hoists,
            std::size_t const_folds,
            std::size_t inv_hoists)
            : expr_(std::move(expr)),
              hoists_(std::move(hoists)),
              const_folds_(const_folds),
              inv_hoists_(inv_hoists) {}

        [[nodiscard]] const Z29Expr::Ptr& expr() const noexcept {
            return expr_;
        }

        [[nodiscard]] const std::vector<Hoist>& hoists() const noexcept {
            return hoists_;
        }

        [[nodiscard]] std::size_t const_folds() const noexcept {
            return const_folds_;
        }

        [[nodiscard]] std::size_t inv_hoists() const noexcept {
            return inv_hoists_;
        }

    private:
        Z29Expr::Ptr expr_;
        std::vector<Hoist> hoists_;
        std::size_t const_folds_ = 0;
        std::size_t inv_hoists_ = 0;
    };

    class TheoryResult {
    public:
        TheoryResult(TheoryIr theory, Result encrypt, Result decrypt)
            : theory_(std::move(theory)),
              encrypt_(std::move(encrypt)),
              decrypt_(std::move(decrypt)) {}

        [[nodiscard]] const TheoryIr& theory() const noexcept {
            return theory_;
        }

        [[nodiscard]] const Result& encrypt() const noexcept {
            return encrypt_;
        }

        [[nodiscard]] const Result& decrypt() const noexcept {
            return decrypt_;
        }

    private:
        TheoryIr theory_;
        Result encrypt_;
        Result decrypt_;
    };

    /// Fold constant-only subtrees. `inv(0)` / `mod(_,0)` → E040.
    [[nodiscard]] static StatusOr<Z29Expr::Ptr> const_fold(const Z29Expr::Ptr& expr) {
        if (!expr) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "const_fold: null expr")
                .to_status();
        }
        std::size_t folds = 0;
        return const_fold_rec(*expr, folds);
    }

    /// Const-fold then hoist cipher-independent `inv` / `z29_inv` to `__parcae_inv_N`.
    [[nodiscard]] static StatusOr<Result> optimize(
        const Z29Expr::Ptr& expr,
        std::string_view cipher_var = "x") {
        if (!expr) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "optimize: null expr")
                .to_status();
        }
        if (cipher_var.empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "optimize: empty cipher_var")
                .to_status();
        }

        std::size_t folds = 0;
        StatusOr<Z29Expr::Ptr> folded = const_fold_rec(*expr, folds);
        if (!folded.ok()) {
            return folded.status();
        }

        std::vector<Hoist> hoists;
        std::size_t hoist_count = 0;
        StatusOr<Z29Expr::Ptr> out =
            hoist_inv_rec(*folded.value(), cipher_var, hoists, hoist_count);
        if (!out.ok()) {
            return out.status();
        }
        return Result{out.value(), std::move(hoists), folds, hoist_count};
    }

    /// Optimize encrypt_step and decrypt_step; returns a new TheoryIr with rewritten bodies.
    [[nodiscard]] static StatusOr<TheoryResult> optimize_theory(
        const TheoryIr& theory,
        std::string_view cipher_var = "x") {
        if (!theory.encrypt_step() || !theory.decrypt_step()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "theory '" + theory.name() +
                           "' needs encrypt_step and decrypt_step to optimize")
                .to_status();
        }
        StatusOr<Result> enc = optimize(theory.encrypt_step(), cipher_var);
        if (!enc.ok()) {
            return enc.status();
        }
        StatusOr<Result> dec = optimize(theory.decrypt_step(), cipher_var);
        if (!dec.ok()) {
            return dec.status();
        }
        StatusOr<TheoryIr> rebuilt = TheoryIr::make(
            theory.name(),
            theory.family(),
            theory.tier(),
            theory.interrupt_mode(),
            theory.params(),
            enc.value().expr(),
            dec.value().expr(),
            theory.structural_claim());
        if (!rebuilt.ok()) {
            return rebuilt.status();
        }
        return TheoryResult{rebuilt.value(), enc.value(), dec.value()};
    }

    [[nodiscard]] static bool depends_on_var(
        const Z29Expr& expr,
        std::string_view var_name) {
        using Kind = Z29Expr::Kind;
        switch (expr.kind()) {
        case Kind::Const:
            return false;
        case Kind::Var:
            return expr.name() == var_name;
        case Kind::Add:
        case Kind::Sub:
        case Kind::Mul:
        case Kind::Mod:
            return depends_on_var(*expr.left(), var_name) ||
                   depends_on_var(*expr.right(), var_name);
        case Kind::Neg:
        case Kind::Inv:
        case Kind::Atbash:
            return depends_on_var(*expr.arg(), var_name);
        case Kind::Call:
            for (const Z29Expr::Ptr& a : expr.args()) {
                if (a && depends_on_var(*a, var_name)) {
                    return true;
                }
            }
            return false;
        }
        return false;
    }

private:
    DslOptimize() = delete;

    [[nodiscard]] static StatusOr<Z29Expr::Ptr> const_fold_rec(
        const Z29Expr& expr,
        std::size_t& folds) {
        using Kind = Z29Expr::Kind;
        switch (expr.kind()) {
        case Kind::Const:
            return Z29Expr::constant(expr.const_value()).value();
        case Kind::Var:
            return Z29Expr::var(expr.name());
        case Kind::Add:
        case Kind::Sub:
        case Kind::Mul:
        case Kind::Mod: {
            StatusOr<Z29Expr::Ptr> l = const_fold_rec(*expr.left(), folds);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Z29Expr::Ptr> r = const_fold_rec(*expr.right(), folds);
            if (!r.ok()) {
                return r.status();
            }
            if (l.value()->kind() == Kind::Const && r.value()->kind() == Kind::Const) {
                const Index29 lv{l.value()->const_value()};
                const Index29 rv{r.value()->const_value()};
                if (expr.kind() == Kind::Mod) {
                    if (rv.value() == 0) {
                        return DslDiag::make(
                                   DslRuleId::E040_param_domain, "const_fold: z29_mod divisor is 0")
                            .to_status();
                    }
                    ++folds;
                    return Z29Expr::constant(static_cast<std::uint8_t>(lv.value() % rv.value()))
                        .value();
                }
                Index29 out{0};
                if (expr.kind() == Kind::Add) {
                    out = Z29::add(lv, rv);
                } else if (expr.kind() == Kind::Sub) {
                    out = Z29::sub(lv, rv);
                } else {
                    out = Z29::mul(lv, rv);
                }
                ++folds;
                return Z29Expr::constant(out.value()).value();
            }
            // Algebraic identities with one const.
            if (expr.kind() == Kind::Add) {
                if (r.value()->kind() == Kind::Const && r.value()->const_value() == 0) {
                    ++folds;
                    return l.value();
                }
                if (l.value()->kind() == Kind::Const && l.value()->const_value() == 0) {
                    ++folds;
                    return r.value();
                }
                return Z29Expr::add(l.value(), r.value());
            }
            if (expr.kind() == Kind::Sub) {
                if (r.value()->kind() == Kind::Const && r.value()->const_value() == 0) {
                    ++folds;
                    return l.value();
                }
                return Z29Expr::sub(l.value(), r.value());
            }
            if (expr.kind() == Kind::Mul) {
                if (l.value()->kind() == Kind::Const && l.value()->const_value() == 0) {
                    ++folds;
                    return Z29Expr::constant(0).value();
                }
                if (r.value()->kind() == Kind::Const && r.value()->const_value() == 0) {
                    ++folds;
                    return Z29Expr::constant(0).value();
                }
                if (l.value()->kind() == Kind::Const && l.value()->const_value() == 1) {
                    ++folds;
                    return r.value();
                }
                if (r.value()->kind() == Kind::Const && r.value()->const_value() == 1) {
                    ++folds;
                    return l.value();
                }
                return Z29Expr::mul(l.value(), r.value());
            }
            return Z29Expr::mod(l.value(), r.value());
        }
        case Kind::Neg: {
            StatusOr<Z29Expr::Ptr> a = const_fold_rec(*expr.arg(), folds);
            if (!a.ok()) {
                return a.status();
            }
            if (a.value()->kind() == Kind::Const) {
                ++folds;
                return Z29Expr::constant(Z29::neg(Index29{a.value()->const_value()}).value())
                    .value();
            }
            return Z29Expr::neg(a.value());
        }
        case Kind::Inv: {
            StatusOr<Z29Expr::Ptr> a = const_fold_rec(*expr.arg(), folds);
            if (!a.ok()) {
                return a.status();
            }
            if (a.value()->kind() == Kind::Const) {
                if (a.value()->const_value() == 0) {
                    return DslDiag::make(
                               DslRuleId::E040_param_domain, "const_fold: z29_inv(0) is undefined")
                        .to_status();
                }
                ++folds;
                return Z29Expr::constant(Z29::inv(Index29{a.value()->const_value()}).value())
                    .value();
            }
            return Z29Expr::inv(a.value());
        }
        case Kind::Atbash: {
            StatusOr<Z29Expr::Ptr> a = const_fold_rec(*expr.arg(), folds);
            if (!a.ok()) {
                return a.status();
            }
            if (a.value()->kind() == Kind::Const) {
                ++folds;
                return Z29Expr::constant(Z29::atbash(Index29{a.value()->const_value()}).value())
                    .value();
            }
            return Z29Expr::atbash(a.value());
        }
        case Kind::Call: {
            // Lower builtins to Kind nodes then fold.
            const std::string& n = expr.name();
            const auto& args = expr.args();
            if (n == "z29_add" && args.size() == 2) {
                return const_fold_rec(*Z29Expr::add(args[0], args[1]), folds);
            }
            if (n == "z29_sub" && args.size() == 2) {
                return const_fold_rec(*Z29Expr::sub(args[0], args[1]), folds);
            }
            if (n == "z29_mul" && args.size() == 2) {
                return const_fold_rec(*Z29Expr::mul(args[0], args[1]), folds);
            }
            if (n == "z29_mod" && args.size() == 2) {
                return const_fold_rec(*Z29Expr::mod(args[0], args[1]), folds);
            }
            if (n == "z29_inv" && args.size() == 1) {
                return const_fold_rec(*Z29Expr::inv(args[0]), folds);
            }
            if (n == "z29_neg" && args.size() == 1) {
                return const_fold_rec(*Z29Expr::neg(args[0]), folds);
            }
            if (n == "z29_atbash" && args.size() == 1) {
                return const_fold_rec(*Z29Expr::atbash(args[0]), folds);
            }
            std::vector<Z29Expr::Ptr> mapped;
            mapped.reserve(args.size());
            for (const Z29Expr::Ptr& a : args) {
                StatusOr<Z29Expr::Ptr> fa = const_fold_rec(*a, folds);
                if (!fa.ok()) {
                    return fa.status();
                }
                mapped.push_back(fa.value());
            }
            return Z29Expr::call(n, std::move(mapped));
        }
        }
        return DslDiag::make(DslRuleId::E032_primitive_body, "const_fold: unknown kind")
            .to_status();
    }

    [[nodiscard]] static StatusOr<Z29Expr::Ptr> hoist_inv_rec(
        const Z29Expr& expr,
        std::string_view cipher_var,
        std::vector<Hoist>& hoists,
        std::size_t& hoist_count) {
        using Kind = Z29Expr::Kind;

        auto maybe_hoist_inv = [&](const Z29Expr::Ptr& arg) -> StatusOr<Z29Expr::Ptr> {
            StatusOr<Z29Expr::Ptr> a = hoist_inv_rec(*arg, cipher_var, hoists, hoist_count);
            if (!a.ok()) {
                return a.status();
            }
            if (depends_on_var(*a.value(), cipher_var)) {
                return Z29Expr::inv(a.value());
            }
            // Already a constant ⇒ const_fold should have removed inv; keep safe.
            if (a.value()->kind() == Kind::Const) {
                if (a.value()->const_value() == 0) {
                    return DslDiag::make(
                               DslRuleId::E040_param_domain, "hoist_inv: z29_inv(0) is undefined")
                        .to_status();
                }
                return Z29Expr::constant(Z29::inv(Index29{a.value()->const_value()}).value())
                    .value();
            }
            const std::string name = "__parcae_inv_" + std::to_string(hoist_count);
            ++hoist_count;
            hoists.emplace_back(name, a.value());
            return Z29Expr::var(name);
        };

        switch (expr.kind()) {
        case Kind::Const:
            return Z29Expr::constant(expr.const_value()).value();
        case Kind::Var:
            return Z29Expr::var(expr.name());
        case Kind::Add: {
            StatusOr<Z29Expr::Ptr> l = hoist_inv_rec(*expr.left(), cipher_var, hoists, hoist_count);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Z29Expr::Ptr> r = hoist_inv_rec(*expr.right(), cipher_var, hoists, hoist_count);
            if (!r.ok()) {
                return r.status();
            }
            return Z29Expr::add(l.value(), r.value());
        }
        case Kind::Sub: {
            StatusOr<Z29Expr::Ptr> l = hoist_inv_rec(*expr.left(), cipher_var, hoists, hoist_count);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Z29Expr::Ptr> r = hoist_inv_rec(*expr.right(), cipher_var, hoists, hoist_count);
            if (!r.ok()) {
                return r.status();
            }
            return Z29Expr::sub(l.value(), r.value());
        }
        case Kind::Mul: {
            StatusOr<Z29Expr::Ptr> l = hoist_inv_rec(*expr.left(), cipher_var, hoists, hoist_count);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Z29Expr::Ptr> r = hoist_inv_rec(*expr.right(), cipher_var, hoists, hoist_count);
            if (!r.ok()) {
                return r.status();
            }
            return Z29Expr::mul(l.value(), r.value());
        }
        case Kind::Mod: {
            StatusOr<Z29Expr::Ptr> l = hoist_inv_rec(*expr.left(), cipher_var, hoists, hoist_count);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Z29Expr::Ptr> r = hoist_inv_rec(*expr.right(), cipher_var, hoists, hoist_count);
            if (!r.ok()) {
                return r.status();
            }
            return Z29Expr::mod(l.value(), r.value());
        }
        case Kind::Neg: {
            StatusOr<Z29Expr::Ptr> a = hoist_inv_rec(*expr.arg(), cipher_var, hoists, hoist_count);
            if (!a.ok()) {
                return a.status();
            }
            return Z29Expr::neg(a.value());
        }
        case Kind::Inv:
            return maybe_hoist_inv(expr.arg());
        case Kind::Atbash: {
            StatusOr<Z29Expr::Ptr> a = hoist_inv_rec(*expr.arg(), cipher_var, hoists, hoist_count);
            if (!a.ok()) {
                return a.status();
            }
            return Z29Expr::atbash(a.value());
        }
        case Kind::Call: {
            if (expr.name() == "z29_inv" && expr.args().size() == 1) {
                return maybe_hoist_inv(expr.args()[0]);
            }
            std::vector<Z29Expr::Ptr> mapped;
            mapped.reserve(expr.args().size());
            for (const Z29Expr::Ptr& a : expr.args()) {
                StatusOr<Z29Expr::Ptr> fa = hoist_inv_rec(*a, cipher_var, hoists, hoist_count);
                if (!fa.ok()) {
                    return fa.status();
                }
                mapped.push_back(fa.value());
            }
            return Z29Expr::call(expr.name(), std::move(mapped));
        }
        }
        return DslDiag::make(DslRuleId::E032_primitive_body, "hoist_inv: unknown kind")
            .to_status();
    }
};

#endif // DSL_OPTIMIZE_HPP
