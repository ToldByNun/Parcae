#ifndef Z29_EXPR_NORMALIZE_HPP
#define Z29_EXPR_NORMALIZE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_optimize.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Algebraic HotLoop normalize for smart hist (`docs/architecture/dsl-smart-hist.md`).
///
/// Pure CPU, deterministic, idempotent. Fails closed on ambiguity (leaves the
/// node unchanged). Does not rewrite across `z29_autokey_shift` Calls or change
/// `prefer_branch` Select policy — children are still normalized.
///
/// Pipeline: `DslOptimize::const_fold` → bottom-up rewrite to fixed point
/// (commute sort, atbash-as-arith, Caesar `Sub→Add+Neg`, affine/keystream
/// operand order). `TheoryShapeMatch` consumes the result (separate class).
class Z29ExprNormalize {
public:
    class Result {
    public:
        Result(Z29Expr::Ptr expr, std::size_t rewrite_steps) noexcept
            : expr_(std::move(expr)), rewrite_steps_(rewrite_steps) {}

        [[nodiscard]] const Z29Expr::Ptr& expr() const noexcept { return expr_; }

        [[nodiscard]] std::size_t rewrite_steps() const noexcept { return rewrite_steps_; }

    private:
        Z29Expr::Ptr expr_;
        std::size_t rewrite_steps_ = 0;
    };

    /// Normalize `expr` for `cipher_var` (default HotLoop cipher `"x"`).
    [[nodiscard]] static StatusOr<Z29Expr::Ptr> normalize(const Z29Expr::Ptr& expr,
                                                          std::string_view cipher_var = "x") {
        StatusOr<Result> r = normalize_counted(expr, cipher_var);
        if (!r.ok()) {
            return r.status();
        }
        return r.value().expr();
    }

    [[nodiscard]] static StatusOr<Result> normalize_counted(const Z29Expr::Ptr& expr,
                                                            std::string_view cipher_var = "x") {
        if (!expr) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: null expr")
                .to_status();
        }
        if (cipher_var.empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: empty cipher_var")
                .to_status();
        }

        StatusOr<Z29Expr::Ptr> folded = DslOptimize::const_fold(expr);
        if (!folded.ok()) {
            return folded.status();
        }

        Z29Expr::Ptr cur = folded.value();
        std::size_t steps = 0;
        constexpr std::size_t kMaxPasses = 32;
        for (std::size_t pass = 0; pass < kMaxPasses; ++pass) {
            bool changed = false;
            StatusOr<Z29Expr::Ptr> next = normalize_bottom_up(*cur, cipher_var, changed, steps);
            if (!next.ok()) {
                return next.status();
            }
            cur = next.value();
            if (!changed) {
                break;
            }
        }

        // Final const-fold (Neg(const) from Sub→Add, etc.).
        StatusOr<Z29Expr::Ptr> again = DslOptimize::const_fold(cur);
        if (!again.ok()) {
            return again.status();
        }
        if (!structurally_equal(*cur, *again.value())) {
            ++steps;
            cur = again.value();
        }
        return Result{std::move(cur), steps};
    }

    [[nodiscard]] static bool structurally_equal(const Z29Expr& a, const Z29Expr& b) {
        using Kind = Z29Expr::Kind;
        if (a.kind() != b.kind()) {
            return false;
        }
        switch (a.kind()) {
        case Kind::Const:
            return a.const_value() == b.const_value();
        case Kind::Var:
            return a.name() == b.name();
        case Kind::Call:
            if (a.name() != b.name() || a.args().size() != b.args().size()) {
                return false;
            }
            for (std::size_t i = 0; i < a.args().size(); ++i) {
                if (!a.args()[i] || !b.args()[i] ||
                    !structurally_equal(*a.args()[i], *b.args()[i])) {
                    return false;
                }
            }
            return true;
        case Kind::Select:
            if (a.prefer_branch() != b.prefer_branch()) {
                return false;
            }
            return a.cond() && b.cond() && a.if_true() && b.if_true() && a.if_false() &&
                   b.if_false() && structurally_equal(*a.cond(), *b.cond()) &&
                   structurally_equal(*a.if_true(), *b.if_true()) &&
                   structurally_equal(*a.if_false(), *b.if_false());
        default:
            break;
        }
        if (Z29Expr::is_binary(a.kind())) {
            return a.left() && b.left() && a.right() && b.right() &&
                   structurally_equal(*a.left(), *b.left()) &&
                   structurally_equal(*a.right(), *b.right());
        }
        if (Z29Expr::is_unary(a.kind())) {
            return a.arg() && b.arg() && structurally_equal(*a.arg(), *b.arg());
        }
        return false;
    }

private:
    Z29ExprNormalize() = delete;

    [[nodiscard]] static StatusOr<Z29Expr::Ptr>
    normalize_bottom_up(const Z29Expr& expr, std::string_view cipher_var, bool& changed,
                        std::size_t& steps) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() == Kind::Const) {
            return Z29Expr::constant(expr.const_value());
        }
        if (expr.kind() == Kind::Var) {
            return Z29Expr::var(expr.name());
        }

        if (expr.kind() == Kind::Call) {
            std::vector<Z29Expr::Ptr> args;
            args.reserve(expr.args().size());
            for (const Z29Expr::Ptr& a : expr.args()) {
                if (!a) {
                    return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: null call arg")
                        .to_status();
                }
                bool child_changed = false;
                StatusOr<Z29Expr::Ptr> na =
                    normalize_bottom_up(*a, cipher_var, child_changed, steps);
                if (!na.ok()) {
                    return na.status();
                }
                if (child_changed) {
                    changed = true;
                }
                args.push_back(na.value());
            }
            // Autokey: normalize args only (fail closed — keep Call).
            if (expr.name() == "z29_autokey_shift") {
                return Z29Expr::call(expr.name(), std::move(args));
            }
            Z29Expr::Ptr call_node = Z29Expr::call(expr.name(), std::move(args));
            return rewrite_local(*call_node, cipher_var, changed, steps);
        }

        if (expr.kind() == Kind::Select) {
            if (!expr.cond() || !expr.if_true() || !expr.if_false()) {
                return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: null select arm")
                    .to_status();
            }
            bool c0 = false;
            bool c1 = false;
            bool c2 = false;
            StatusOr<Z29Expr::Ptr> c =
                normalize_bottom_up(*expr.cond(), cipher_var, c0, steps);
            if (!c.ok()) {
                return c.status();
            }
            StatusOr<Z29Expr::Ptr> t =
                normalize_bottom_up(*expr.if_true(), cipher_var, c1, steps);
            if (!t.ok()) {
                return t.status();
            }
            StatusOr<Z29Expr::Ptr> f =
                normalize_bottom_up(*expr.if_false(), cipher_var, c2, steps);
            if (!f.ok()) {
                return f.status();
            }
            if (c0 || c1 || c2) {
                changed = true;
            }
            Z29Expr::Ptr sel =
                Z29Expr::make_select(c.value(), t.value(), f.value(), expr.prefer_branch());
            // No structural rewrite of prefer_branch Selects — children only.
            return sel;
        }

        if (Z29Expr::is_unary(expr.kind())) {
            if (!expr.arg()) {
                return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: null unary arg")
                    .to_status();
            }
            bool child_changed = false;
            StatusOr<Z29Expr::Ptr> a =
                normalize_bottom_up(*expr.arg(), cipher_var, child_changed, steps);
            if (!a.ok()) {
                return a.status();
            }
            if (child_changed) {
                changed = true;
            }
            Z29Expr::Ptr u = Z29Expr::make_unary_kind(expr.kind(), a.value());
            return rewrite_local(*u, cipher_var, changed, steps);
        }

        if (Z29Expr::is_binary(expr.kind())) {
            if (!expr.left() || !expr.right()) {
                return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: null binary child")
                    .to_status();
            }
            bool lch = false;
            bool rch = false;
            StatusOr<Z29Expr::Ptr> l =
                normalize_bottom_up(*expr.left(), cipher_var, lch, steps);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Z29Expr::Ptr> r =
                normalize_bottom_up(*expr.right(), cipher_var, rch, steps);
            if (!r.ok()) {
                return r.status();
            }
            if (lch || rch) {
                changed = true;
            }
            Z29Expr::Ptr b = Z29Expr::make_binary(expr.kind(), l.value(), r.value());
            return rewrite_local(*b, cipher_var, changed, steps);
        }

        return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: unsupported kind")
            .to_status();
    }

    [[nodiscard]] static StatusOr<Z29Expr::Ptr>
    rewrite_local(const Z29Expr& expr, std::string_view cipher_var, bool& changed,
                  std::size_t& steps) {
        using Kind = Z29Expr::Kind;

        // --- Atbash-as-arith / builtin Call / BitNot ------------------------
        if (expr.kind() == Kind::Call && expr.name() == "z29_atbash" && expr.args().size() == 1 &&
            expr.args()[0]) {
            ++steps;
            changed = true;
            return Z29Expr::atbash(expr.args()[0]);
        }
        if (expr.kind() == Kind::BitNot && expr.arg()) {
            // On Z_29, (~x) mod 29 == 28 - x == atbash(x).
            ++steps;
            changed = true;
            return Z29Expr::atbash(expr.arg());
        }
        if (expr.kind() == Kind::Sub && expr.left() && expr.right() &&
            expr.left()->kind() == Kind::Const && expr.left()->const_value() == 28) {
            ++steps;
            changed = true;
            return Z29Expr::atbash(expr.right());
        }
        if (expr.kind() == Kind::Add && expr.left() && expr.right()) {
            if (expr.left()->kind() == Kind::Const && expr.left()->const_value() == 28 &&
                expr.right()->kind() == Kind::Neg && expr.right()->arg()) {
                ++steps;
                changed = true;
                return Z29Expr::atbash(expr.right()->arg());
            }
            if (expr.right()->kind() == Kind::Const && expr.right()->const_value() == 28 &&
                expr.left()->kind() == Kind::Neg && expr.left()->arg()) {
                ++steps;
                changed = true;
                return Z29Expr::atbash(expr.left()->arg());
            }
        }

        // --- Caesar: Sub(cipher_side, shift) → Add(cipher_side, Neg(shift)) -
        if (expr.kind() == Kind::Sub && expr.left() && expr.right()) {
            const bool left_c = DslOptimize::depends_on_var(*expr.left(), cipher_var);
            const bool right_c = DslOptimize::depends_on_var(*expr.right(), cipher_var);
            if (left_c && !right_c) {
                ++steps;
                changed = true;
                return Z29Expr::add(expr.left(), Z29Expr::neg(expr.right()));
            }
        }

        // --- Commute Add / Mul into canonical operand order -----------------
        if (expr.kind() == Kind::Add || expr.kind() == Kind::Mul) {
            if (expr.left() && expr.right() &&
                should_swap_commute(expr.kind(), *expr.left(), *expr.right(), cipher_var)) {
                ++steps;
                changed = true;
                return Z29Expr::make_binary(expr.kind(), expr.right(), expr.left());
            }
        }

        // --- Inner keystream: Mul(i, b1) → Mul(b1, i) when b1 independent of i
        if (expr.kind() == Kind::Mul && expr.left() && expr.right()) {
            if (expr.left()->kind() == Kind::Var && expr.left()->name() == "i" &&
                !DslOptimize::depends_on_var(*expr.right(), "i")) {
                ++steps;
                changed = true;
                return Z29Expr::mul(expr.right(), expr.left());
            }
        }

        // --- Keystream Add: prefer Add(b0, Mul(b1,i)) over Add(Mul(b1,i), b0)
        if (expr.kind() == Kind::Add && expr.left() && expr.right()) {
            if (is_mul_of_i(*expr.left()) && !DslOptimize::depends_on_var(*expr.right(), "i")) {
                ++steps;
                changed = true;
                return Z29Expr::add(expr.right(), expr.left());
            }
        }

        // Clone rebuilt node (already has normalized children).
        return clone_shallow(expr);
    }

    [[nodiscard]] static bool is_mul_of_i(const Z29Expr& expr) {
        return expr.kind() == Z29Expr::Kind::Mul && expr.right() &&
               expr.right()->kind() == Z29Expr::Kind::Var && expr.right()->name() == "i";
    }

    /// Commute policy: cipher-bearing side left for Add; coeff left / cipher right for Mul;
    /// else lexicographic structural key.
    [[nodiscard]] static bool should_swap_commute(Z29Expr::Kind kind, const Z29Expr& left,
                                                  const Z29Expr& right,
                                                  std::string_view cipher_var) {
        const bool l_c = DslOptimize::depends_on_var(left, cipher_var);
        const bool r_c = DslOptimize::depends_on_var(right, cipher_var);
        if (kind == Z29Expr::Kind::Add) {
            if (!l_c && r_c) {
                return true;
            }
            if (l_c && !r_c) {
                return false;
            }
        }
        if (kind == Z29Expr::Kind::Mul) {
            // Prefer Mul(coeff, cipher_expr): coeff has no cipher, cipher side has cipher.
            if (l_c && !r_c) {
                return true;
            }
            if (!l_c && r_c) {
                return false;
            }
        }
        return canonical_key(left) > canonical_key(right);
    }

    [[nodiscard]] static std::string canonical_key(const Z29Expr& expr) {
        using Kind = Z29Expr::Kind;
        std::string out;
        out.push_back(static_cast<char>(static_cast<unsigned>(expr.kind())));
        switch (expr.kind()) {
        case Kind::Const:
            out.push_back(static_cast<char>(expr.const_value()));
            break;
        case Kind::Var:
            out += expr.name();
            break;
        case Kind::Call:
            out += expr.name();
            for (const Z29Expr::Ptr& a : expr.args()) {
                out.push_back('|');
                if (a) {
                    out += canonical_key(*a);
                }
            }
            break;
        case Kind::Select:
            out.push_back(expr.prefer_branch() ? 'B' : 'b');
            if (expr.cond()) {
                out += canonical_key(*expr.cond());
            }
            out.push_back(';');
            if (expr.if_true()) {
                out += canonical_key(*expr.if_true());
            }
            out.push_back(';');
            if (expr.if_false()) {
                out += canonical_key(*expr.if_false());
            }
            break;
        default:
            if (Z29Expr::is_binary(expr.kind())) {
                if (expr.left()) {
                    out += canonical_key(*expr.left());
                }
                out.push_back(',');
                if (expr.right()) {
                    out += canonical_key(*expr.right());
                }
            } else if (Z29Expr::is_unary(expr.kind()) && expr.arg()) {
                out += canonical_key(*expr.arg());
            }
            break;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<Z29Expr::Ptr> clone_shallow(const Z29Expr& expr) {
        using Kind = Z29Expr::Kind;
        switch (expr.kind()) {
        case Kind::Const:
            return Z29Expr::constant(expr.const_value());
        case Kind::Var:
            return Z29Expr::var(expr.name());
        case Kind::Call: {
            std::vector<Z29Expr::Ptr> args = expr.args();
            return Z29Expr::call(expr.name(), std::move(args));
        }
        case Kind::Select:
            return Z29Expr::make_select(expr.cond(), expr.if_true(), expr.if_false(),
                                        expr.prefer_branch());
        default:
            break;
        }
        if (Z29Expr::is_binary(expr.kind())) {
            return Z29Expr::make_binary(expr.kind(), expr.left(), expr.right());
        }
        if (Z29Expr::is_unary(expr.kind())) {
            return Z29Expr::make_unary_kind(expr.kind(), expr.arg());
        }
        return DslDiag::make(DslRuleId::E032_primitive_body, "normalize: clone unsupported")
            .to_status();
    }
};

#endif // Z29_EXPR_NORMALIZE_HPP
