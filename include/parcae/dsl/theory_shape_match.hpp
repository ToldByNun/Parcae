#ifndef THEORY_SHAPE_MATCH_HPP
#define THEORY_SHAPE_MATCH_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_optimize.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/dsl/z29_expr_normalize.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

/// Algebraic ShapeId match for smart hist (`docs/architecture/dsl-smart-hist.md`).
///
/// Input is normalized via `Z29ExprNormalize` (unless `already_normalized`).
/// Matching is by math only — never by theory name / family / catalog API.
///
/// Prefer order (first hit wins after special cases):
/// prefer_branch → Unknown; autokey → Autokey; then Atbash → Caesar → Affine →
/// LinearKeystream → PolyKeystream → FxOnly → KeyedGeneral → Unknown.
class TheoryShapeMatch {
public:
    enum class ShapeId : std::uint8_t {
        Atbash = 0,
        Caesar,
        Affine,
        LinearKeystream,
        FxOnly,
        KeyedGeneral,
        Autokey,
        PolyKeystream,
        Unknown,
    };

    class Match {
    public:
        Match(ShapeId shape, std::string reason, Z29Expr::Ptr normalized) noexcept
            : shape_(shape), reason_(std::move(reason)), normalized_(std::move(normalized)) {}

        [[nodiscard]] ShapeId shape() const noexcept { return shape_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] const Z29Expr::Ptr& normalized() const noexcept { return normalized_; }

        /// Caesar: param/const shift binding name (empty if const-only / Neg-wrapped).
        [[nodiscard]] const std::string& shift_name() const noexcept { return shift_name_; }

        [[nodiscard]] const std::string& a_name() const noexcept { return a_name_; }

        [[nodiscard]] const std::string& b_name() const noexcept { return b_name_; }

        [[nodiscard]] const std::string& b0_name() const noexcept { return b0_name_; }

        [[nodiscard]] const std::string& b1_name() const noexcept { return b1_name_; }

        [[nodiscard]] bool cipher_minus_ks() const noexcept { return cipher_minus_ks_; }

        [[nodiscard]] bool has_const_shift() const noexcept { return has_const_shift_; }

        [[nodiscard]] std::uint8_t const_shift() const noexcept { return const_shift_; }

        [[nodiscard]] bool has_const_a() const noexcept { return has_const_a_; }

        [[nodiscard]] std::uint8_t const_a() const noexcept { return const_a_; }

        [[nodiscard]] bool has_const_b() const noexcept { return has_const_b_; }

        [[nodiscard]] std::uint8_t const_b() const noexcept { return const_b_; }

        [[nodiscard]] bool has_const_b0() const noexcept { return has_const_b0_; }

        [[nodiscard]] std::uint8_t const_b0() const noexcept { return const_b0_; }

        [[nodiscard]] bool has_const_b1() const noexcept { return has_const_b1_; }

        [[nodiscard]] std::uint8_t const_b1() const noexcept { return const_b1_; }

        /// Linear coeffs complete when each of b0/b1 is either a param name or a const.
        [[nodiscard]] bool linear_coeffs_ok() const noexcept {
            return (has_const_b0_ || !b0_name_.empty()) && (has_const_b1_ || !b1_name_.empty());
        }

        void set_shift_name(std::string name) { shift_name_ = std::move(name); }

        void set_const_shift(std::uint8_t v) {
            has_const_shift_ = true;
            const_shift_ = v;
        }

        void set_affine(std::string a, std::string b) {
            a_name_ = std::move(a);
            b_name_ = std::move(b);
        }

        void set_const_a(std::uint8_t v) {
            has_const_a_ = true;
            const_a_ = v;
        }

        void set_const_b(std::uint8_t v) {
            has_const_b_ = true;
            const_b_ = v;
        }

        /// True when match is decrypt `inv(a)·(x−b)` (shape hist twin eligible).
        void set_affine_decrypt(bool v) noexcept { affine_decrypt_ = v; }

        [[nodiscard]] bool affine_decrypt() const noexcept { return affine_decrypt_; }

        void set_linear(std::string b0, std::string b1, bool cipher_minus_ks) {
            b0_name_ = std::move(b0);
            b1_name_ = std::move(b1);
            cipher_minus_ks_ = cipher_minus_ks;
        }

        void set_const_b0(std::uint8_t v) {
            has_const_b0_ = true;
            const_b0_ = v;
        }

        void set_const_b1(std::uint8_t v) {
            has_const_b1_ = true;
            const_b1_ = v;
        }

    private:
        ShapeId shape_ = ShapeId::Unknown;
        std::string reason_;
        Z29Expr::Ptr normalized_;
        std::string shift_name_;
        std::string a_name_;
        std::string b_name_;
        std::string b0_name_;
        std::string b1_name_;
        bool cipher_minus_ks_ = false;
        bool has_const_shift_ = false;
        std::uint8_t const_shift_ = 0;
        bool has_const_a_ = false;
        std::uint8_t const_a_ = 0;
        bool has_const_b_ = false;
        std::uint8_t const_b_ = 0;
        bool has_const_b0_ = false;
        std::uint8_t const_b0_ = 0;
        bool has_const_b1_ = false;
        std::uint8_t const_b1_ = 0;
        bool affine_decrypt_ = false;
    };

    [[nodiscard]] static const char* shape_str(ShapeId id) noexcept {
        switch (id) {
        case ShapeId::Atbash:
            return "Atbash";
        case ShapeId::Caesar:
            return "Caesar";
        case ShapeId::Affine:
            return "Affine";
        case ShapeId::LinearKeystream:
            return "LinearKeystream";
        case ShapeId::FxOnly:
            return "FxOnly";
        case ShapeId::KeyedGeneral:
            return "KeyedGeneral";
        case ShapeId::Autokey:
            return "Autokey";
        case ShapeId::PolyKeystream:
            return "PolyKeystream";
        case ShapeId::Unknown:
            return "Unknown";
        }
        return "Unknown";
    }

    /// Match decrypt expr. Normalizes first unless `already_normalized`.
    [[nodiscard]] static StatusOr<Match> match(const Z29Expr::Ptr& expr,
                                               std::string_view cipher_var = "x",
                                               bool already_normalized = false) {
        if (!expr) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "TheoryShapeMatch: null expr")
                .to_status();
        }
        if (cipher_var.empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryShapeMatch: empty cipher_var")
                .to_status();
        }

        Z29Expr::Ptr norm = expr;
        if (!already_normalized) {
            StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr, cipher_var);
            if (!n.ok()) {
                return n.status();
            }
            norm = n.value();
        }
        return match_normalized(*norm, norm, cipher_var);
    }

    /// Match `theory.decrypt_step()` (name/family ignored).
    [[nodiscard]] static StatusOr<Match> match_theory(const TheoryIr& theory,
                                                      std::string_view cipher_var = "x") {
        if (!theory.decrypt_step()) {
            return Match{ShapeId::Unknown, "missing decrypt_step", nullptr};
        }
        return match(theory.decrypt_step(), cipher_var, /*already_normalized=*/false);
    }

private:
    TheoryShapeMatch() = delete;

    [[nodiscard]] static Match match_normalized(const Z29Expr& expr, Z29Expr::Ptr owned,
                                                std::string_view cipher_var) {
        if (has_prefer_branch(expr)) {
            return Match{ShapeId::Unknown, "prefer_branch Select is Unknown until specialized",
                         std::move(owned)};
        }
        if (has_autokey(expr)) {
            return Match{ShapeId::Autokey, "contains z29_autokey_shift", std::move(owned)};
        }
        if (!DslOptimize::depends_on_var(expr, cipher_var)) {
            return Match{ShapeId::Unknown, "decrypt does not reference cipher_var",
                         std::move(owned)};
        }

        if (std::optional<Match> m = try_atbash(expr, owned, cipher_var)) {
            return std::move(*m);
        }
        if (std::optional<Match> m = try_caesar(expr, owned, cipher_var)) {
            return std::move(*m);
        }
        // Decrypt form inv(a)·(x−b) before encrypt-form a·x+b.
        if (std::optional<Match> m = try_affine_decrypt(expr, owned, cipher_var)) {
            return std::move(*m);
        }
        if (std::optional<Match> m = try_affine(expr, owned, cipher_var)) {
            return std::move(*m);
        }
        if (std::optional<Match> m = try_linear_keystream(expr, owned, cipher_var)) {
            return std::move(*m);
        }
        if (std::optional<Match> m = try_poly_keystream(expr, owned, cipher_var)) {
            return std::move(*m);
        }

        if (!DslOptimize::depends_on_var(expr, "i")) {
            return Match{ShapeId::FxOnly, "f(x;params) without stream index", std::move(owned)};
        }
        return Match{ShapeId::KeyedGeneral, "uses i but not linear/poly keystream shape",
                     std::move(owned)};
    }

    [[nodiscard]] static bool has_autokey(const Z29Expr& expr) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() == Kind::Call && expr.name() == "z29_autokey_shift") {
            return true;
        }
        if (expr.kind() == Kind::Call) {
            for (const Z29Expr::Ptr& a : expr.args()) {
                if (a && has_autokey(*a)) {
                    return true;
                }
            }
            return false;
        }
        if (expr.kind() == Kind::Select) {
            return (expr.cond() && has_autokey(*expr.cond())) ||
                   (expr.if_true() && has_autokey(*expr.if_true())) ||
                   (expr.if_false() && has_autokey(*expr.if_false()));
        }
        if (Z29Expr::is_binary(expr.kind())) {
            return (expr.left() && has_autokey(*expr.left())) ||
                   (expr.right() && has_autokey(*expr.right()));
        }
        if (Z29Expr::is_unary(expr.kind())) {
            return expr.arg() && has_autokey(*expr.arg());
        }
        return false;
    }

    [[nodiscard]] static bool has_prefer_branch(const Z29Expr& expr) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() == Kind::Select && expr.prefer_branch()) {
            return true;
        }
        if (expr.kind() == Kind::Call) {
            for (const Z29Expr::Ptr& a : expr.args()) {
                if (a && has_prefer_branch(*a)) {
                    return true;
                }
            }
            return false;
        }
        if (expr.kind() == Kind::Select) {
            return (expr.cond() && has_prefer_branch(*expr.cond())) ||
                   (expr.if_true() && has_prefer_branch(*expr.if_true())) ||
                   (expr.if_false() && has_prefer_branch(*expr.if_false()));
        }
        if (Z29Expr::is_binary(expr.kind())) {
            return (expr.left() && has_prefer_branch(*expr.left())) ||
                   (expr.right() && has_prefer_branch(*expr.right()));
        }
        if (Z29Expr::is_unary(expr.kind())) {
            return expr.arg() && has_prefer_branch(*expr.arg());
        }
        return false;
    }

    [[nodiscard]] static bool is_cipher_var(const Z29Expr& expr, std::string_view cipher_var) {
        return expr.kind() == Z29Expr::Kind::Var && expr.name() == cipher_var;
    }

    [[nodiscard]] static std::optional<Match>
    try_atbash(const Z29Expr& expr, Z29Expr::Ptr owned, std::string_view cipher_var) {
        if (expr.kind() == Z29Expr::Kind::Atbash && expr.arg() &&
            is_cipher_var(*expr.arg(), cipher_var)) {
            return Match{ShapeId::Atbash, "Atbash(cipher)", std::move(owned)};
        }
        return std::nullopt;
    }

    /// After normalize: `Add(cipher, shift)` with shift independent of cipher / `i`.
    [[nodiscard]] static std::optional<Match>
    try_caesar(const Z29Expr& expr, Z29Expr::Ptr owned, std::string_view cipher_var) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() != Kind::Add || !expr.left() || !expr.right()) {
            return std::nullopt;
        }
        if (!is_cipher_var(*expr.left(), cipher_var)) {
            return std::nullopt;
        }
        const Z29Expr& shift = *expr.right();
        if (DslOptimize::depends_on_var(shift, cipher_var) ||
            DslOptimize::depends_on_var(shift, "i")) {
            return std::nullopt;
        }
        // Affine uses Mul on the non-cipher side — not Caesar.
        if (shift.kind() == Kind::Mul) {
            return std::nullopt;
        }

        if (shift.kind() == Kind::Var) {
            Match m{ShapeId::Caesar, "Add(cipher, shift_param)", std::move(owned)};
            m.set_shift_name(shift.name());
            return m;
        }
        if (shift.kind() == Kind::Const) {
            Match m{ShapeId::Caesar, "Add(cipher, const)", std::move(owned)};
            m.set_const_shift(shift.const_value());
            return m;
        }
        if (shift.kind() == Kind::Neg && shift.arg() && shift.arg()->kind() == Kind::Var) {
            Match m{ShapeId::Caesar, "Add(cipher, Neg(shift_param))", std::move(owned)};
            m.set_shift_name(shift.arg()->name());
            return m;
        }
        if (shift.kind() == Kind::Neg && shift.arg() && shift.arg()->kind() == Kind::Const) {
            Match m{ShapeId::Caesar, "Add(cipher, Neg(const))", std::move(owned)};
            m.set_const_shift(shift.arg()->const_value());
            return m;
        }
        return std::nullopt;
    }

    /// Decrypt Affine: `Mul(Inv(a), Sub/Add(cipher, ±b))` or `Mul(Inv(a), cipher)`.
    [[nodiscard]] static std::optional<Match>
    try_affine_decrypt(const Z29Expr& expr, Z29Expr::Ptr owned, std::string_view cipher_var) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() != Kind::Mul || !expr.left() || !expr.right()) {
            return std::nullopt;
        }
        const Z29Expr* inv_side = nullptr;
        const Z29Expr* body = nullptr;
        if (expr.left()->kind() == Kind::Inv) {
            inv_side = expr.left().get();
            body = expr.right().get();
        } else if (expr.right()->kind() == Kind::Inv) {
            inv_side = expr.right().get();
            body = expr.left().get();
        } else {
            return std::nullopt;
        }
        if (!inv_side->arg() || !body) {
            return std::nullopt;
        }
        const Z29Expr& a_expr = *inv_side->arg();
        std::string a_name;
        std::optional<std::uint8_t> const_a;
        if (a_expr.kind() == Kind::Var) {
            a_name = a_expr.name();
        } else if (a_expr.kind() == Kind::Const) {
            if (a_expr.const_value() == 0 || !Z29::try_inv(Index29{a_expr.const_value()}).ok()) {
                return std::nullopt;
            }
            const_a = static_cast<std::uint8_t>(a_expr.const_value());
        } else {
            return std::nullopt;
        }

        std::string b_name;
        std::optional<std::uint8_t> const_b;
        if (is_cipher_var(*body, cipher_var)) {
            const_b = static_cast<std::uint8_t>(0);
        } else if ((body->kind() == Kind::Sub || body->kind() == Kind::Add) && body->left() &&
                   body->right() && is_cipher_var(*body->left(), cipher_var) &&
                   !DslOptimize::depends_on_var(*body->right(), cipher_var) &&
                   !DslOptimize::depends_on_var(*body->right(), "i")) {
            const Z29Expr* b_expr = body->right().get();
            // Add(cipher, Neg(b)) after Caesar-style normalize of Sub(cipher, b).
            if (body->kind() == Kind::Add && b_expr->kind() == Kind::Neg && b_expr->arg()) {
                b_expr = b_expr->arg().get();
            } else if (body->kind() == Kind::Add) {
                // Add(cipher, +b) is not standard affine decrypt.
                return std::nullopt;
            }
            if (b_expr->kind() == Kind::Var) {
                b_name = b_expr->name();
            } else if (b_expr->kind() == Kind::Const) {
                const_b = static_cast<std::uint8_t>(b_expr->const_value());
            } else {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }

        Match m{ShapeId::Affine, "Mul(Inv(a), cipher-b) decrypt", std::move(owned)};
        m.set_affine(std::move(a_name), std::move(b_name));
        m.set_affine_decrypt(true);
        if (const_a) {
            m.set_const_a(*const_a);
        }
        if (const_b) {
            m.set_const_b(*const_b);
        }
        return m;
    }

    /// `Add(Mul(a, cipher), b)` or `Mul(a, cipher)` with invertible const `a` (or param `a`).
    [[nodiscard]] static std::optional<Match>
    try_affine(const Z29Expr& expr, Z29Expr::Ptr owned, std::string_view cipher_var) {
        using Kind = Z29Expr::Kind;
        const Z29Expr* mul = nullptr;
        const Z29Expr* b_expr = nullptr;
        bool b_is_zero = false;

        if (expr.kind() == Kind::Mul) {
            mul = &expr;
            b_is_zero = true;
        } else if (expr.kind() == Kind::Add && expr.left() && expr.right()) {
            if (expr.left()->kind() == Kind::Mul &&
                !DslOptimize::depends_on_var(*expr.right(), cipher_var) &&
                !DslOptimize::depends_on_var(*expr.right(), "i")) {
                mul = expr.left().get();
                b_expr = expr.right().get();
            } else {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }

        if (!mul || !mul->left() || !mul->right()) {
            return std::nullopt;
        }
        if (!is_cipher_var(*mul->right(), cipher_var)) {
            return std::nullopt;
        }
        if (DslOptimize::depends_on_var(*mul->left(), cipher_var) ||
            DslOptimize::depends_on_var(*mul->left(), "i")) {
            return std::nullopt;
        }

        const Z29Expr& a_expr = *mul->left();
        std::string a_name;
        std::optional<std::uint8_t> const_a;
        if (a_expr.kind() == Kind::Const) {
            if (a_expr.const_value() == 0 || !Z29::try_inv(Index29{a_expr.const_value()}).ok()) {
                return std::nullopt; // non-invertible — not Affine
            }
            const_a = static_cast<std::uint8_t>(a_expr.const_value());
        } else if (a_expr.kind() == Kind::Var) {
            a_name = a_expr.name();
        } else {
            return std::nullopt;
        }

        std::string b_name;
        std::optional<std::uint8_t> const_b;
        if (b_is_zero) {
            const_b = static_cast<std::uint8_t>(0);
        } else if (b_expr->kind() == Kind::Const) {
            const_b = static_cast<std::uint8_t>(b_expr->const_value());
        } else if (b_expr->kind() == Kind::Var) {
            b_name = b_expr->name();
        } else {
            return std::nullopt;
        }

        Match m{ShapeId::Affine, "Add(Mul(a,cipher), b)", std::move(owned)};
        m.set_affine(std::move(a_name), std::move(b_name));
        if (const_a) {
            m.set_const_a(*const_a);
        }
        if (const_b) {
            m.set_const_b(*const_b);
        }
        return m;
    }

    /// Linear `b0 + b1·i` coeffs (param name and/or folded const). Empty name ⇒ const.
    struct LinearCoeffs {
        std::string b0_name;
        std::string b1_name;
        bool has_const_b0 = false;
        std::uint8_t const_b0 = 0;
        bool has_const_b1 = false;
        std::uint8_t const_b1 = 0;

        [[nodiscard]] bool ok() const noexcept {
            return (has_const_b0 || !b0_name.empty()) && (has_const_b1 || !b1_name.empty());
        }
    };

    static void apply_linear_coeffs(Match& m, const LinearCoeffs& c, bool cipher_minus_ks) {
        m.set_linear(c.b0_name, c.b1_name, cipher_minus_ks);
        if (c.has_const_b0) {
            m.set_const_b0(c.const_b0);
        }
        if (c.has_const_b1) {
            m.set_const_b1(c.const_b1);
        }
    }

    /// Match scalar coeff: Var param, Const literal, or bare `i` as Mul(1,i) via caller.
    [[nodiscard]] static bool match_scalar_coeff(const Z29Expr& e, std::string& name_out,
                                                 bool& has_const_out, std::uint8_t& const_out) {
        using Kind = Z29Expr::Kind;
        if (e.kind() == Kind::Var && e.name() != "i") {
            name_out = e.name();
            has_const_out = false;
            return true;
        }
        if (e.kind() == Kind::Const) {
            name_out.clear();
            has_const_out = true;
            const_out = static_cast<std::uint8_t>(e.const_value());
            return true;
        }
        return false;
    }

    /// Match `b1*i` / `i*b1` / bare `i` (⇒ b1=1).
    [[nodiscard]] static std::optional<LinearCoeffs> match_mul_b1_i_term(const Z29Expr& e) {
        using Kind = Z29Expr::Kind;
        LinearCoeffs out;
        out.has_const_b0 = true;
        out.const_b0 = 0;
        if (e.kind() == Kind::Var && e.name() == "i") {
            out.has_const_b1 = true;
            out.const_b1 = 1;
            return out;
        }
        if (e.kind() != Kind::Mul || !e.left() || !e.right()) {
            return std::nullopt;
        }
        const Z29Expr& ml = *e.left();
        const Z29Expr& mr = *e.right();
        const bool l_i = ml.kind() == Kind::Var && ml.name() == "i";
        const bool r_i = mr.kind() == Kind::Var && mr.name() == "i";
        if (r_i && match_scalar_coeff(ml, out.b1_name, out.has_const_b1, out.const_b1)) {
            return out;
        }
        if (l_i && match_scalar_coeff(mr, out.b1_name, out.has_const_b1, out.const_b1)) {
            return out;
        }
        return std::nullopt;
    }

    /// Match `b0 + b1*i` (any commute), bare `b1*i`, bare `i`, or `b0 + i`.
    [[nodiscard]] static std::optional<LinearCoeffs> match_b0_b1_i(const Z29Expr& ks) {
        using Kind = Z29Expr::Kind;

        // Bare Mul(b1,i) / Mul(i,b1) / bare i → implied b0=0.
        if (std::optional<LinearCoeffs> bare = match_mul_b1_i_term(ks)) {
            return bare;
        }

        if (ks.kind() != Kind::Add || !ks.left() || !ks.right()) {
            return std::nullopt;
        }

        auto try_sides = [](const Z29Expr& b0_side,
                            const Z29Expr& mul_side) -> std::optional<LinearCoeffs> {
            LinearCoeffs out;
            if (!match_scalar_coeff(b0_side, out.b0_name, out.has_const_b0, out.const_b0)) {
                return std::nullopt;
            }
            std::optional<LinearCoeffs> mul = match_mul_b1_i_term(mul_side);
            if (!mul) {
                return std::nullopt;
            }
            // mul term carries implied b0=0 — take only b1 from it.
            out.b1_name = std::move(mul->b1_name);
            out.has_const_b1 = mul->has_const_b1;
            out.const_b1 = mul->const_b1;
            // Reject identical param names for b0 and b1.
            if (!out.has_const_b0 && !out.has_const_b1 && out.b0_name == out.b1_name) {
                return std::nullopt;
            }
            return out;
        };

        if (std::optional<LinearCoeffs> a = try_sides(*ks.left(), *ks.right())) {
            return a;
        }
        if (std::optional<LinearCoeffs> b = try_sides(*ks.right(), *ks.left())) {
            return b;
        }
        return std::nullopt;
    }

    [[nodiscard]] static std::optional<Match>
    try_linear_keystream(const Z29Expr& expr, Z29Expr::Ptr owned, std::string_view cipher_var) {
        using Kind = Z29Expr::Kind;
        if ((expr.kind() != Kind::Add && expr.kind() != Kind::Sub) || !expr.left() ||
            !expr.right()) {
            return std::nullopt;
        }

        // Prefer normalized Add(cipher, ±ks).
        if (expr.kind() == Kind::Add && is_cipher_var(*expr.left(), cipher_var)) {
            const Z29Expr& rhs = *expr.right();
            if (DslOptimize::depends_on_var(rhs, cipher_var)) {
                return std::nullopt;
            }
            if (rhs.kind() == Kind::Neg && rhs.arg()) {
                std::optional<LinearCoeffs> coeffs = match_b0_b1_i(*rhs.arg());
                if (coeffs && coeffs->ok()) {
                    Match m{ShapeId::LinearKeystream, "Add(cipher, Neg(linear_ks))",
                            std::move(owned)};
                    apply_linear_coeffs(m, *coeffs, /*cipher_minus_ks=*/true);
                    return m;
                }
            } else {
                std::optional<LinearCoeffs> coeffs = match_b0_b1_i(rhs);
                if (coeffs && coeffs->ok()) {
                    Match m{ShapeId::LinearKeystream, "Add(cipher, linear_ks)", std::move(owned)};
                    apply_linear_coeffs(m, *coeffs, /*cipher_minus_ks=*/false);
                    return m;
                }
            }
            return std::nullopt;
        }

        // Unnormalized Sub(cipher, ks).
        if (expr.kind() == Kind::Sub && is_cipher_var(*expr.left(), cipher_var) &&
            !DslOptimize::depends_on_var(*expr.right(), cipher_var)) {
            std::optional<LinearCoeffs> coeffs = match_b0_b1_i(*expr.right());
            if (coeffs && coeffs->ok()) {
                Match m{ShapeId::LinearKeystream, "Sub(cipher, linear_ks)", std::move(owned)};
                apply_linear_coeffs(m, *coeffs, /*cipher_minus_ks=*/true);
                return m;
            }
        }
        return std::nullopt;
    }

    /// Low-degree: `b0 + b1*i + b2*i*i` as Add(Add(b0, Mul(b1,i)), Mul(b2, Mul(i,i))).
    [[nodiscard]] static std::optional<Match>
    try_poly_keystream(const Z29Expr& expr, Z29Expr::Ptr owned, std::string_view cipher_var) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() != Kind::Add || !expr.left() || !expr.right()) {
            return std::nullopt;
        }
        if (!is_cipher_var(*expr.left(), cipher_var)) {
            return std::nullopt;
        }
        const Z29Expr& ks = *expr.right();
        // Unwrap Neg for minus forms.
        const Z29Expr* ksp = &ks;
        if (ks.kind() == Kind::Neg && ks.arg()) {
            ksp = ks.arg().get();
        }
        if (!ksp || ksp->kind() != Kind::Add || !ksp->left() || !ksp->right()) {
            return std::nullopt;
        }
        // Outer Add(linear_or_b0b1, Mul(b2, Mul(i,i)))
        auto is_i_squared_mul = [](const Z29Expr& e) -> bool {
            if (e.kind() != Kind::Mul || !e.left() || !e.right()) {
                return false;
            }
            // Mul(b2, Mul(i,i)) or Mul(Mul(i,i), b2)
            auto is_ii = [](const Z29Expr& m) {
                return m.kind() == Kind::Mul && m.left() && m.right() &&
                       m.left()->kind() == Kind::Var && m.left()->name() == "i" &&
                       m.right()->kind() == Kind::Var && m.right()->name() == "i";
            };
            if (is_ii(*e.right()) && e.left()->kind() == Kind::Var && e.left()->name() != "i") {
                return true;
            }
            if (is_ii(*e.left()) && e.right()->kind() == Kind::Var && e.right()->name() != "i") {
                return true;
            }
            return false;
        };
        if (is_i_squared_mul(*ksp->right()) && match_b0_b1_i(*ksp->left())) {
            return Match{ShapeId::PolyKeystream, "cipher ± (b0+b1*i+b2*i*i)", std::move(owned)};
        }
        if (is_i_squared_mul(*ksp->left()) && match_b0_b1_i(*ksp->right())) {
            return Match{ShapeId::PolyKeystream, "cipher ± (b0+b1*i+b2*i*i)", std::move(owned)};
        }
        return std::nullopt;
    }
};

#endif // THEORY_SHAPE_MATCH_HPP
