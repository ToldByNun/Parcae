#ifndef THEORY_HIST_CHI2_EMIT_HPP
#define THEORY_HIST_CHI2_EMIT_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_emit_cuda.hpp"
#include "parcae/dsl/dsl_launch_plan.hpp"
#include "parcae/dsl/dsl_optimize.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_shape_match.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// AOT fused-χ² hist emit for theory search (`docs/architecture/dsl-smart-hist.md`).
///
/// Classify via `Z29ExprNormalize` + `TheoryShapeMatch` (name-irrelevant), then:
/// ShapeInline (Atbash/Caesar/Affine) → S1 LUT twin until shape kernels land;
/// LinearKeystream → S2; FxOnly → S1; Autokey/prefer_branch → S0; else S3 stub→S0.
///
/// No C++ namespaces. Emitted CUDA uses file-scope `__global__` names —
/// never anonymous `namespace {}`.
class TheoryHistChi2Emit {
public:
    /// Caps aligned with `TheoryChi2Batch` / search fused export.
    static constexpr std::uint32_t kMaxProgramOps = 4096;
    static constexpr std::uint16_t kMaxDeviceStack = 64;
    static constexpr std::uint16_t kMaxSlots = 64;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    enum class Strategy : std::uint8_t {
        /// HotLoop bytecode interpreter (`TheoryChi2Batch`) — always available.
        S0Bytecode = 0,
        /// Decrypt is `f(x; params)` only (no stream index / lag) → LUT-29 hist.
        S1Lut29,
        /// `x ± g(i; params)` / bitmask_blend-shaped → uchar4 inline hist.
        S2Uchar4Inline,
        /// Inline scalar hist (no interpreter); not uchar4-packable.
        S3ScalarInline,
        /// Algebraic Atbash / Caesar / Affine (`TheoryShapeMatch`); runtime may
        /// use S1 twin until dedicated HistFast shape kernels ship.
        ShapeInline,
    };

    /// Persisted shape match (algebra only — not theory name / catalog API).
    class ShapePlan {
    public:
        explicit ShapePlan(TheoryShapeMatch::ShapeId shape, std::string reason)
            : shape_(shape), reason_(std::move(reason)) {}

        [[nodiscard]] static ShapePlan from_match(const TheoryShapeMatch::Match& m) {
            ShapePlan p{m.shape(), m.reason()};
            p.shift_name_ = m.shift_name();
            p.a_name_ = m.a_name();
            p.b_name_ = m.b_name();
            p.b0_name_ = m.b0_name();
            p.b1_name_ = m.b1_name();
            p.cipher_minus_ks_ = m.cipher_minus_ks();
            return p;
        }

        [[nodiscard]] TheoryShapeMatch::ShapeId shape() const noexcept { return shape_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] const std::string& shift_name() const noexcept { return shift_name_; }

        [[nodiscard]] const std::string& a_name() const noexcept { return a_name_; }

        [[nodiscard]] const std::string& b_name() const noexcept { return b_name_; }

        [[nodiscard]] const std::string& b0_name() const noexcept { return b0_name_; }

        [[nodiscard]] const std::string& b1_name() const noexcept { return b1_name_; }

        [[nodiscard]] bool cipher_minus_ks() const noexcept { return cipher_minus_ks_; }

        [[nodiscard]] bool is_shape_inline() const noexcept {
            return shape_ == TheoryShapeMatch::ShapeId::Atbash ||
                   shape_ == TheoryShapeMatch::ShapeId::Caesar ||
                   shape_ == TheoryShapeMatch::ShapeId::Affine;
        }

        /// In-lib `TheoryHistChi2Shape::launch_atbash_async` is available.
        [[nodiscard]] bool has_shape_atbash_kernel() const noexcept {
            return shape_ == TheoryShapeMatch::ShapeId::Atbash;
        }

    private:
        TheoryShapeMatch::ShapeId shape_ = TheoryShapeMatch::ShapeId::Unknown;
        std::string reason_;
        std::string shift_name_;
        std::string a_name_;
        std::string b_name_;
        std::string b0_name_;
        std::string b1_name_;
        bool cipher_minus_ks_ = false;
    };

    /// S1 plan: params in `TheoryIr::params()` order; LUT rows are `C × 29`.
    class S1LutPlan {
    public:
        explicit S1LutPlan(std::vector<std::string> param_names)
            : param_names_(std::move(param_names)) {}

        [[nodiscard]] const std::vector<std::string>& param_names() const noexcept {
            return param_names_;
        }

        [[nodiscard]] std::size_t param_count() const noexcept { return param_names_.size(); }

    private:
        std::vector<std::string> param_names_;
    };

    /// Linear keystream plan: `ks = b0 + b1·(t mod 29)` (param names from the expr).
    class S2LinearPlan {
    public:
        S2LinearPlan(std::string b0_name, std::string b1_name, bool cipher_minus_ks)
            : b0_name_(std::move(b0_name)), b1_name_(std::move(b1_name)),
              cipher_minus_ks_(cipher_minus_ks) {}

        [[nodiscard]] const std::string& b0_name() const noexcept { return b0_name_; }

        [[nodiscard]] const std::string& b1_name() const noexcept { return b1_name_; }

        /// `true` → `HistFast::dec_sub(x, ks)`; `false` → `HistFast::enc_caesar(x, ks)`.
        [[nodiscard]] bool cipher_minus_ks() const noexcept { return cipher_minus_ks_; }

    private:
        std::string b0_name_;
        std::string b1_name_;
        bool cipher_minus_ks_ = true;
    };

    class Selection {
    public:
        Selection(Strategy strategy, std::string reason,
                  std::optional<ShapePlan> shape = std::nullopt)
            : strategy_(strategy), reason_(std::move(reason)), shape_(std::move(shape)) {}

        [[nodiscard]] Strategy strategy() const noexcept { return strategy_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] const std::optional<ShapePlan>& shape() const noexcept { return shape_; }

        [[nodiscard]] bool is_specialized() const noexcept {
            return strategy_ != Strategy::S0Bytecode;
        }

    private:
        Strategy strategy_ = Strategy::S0Bytecode;
        std::string reason_;
        std::optional<ShapePlan> shape_;
    };

    /// Emit / artifact wiring result. `specialized()==false` → use bytecode launch.
    class EmitBundle {
    public:
        EmitBundle(Strategy intended, Strategy emitted, bool specialized, std::string reason,
                   std::string header_text, std::string cu_text, std::string kernel_symbol,
                   std::optional<S1LutPlan> s1_lut = std::nullopt,
                   std::optional<S2LinearPlan> s2_linear = std::nullopt,
                   std::optional<ShapePlan> shape = std::nullopt)
            : intended_(intended), emitted_(emitted), specialized_(specialized),
              reason_(std::move(reason)), header_text_(std::move(header_text)),
              cu_text_(std::move(cu_text)), kernel_symbol_(std::move(kernel_symbol)),
              s1_lut_(std::move(s1_lut)), s2_linear_(std::move(s2_linear)),
              shape_(std::move(shape)) {}

        [[nodiscard]] Strategy intended_strategy() const noexcept { return intended_; }

        [[nodiscard]] Strategy emitted_strategy() const noexcept { return emitted_; }

        [[nodiscard]] bool specialized() const noexcept { return specialized_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] const std::string& header_text() const noexcept { return header_text_; }

        [[nodiscard]] const std::string& cu_text() const noexcept { return cu_text_; }

        [[nodiscard]] const std::string& kernel_symbol() const noexcept { return kernel_symbol_; }

        [[nodiscard]] const std::optional<S1LutPlan>& s1_lut() const noexcept { return s1_lut_; }

        [[nodiscard]] const std::optional<S2LinearPlan>& s2_linear() const noexcept {
            return s2_linear_;
        }

        [[nodiscard]] const std::optional<ShapePlan>& shape() const noexcept { return shape_; }

        [[nodiscard]] bool is_shape_inline() const noexcept {
            return emitted_ == Strategy::ShapeInline;
        }

        /// Soft runtime path until dedicated shape HistFast kernels ship.
        [[nodiscard]] bool has_s1_soft_path() const noexcept {
            return s1_lut_.has_value() &&
                   (emitted_ == Strategy::S1Lut29 || emitted_ == Strategy::ShapeInline) &&
                   !has_shape_atbash_kernel();
        }

        [[nodiscard]] bool has_shape_atbash_kernel() const noexcept {
            return emitted_ == Strategy::ShapeInline && shape_.has_value() &&
                   shape_->has_shape_atbash_kernel();
        }

    private:
        Strategy intended_ = Strategy::S0Bytecode;
        Strategy emitted_ = Strategy::S0Bytecode;
        bool specialized_ = false;
        std::string reason_;
        std::string header_text_;
        std::string cu_text_;
        std::string kernel_symbol_;
        std::optional<S1LutPlan> s1_lut_;
        std::optional<S2LinearPlan> s2_linear_;
        std::optional<ShapePlan> shape_;
    };

    [[nodiscard]] static const char* strategy_str(Strategy s) noexcept {
        switch (s) {
        case Strategy::S0Bytecode:
            return "S0_bytecode";
        case Strategy::S1Lut29:
            return "S1_lut29";
        case Strategy::S2Uchar4Inline:
            return "S2_uchar4_inline";
        case Strategy::S3ScalarInline:
            return "S3_scalar_inline";
        case Strategy::ShapeInline:
            return "ShapeInline";
        }
        return "S0_bytecode";
    }

    [[nodiscard]] static const char* shape_id_str(TheoryShapeMatch::ShapeId id) noexcept {
        return TheoryShapeMatch::shape_str(id);
    }

    /// Classify decrypt HotLoop: normalize+match before generic S1/S2 heuristics.
    [[nodiscard]] static Selection select_strategy(const TheoryIr& theory,
                                                   std::string_view cipher_var = "x") {
        if (!theory.decrypt_step()) {
            return Selection{Strategy::S0Bytecode, "missing decrypt_step"};
        }
        if (cipher_var.empty()) {
            return Selection{Strategy::S0Bytecode, "empty cipher_var"};
        }

        StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt, cipher_var);
        if (!prog.ok()) {
            return Selection{Strategy::S0Bytecode,
                             std::string("bytecode compile failed: ") + prog.status().message()};
        }
        if (prog.value().ops.size() > kMaxProgramOps) {
            return Selection{Strategy::S0Bytecode, "op_count exceeds kMaxProgramOps"};
        }
        if (prog.value().max_stack > kMaxDeviceStack) {
            return Selection{Strategy::S0Bytecode, "max_stack exceeds kMaxDeviceStack"};
        }
        if (prog.value().slot_names.size() > kMaxSlots) {
            return Selection{Strategy::S0Bytecode, "slot_count exceeds kMaxSlots"};
        }

        const Z29Expr& dec = *theory.decrypt_step();
        if (expr_has_autokey(dec)) {
            return Selection{Strategy::S0Bytecode, "autokey_shift requires S0 (or ring emit later)"};
        }
        if (expr_has_prefer_branch_select(dec)) {
            return Selection{Strategy::S0Bytecode, "prefer_branch Select is S0-only"};
        }

        const bool uses_cipher = DslOptimize::depends_on_var(dec, cipher_var);
        if (!uses_cipher) {
            return Selection{Strategy::S0Bytecode, "decrypt_step does not reference cipher_var"};
        }

        StatusOr<TheoryShapeMatch::Match> matched =
            TheoryShapeMatch::match_theory(theory, cipher_var);
        if (matched.ok()) {
            const ShapePlan plan = ShapePlan::from_match(matched.value());
            switch (matched.value().shape()) {
            case TheoryShapeMatch::ShapeId::Atbash:
                return Selection{Strategy::ShapeInline,
                                 "shape Atbash — ShapeInline (HistFast::dec_atbash twin)", plan};
            case TheoryShapeMatch::ShapeId::Caesar:
            case TheoryShapeMatch::ShapeId::Affine:
                return Selection{Strategy::ShapeInline,
                                 std::string("shape ") + shape_id_str(matched.value().shape()) +
                                     " — ShapeInline (S1 soft twin until shape kernels)",
                                 plan};
            case TheoryShapeMatch::ShapeId::LinearKeystream:
                return Selection{Strategy::S2Uchar4Inline,
                                 std::string("shape LinearKeystream — S2 uchar4 candidate (") +
                                     matched.value().reason() + ")",
                                 plan};
            case TheoryShapeMatch::ShapeId::FxOnly:
                return Selection{Strategy::S1Lut29,
                                 std::string("shape FxOnly — S1 LUT candidate (") +
                                     matched.value().reason() + ")",
                                 plan};
            case TheoryShapeMatch::ShapeId::PolyKeystream:
                return Selection{Strategy::S3ScalarInline,
                                 std::string("shape PolyKeystream — S3/S5 stub (") +
                                     matched.value().reason() + ")",
                                 plan};
            case TheoryShapeMatch::ShapeId::KeyedGeneral:
                return Selection{Strategy::S3ScalarInline,
                                 std::string("shape KeyedGeneral — S3 scalar candidate (") +
                                     matched.value().reason() + ")",
                                 plan};
            case TheoryShapeMatch::ShapeId::Autokey:
                return Selection{Strategy::S0Bytecode,
                                 std::string("shape Autokey — S0 until S4 (") +
                                     matched.value().reason() + ")",
                                 plan};
            case TheoryShapeMatch::ShapeId::Unknown:
                break;
            }
        }

        // Legacy soft heuristics when shape is Unknown / match failed.
        const bool uses_i = DslOptimize::depends_on_var(dec, "i");
        if (!uses_i) {
            return Selection{Strategy::S1Lut29,
                             "decrypt is f(x;params) without stream index — S1 LUT candidate"};
        }
        if (looks_like_keystream_add_sub(dec, cipher_var)) {
            return Selection{Strategy::S2Uchar4Inline,
                             "decrypt looks like x ± g(i;params) — S2 uchar4 candidate"};
        }
        return Selection{Strategy::S3ScalarInline,
                         "decrypt uses i but is not simple ± keystream — S3 scalar candidate"};
    }

    /// Match progressive / bitmask_blend-linear decrypt: `x ± (b0 + b1·i)`.
    [[nodiscard]] static std::optional<S2LinearPlan>
    match_s2_linear(const TheoryIr& theory, std::string_view cipher_var = "x") {
        if (!theory.decrypt_step()) {
            return std::nullopt;
        }
        const Z29Expr& dec = *theory.decrypt_step();
        using Kind = Z29Expr::Kind;
        if (dec.kind() != Kind::Add && dec.kind() != Kind::Sub) {
            return std::nullopt;
        }
        const Z29Expr& l = *dec.left();
        const Z29Expr& r = *dec.right();
        const bool l_cipher = l.kind() == Kind::Var && l.name() == cipher_var;
        const bool r_cipher = r.kind() == Kind::Var && r.name() == cipher_var;
        if (l_cipher == r_cipher) {
            return std::nullopt;
        }
        if (l_cipher && DslOptimize::depends_on_var(r, cipher_var)) {
            return std::nullopt;
        }
        if (r_cipher && DslOptimize::depends_on_var(l, cipher_var)) {
            return std::nullopt;
        }

        const Z29Expr& ks = l_cipher ? r : l;
        std::optional<std::pair<std::string, std::string>> names = match_linear_b0_b1_i(ks);
        if (!names) {
            return std::nullopt;
        }

        // Decrypt root Sub(x, ks) or Add(x, ks) only (cipher on the left).
        // Add(ks, x) is commutative → treat as enc_caesar; Sub(ks, x) is not linear S2.
        if (r_cipher && dec.kind() == Kind::Sub) {
            return std::nullopt;
        }
        const bool cipher_minus_ks = (dec.kind() == Kind::Sub) && l_cipher;
        const bool cipher_plus_ks =
            (dec.kind() == Kind::Add) && (l_cipher || r_cipher);
        if (!cipher_minus_ks && !cipher_plus_ks) {
            return std::nullopt;
        }
        return S2LinearPlan{names->first, names->second, cipher_minus_ks};
    }

    /// Emit fused-hist sources for decrypt search path.
    /// ShapeInline → S1 LUT soft twin (+ ShapePlan); S1 → LUT-29; S2 linear → uchar4;
    /// else soft S0 fallback.
    [[nodiscard]] static StatusOr<EmitBundle>
    emit_decrypt_hist(const TheoryIr& theory, std::string_view cipher_var = "x",
                      const std::vector<DslOptimize::Hoist>& decrypt_hoists = {}) {
        if (theory.name().empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistChi2Emit: theory name must be non-empty")
                .to_status();
        }
        Selection sel = select_strategy(theory, cipher_var);
        if (sel.strategy() == Strategy::S0Bytecode) {
            return EmitBundle{Strategy::S0Bytecode, Strategy::S0Bytecode, false, sel.reason(), "",
                              "", "", std::nullopt, std::nullopt, sel.shape()};
        }

        if (sel.strategy() == Strategy::ShapeInline) {
            if (!decrypt_hoists.empty()) {
                return EmitBundle{Strategy::ShapeInline, Strategy::S0Bytecode, false,
                                  std::string("ShapeInline classified but decrypt hoists not yet "
                                              "wired; fallback S0 (") +
                                      sel.reason() + ")",
                                  "", "", "", std::nullopt, std::nullopt, sel.shape()};
            }
            if (sel.shape() && sel.shape()->has_shape_atbash_kernel()) {
                // In-lib twin — sources discarded; export launches TheoryHistChi2Shape.
                return EmitBundle{Strategy::ShapeInline, Strategy::ShapeInline, true,
                                  std::string("ShapeInline Atbash hist twin: ") + sel.reason(), "",
                                  "", "theory_hist_chi2_shape_atbash_kernel", std::nullopt,
                                  std::nullopt, sel.shape()};
            }
            StatusOr<EmitBundle> soft = emit_s1_lut_sources(theory, cipher_var, sel.reason());
            if (!soft.ok()) {
                return EmitBundle{Strategy::ShapeInline, Strategy::S0Bytecode, false,
                                  std::string("ShapeInline S1 soft twin failed: ") +
                                      soft.status().message() + "; fallback S0 (" + sel.reason() +
                                      ")",
                                  "", "", "", std::nullopt, std::nullopt, sel.shape()};
            }
            EmitBundle& b = soft.value();
            return EmitBundle{Strategy::ShapeInline, Strategy::ShapeInline, true,
                              std::string("ShapeInline + S1 soft twin: ") + sel.reason(),
                              b.header_text(), b.cu_text(), b.kernel_symbol(), b.s1_lut(),
                              std::nullopt, sel.shape()};
        }

        if (sel.strategy() == Strategy::S1Lut29) {
            if (!decrypt_hoists.empty()) {
                return EmitBundle{Strategy::S1Lut29, Strategy::S0Bytecode, false,
                                  std::string("S1 classified but decrypt hoists not yet wired; "
                                              "fallback S0 (") +
                                      sel.reason() + ")",
                                  "", "", "", std::nullopt, std::nullopt, sel.shape()};
            }
            StatusOr<EmitBundle> bundle = emit_s1_lut_sources(theory, cipher_var, sel.reason());
            if (!bundle.ok()) {
                return EmitBundle{Strategy::S1Lut29, Strategy::S0Bytecode, false,
                                  std::string("S1 emit failed: ") + bundle.status().message() +
                                      "; fallback S0 (" + sel.reason() + ")",
                                  "", "", "", std::nullopt, std::nullopt, sel.shape()};
            }
            return EmitBundle{bundle.value().intended_strategy(),
                              bundle.value().emitted_strategy(), bundle.value().specialized(),
                              bundle.value().reason(), bundle.value().header_text(),
                              bundle.value().cu_text(), bundle.value().kernel_symbol(),
                              bundle.value().s1_lut(), bundle.value().s2_linear(), sel.shape()};
        }

        if (sel.strategy() == Strategy::S2Uchar4Inline) {
            std::optional<S2LinearPlan> plan = match_s2_linear(theory, cipher_var);
            if (!plan && sel.shape() && !sel.shape()->b0_name().empty() &&
                !sel.shape()->b1_name().empty()) {
                plan = S2LinearPlan{sel.shape()->b0_name(), sel.shape()->b1_name(),
                                    sel.shape()->cipher_minus_ks()};
            }
            if (plan) {
                StatusOr<EmitBundle> bundle = emit_s2_linear_sources(theory, *plan, sel.reason());
                if (!bundle.ok()) {
                    return bundle.status();
                }
                return EmitBundle{bundle.value().intended_strategy(),
                                  bundle.value().emitted_strategy(), bundle.value().specialized(),
                                  bundle.value().reason(), bundle.value().header_text(),
                                  bundle.value().cu_text(), bundle.value().kernel_symbol(),
                                  bundle.value().s1_lut(), bundle.value().s2_linear(),
                                  sel.shape()};
            }
            return EmitBundle{Strategy::S2Uchar4Inline, Strategy::S0Bytecode, false,
                              std::string("S2 classified but keystream is not linear b0+b1*i; "
                                          "fallback S0 (") +
                                  sel.reason() + ")",
                              "", "", "", std::nullopt, std::nullopt, sel.shape()};
        }

        // S3 emit not implemented yet — soft S0 fallback (export stays correct).
        return EmitBundle{sel.strategy(), Strategy::S0Bytecode, false,
                          std::string("skeleton: ") + strategy_str(sel.strategy()) +
                              " classified but emit not implemented; fallback S0 (" + sel.reason() +
                              ")",
                          "", "", "", std::nullopt, std::nullopt, sel.shape()};
    }

    /// Launch geometry for specialized hist (same as FamilyChi2 / HistFast).
    [[nodiscard]] static StatusOr<DslLaunchPlan::Plan>
    hist_launch_plan(std::size_t candidates, std::size_t tokens) {
        return DslLaunchPlan::hist_chi2_2d(candidates, tokens);
    }

private:
    TheoryHistChi2Emit() = delete;

    [[nodiscard]] static bool expr_has_autokey(const Z29Expr& expr) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() == Kind::Call && expr.name() == "z29_autokey_shift") {
            return true;
        }
        if (expr.kind() == Kind::Call) {
            for (const Z29Expr::Ptr& a : expr.args()) {
                if (a && expr_has_autokey(*a)) {
                    return true;
                }
            }
            return false;
        }
        if (expr.kind() == Kind::Select) {
            return expr_has_autokey(*expr.cond()) || expr_has_autokey(*expr.if_true()) ||
                   expr_has_autokey(*expr.if_false());
        }
        if (Z29Expr::is_binary(expr.kind())) {
            return expr_has_autokey(*expr.left()) || expr_has_autokey(*expr.right());
        }
        if (Z29Expr::is_unary(expr.kind())) {
            return expr_has_autokey(*expr.arg());
        }
        return false;
    }

    [[nodiscard]] static bool expr_has_prefer_branch_select(const Z29Expr& expr) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() == Kind::Select && expr.prefer_branch()) {
            return true;
        }
        if (expr.kind() == Kind::Call) {
            for (const Z29Expr::Ptr& a : expr.args()) {
                if (a && expr_has_prefer_branch_select(*a)) {
                    return true;
                }
            }
            return false;
        }
        if (expr.kind() == Kind::Select) {
            return expr_has_prefer_branch_select(*expr.cond()) ||
                   expr_has_prefer_branch_select(*expr.if_true()) ||
                   expr_has_prefer_branch_select(*expr.if_false());
        }
        if (Z29Expr::is_binary(expr.kind())) {
            return expr_has_prefer_branch_select(*expr.left()) ||
                   expr_has_prefer_branch_select(*expr.right());
        }
        if (Z29Expr::is_unary(expr.kind())) {
            return expr_has_prefer_branch_select(*expr.arg());
        }
        return false;
    }

    /// True when decrypt is `cipher ± keystream` (or keystream ± cipher) at the root.
    [[nodiscard]] static bool looks_like_keystream_add_sub(const Z29Expr& expr,
                                                           std::string_view cipher_var) {
        using Kind = Z29Expr::Kind;
        if (expr.kind() != Kind::Add && expr.kind() != Kind::Sub) {
            return false;
        }
        const Z29Expr& l = *expr.left();
        const Z29Expr& r = *expr.right();
        const bool l_cipher = l.kind() == Kind::Var && l.name() == cipher_var;
        const bool r_cipher = r.kind() == Kind::Var && r.name() == cipher_var;
        if (l_cipher && !DslOptimize::depends_on_var(r, cipher_var)) {
            return DslOptimize::depends_on_var(r, "i");
        }
        if (r_cipher && !DslOptimize::depends_on_var(l, cipher_var)) {
            return DslOptimize::depends_on_var(l, "i");
        }
        return false;
    }

    /// Match `b0 + b1*i` / `b1*i + b0` (Mul operands either order). Returns `{b0,b1}`.
    [[nodiscard]] static std::optional<std::pair<std::string, std::string>>
    match_linear_b0_b1_i(const Z29Expr& ks) {
        using Kind = Z29Expr::Kind;
        if (ks.kind() != Kind::Add) {
            return std::nullopt;
        }
        const Z29Expr& a = *ks.left();
        const Z29Expr& b = *ks.right();

        auto match_mul_b1_i = [](const Z29Expr& e) -> std::optional<std::string> {
            if (e.kind() != Kind::Mul) {
                return std::nullopt;
            }
            const Z29Expr& ml = *e.left();
            const Z29Expr& mr = *e.right();
            const bool l_i = ml.kind() == Kind::Var && ml.name() == "i";
            const bool r_i = mr.kind() == Kind::Var && mr.name() == "i";
            if (l_i && mr.kind() == Kind::Var && mr.name() != "i") {
                return mr.name();
            }
            if (r_i && ml.kind() == Kind::Var && ml.name() != "i") {
                return ml.name();
            }
            return std::nullopt;
        };

        if (a.kind() == Kind::Var) {
            std::optional<std::string> b1 = match_mul_b1_i(b);
            if (b1 && *b1 != a.name()) {
                return std::make_pair(a.name(), *b1);
            }
        }
        if (b.kind() == Kind::Var) {
            std::optional<std::string> b1 = match_mul_b1_i(a);
            if (b1 && *b1 != b.name()) {
                return std::make_pair(b.name(), *b1);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] static StatusOr<EmitBundle>
    emit_s2_linear_sources(const TheoryIr& theory, const S2LinearPlan& plan,
                           const std::string& select_reason) {
        const std::string& id = theory.name();
        const std::string kern = id + "_s2_hist_kernel";
        const std::string guard = "PARCAE_EMIT_" + id + "_S2_HIST_HPP";

        std::ostringstream hdr;
        hdr << "// Generated by TheoryHistChi2Emit (S2 linear uchar4) — do not hand-edit.\n";
        hdr << "#ifndef " << guard << "\n";
        hdr << "#define " << guard << "\n\n";
        hdr << "#include \"parcae/core/status.hpp\"\n\n";
        hdr << "#include <cstddef>\n";
        hdr << "#include <cstdint>\n\n";
        hdr << "/// S2 uchar4 fused χ² hist for `" << id << "` "
            << "(ks = " << plan.b0_name() << " + " << plan.b1_name() << "·i).\n";
        hdr << "/// Runtime twin: TheoryHistChi2S2::launch_linear_async "
            << "(cipher_minus_ks=" << (plan.cipher_minus_ks() ? "true" : "false") << ").\n";
        hdr << "class " << to_pascal(id) << "S2Hist {\n";
        hdr << "public:\n";
        hdr << "    static constexpr std::size_t alphabet_size = 29;\n";
        hdr << "    [[nodiscard]] static Status launch_linear_async(\n";
        hdr << "        const std::uint8_t* device_in, const std::uint8_t* device_"
            << plan.b0_name() << ",\n";
        hdr << "        const std::uint8_t* device_" << plan.b1_name()
            << ", const double* device_probabilities,\n";
        hdr << "        std::uint32_t* device_counts, double* device_scores, "
               "std::size_t candidate_count,\n";
        hdr << "        std::size_t token_count);\n";
        hdr << "private:\n";
        hdr << "    " << to_pascal(id) << "S2Hist() = delete;\n";
        hdr << "};\n\n";
        hdr << "#endif // " << guard << "\n";

        std::ostringstream cu;
        cu << "// Generated by TheoryHistChi2Emit (S2 linear uchar4) — do not hand-edit.\n";
        cu << "// File-scope kernel (no anonymous namespace). Prefer linking "
              "TheoryHistChi2S2 for search.\n";
        cu << "#include \"" << to_pascal(id) << "S2Hist.hpp\"\n\n";
        cu << "#include \"chi2_batch_score.hpp\"\n";
        cu << "#include \"cuda_error.hpp\"\n";
        cu << "#include \"hist_fast.hpp\"\n";
        cu << "#include \"z29_device.hpp\"\n\n";
        cu << "#include <cuda_runtime_api.h>\n\n";
        cu << "__global__ void " << kern << "(const std::uint8_t* in, const std::uint8_t* b0,\n";
        cu << "    const std::uint8_t* b1, std::uint32_t* counts, std::size_t token_count,\n";
        cu << "    std::uint8_t cipher_minus_ks) {\n";
        cu << "    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];\n";
        cu << "    HistFast::clear_private(priv);\n";
        cu << "    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);\n";
        cu << "    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);\n";
        cu << "    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);\n";
        cu << "    const std::uint8_t pb0 = b0[candidate];\n";
        cu << "    const std::uint8_t pb1 = b1[candidate];\n";
        cu << "    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;\n";
        cu << "    const std::size_t n4 = token_count / 4u;\n";
        cu << "    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);\n";
        cu << "    auto ks_at = [&](std::size_t t) -> std::uint8_t {\n";
        cu << "        const std::uint8_t im = static_cast<std::uint8_t>(\n";
        cu << "            t % static_cast<std::size_t>(Z29Device::modulus));\n";
        cu << "        return Z29Device::add(pb0, Z29Device::mul(pb1, im));\n";
        cu << "    };\n";
        cu << "    auto out_byte = [&](std::uint8_t x, std::uint8_t ks) -> std::uint8_t {\n";
        cu << "        return cipher_minus_ks != 0u ? HistFast::dec_sub(x, ks)\n";
        cu << "                                    : HistFast::enc_caesar(x, ks);\n";
        cu << "    };\n";
        cu << "    for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +\n";
        cu << "                         static_cast<std::size_t>(threadIdx.x);\n";
        cu << "         i < n4; i += stride) {\n";
        cu << "        const uchar4 v = in4[i];\n";
        cu << "        const std::size_t t0 = i * 4u;\n";
        cu << "        HistFast::add_private(priv, out_byte(v.x, ks_at(t0)));\n";
        cu << "        HistFast::add_private(priv, out_byte(v.y, ks_at(t0 + 1u)));\n";
        cu << "        HistFast::add_private(priv, out_byte(v.z, ks_at(t0 + 2u)));\n";
        cu << "        HistFast::add_private(priv, out_byte(v.w, ks_at(t0 + 3u)));\n";
        cu << "    }\n";
        cu << "    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +\n";
        cu << "                         static_cast<std::size_t>(threadIdx.x);\n";
        cu << "         t < token_count; t += stride) {\n";
        cu << "        HistFast::add_private(priv, out_byte(in[t], ks_at(t)));\n";
        cu << "    }\n";
        cu << "    HistFast::flush_private(\n";
        cu << "        priv, counts + candidate * static_cast<std::size_t>(HistFast::alphabet));\n";
        cu << "}\n\n";
        cu << "Status " << to_pascal(id) << "S2Hist::launch_linear_async(\n";
        cu << "    const std::uint8_t* device_in, const std::uint8_t* device_b0,\n";
        cu << "    const std::uint8_t* device_b1, const double* device_probabilities,\n";
        cu << "    std::uint32_t* device_counts, double* device_scores, "
              "std::size_t candidate_count,\n";
        cu << "    std::size_t token_count) {\n";
        cu << "    if (device_in == nullptr || device_b0 == nullptr || device_b1 == nullptr ||\n";
        cu << "        device_probabilities == nullptr || device_counts == nullptr ||\n";
        cu << "        device_scores == nullptr) {\n";
        cu << "        return Status::error(\"" << to_pascal(id) << "S2Hist: null\");\n";
        cu << "    }\n";
        cu << "    const std::size_t hist_bytes =\n";
        cu << "        candidate_count * alphabet_size * sizeof(std::uint32_t);\n";
        cu << "    Status cleared = CudaError::to_status(\n";
        cu << "        cudaMemsetAsync(device_counts, 0, hist_bytes, 0), \"" << to_pascal(id)
           << "S2Hist clear\");\n";
        cu << "    if (!cleared.ok()) {\n";
        cu << "        return cleared;\n";
        cu << "    }\n";
        cu << "    const dim3 grid(static_cast<unsigned>(candidate_count),\n";
        cu << "                    static_cast<unsigned>(HistFast::tiles_for(token_count)));\n";
        cu << "    " << kern << "<<<grid, HistFast::threads>>>(\n";
        cu << "        device_in, device_b0, device_b1, device_counts, token_count,\n";
        cu << "        " << (plan.cipher_minus_ks() ? "1u" : "0u") << ");\n";
        cu << "    Status hist = CudaError::to_status(cudaGetLastError(), \"" << to_pascal(id)
           << "S2Hist hist\");\n";
        cu << "    if (!hist.ok()) {\n";
        cu << "        return hist;\n";
        cu << "    }\n";
        cu << "    return Chi2BatchScore::finalize_async(device_counts, device_probabilities,\n";
        cu << "                                          device_scores, candidate_count, "
              "token_count);\n";
        cu << "}\n";

        return EmitBundle{Strategy::S2Uchar4Inline, Strategy::S2Uchar4Inline, true,
                          std::string("S2 linear uchar4 emit: ") + select_reason, hdr.str(),
                          cu.str(), kern, std::nullopt, plan};
    }

    [[nodiscard]] static StatusOr<EmitBundle>
    emit_s1_lut_sources(const TheoryIr& theory, std::string_view cipher_var,
                        const std::string& select_reason) {
        StatusOr<std::string> f_cpp =
            DslEmitCuda::emit_expr(theory.decrypt_step(), cipher_var, "x");
        if (!f_cpp.ok()) {
            return f_cpp.status();
        }

        std::vector<std::string> param_names;
        param_names.reserve(theory.params().size());
        for (const ParamIr& p : theory.params()) {
            param_names.push_back(p.name());
        }
        const S1LutPlan plan{param_names};
        const std::size_t P = plan.param_count();
        const std::string& id = theory.name();
        const std::string kern = id + "_s1_hist_kernel";
        const std::string guard = "PARCAE_EMIT_" + id + "_S1_HIST_HPP";
        const std::string cls = to_pascal(id) + "S1Hist";

        std::ostringstream hdr;
        hdr << "// Generated by TheoryHistChi2Emit (S1 LUT-29) — do not hand-edit.\n";
        hdr << "#ifndef " << guard << "\n";
        hdr << "#define " << guard << "\n\n";
        hdr << "#include \"parcae/core/status.hpp\"\n\n";
        hdr << "#include <cstddef>\n";
        hdr << "#include <cstdint>\n\n";
        hdr << "/// S1 LUT-29 fused χ² hist for `" << id << "` (f(x)-only).\n";
        hdr << "/// Runtime twin: TheoryHistChi2S1::launch_lut_async (host-filled C×29 LUTs).\n";
        hdr << "class " << cls << " {\n";
        hdr << "public:\n";
        hdr << "    static constexpr std::size_t alphabet_size = 29;\n";
        hdr << "    static constexpr std::size_t param_count = " << P << ";\n";
        hdr << "    /// `device_params` is row-major C×" << P
            << " (may be null when param_count==0).\n";
        hdr << "    [[nodiscard]] static Status launch_async(\n";
        hdr << "        const std::uint8_t* device_in, const std::uint8_t* device_params,\n";
        hdr << "        const double* device_probabilities, std::uint32_t* device_counts,\n";
        hdr << "        double* device_scores, std::size_t candidate_count, "
               "std::size_t token_count);\n";
        hdr << "private:\n";
        hdr << "    " << cls << "() = delete;\n";
        hdr << "};\n\n";
        hdr << "#endif // " << guard << "\n";

        std::ostringstream cu;
        cu << "// Generated by TheoryHistChi2Emit (S1 LUT-29) — do not hand-edit.\n";
        cu << "// File-scope kernel (no anonymous namespace). Prefer linking "
              "TheoryHistChi2S1 for search.\n";
        cu << "#include \"" << cls << ".hpp\"\n\n";
        cu << "#include \"chi2_batch_score.hpp\"\n";
        cu << "#include \"cuda_error.hpp\"\n";
        cu << "#include \"hist_fast.hpp\"\n";
        cu << "#include \"z29_device.hpp\"\n\n";
        cu << "#include <cuda_runtime_api.h>\n\n";
        cu << "__global__ void " << kern << "(const std::uint8_t* in, const std::uint8_t* params,\n";
        cu << "    std::uint32_t* counts, std::size_t token_count) {\n";
        cu << "    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];\n";
        cu << "    __shared__ std::uint8_t lut[HistFast::alphabet];\n";
        cu << "    HistFast::clear_private(priv);\n";
        cu << "    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);\n";
        cu << "    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);\n";
        cu << "    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);\n";
        if (P > 0) {
            cu << "    const std::uint8_t* prow =\n";
            cu << "        params + candidate * static_cast<std::size_t>(" << cls
               << "::param_count);\n";
            for (std::size_t i = 0; i < P; ++i) {
                cu << "    const std::uint8_t " << param_names[i] << " = prow[" << i << "];\n";
            }
        }
        cu << "    if (threadIdx.x < HistFast::alphabet) {\n";
        cu << "        const std::uint8_t x = static_cast<std::uint8_t>(threadIdx.x);\n";
        cu << "        lut[threadIdx.x] = " << f_cpp.value() << ";\n";
        cu << "    }\n";
        cu << "    __syncthreads();\n";
        cu << "    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;\n";
        cu << "    const std::size_t n4 = token_count / 4u;\n";
        cu << "    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);\n";
        cu << "    for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +\n";
        cu << "                         static_cast<std::size_t>(threadIdx.x);\n";
        cu << "         i < n4; i += stride) {\n";
        cu << "        const uchar4 v = in4[i];\n";
        cu << "        HistFast::add_private(priv, lut[v.x]);\n";
        cu << "        HistFast::add_private(priv, lut[v.y]);\n";
        cu << "        HistFast::add_private(priv, lut[v.z]);\n";
        cu << "        HistFast::add_private(priv, lut[v.w]);\n";
        cu << "    }\n";
        cu << "    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +\n";
        cu << "                         static_cast<std::size_t>(threadIdx.x);\n";
        cu << "         t < token_count; t += stride) {\n";
        cu << "        HistFast::add_private(priv, lut[in[t]]);\n";
        cu << "    }\n";
        cu << "    HistFast::flush_private(\n";
        cu << "        priv, counts + candidate * static_cast<std::size_t>(HistFast::alphabet));\n";
        cu << "}\n\n";
        cu << "Status " << cls << "::launch_async(\n";
        cu << "    const std::uint8_t* device_in, const std::uint8_t* device_params,\n";
        cu << "    const double* device_probabilities, std::uint32_t* device_counts,\n";
        cu << "    double* device_scores, std::size_t candidate_count, "
              "std::size_t token_count) {\n";
        cu << "    if (device_in == nullptr || device_probabilities == nullptr ||\n";
        cu << "        device_counts == nullptr || device_scores == nullptr) {\n";
        cu << "        return Status::error(\"" << cls << ": null\");\n";
        cu << "    }\n";
        if (P > 0) {
            cu << "    if (device_params == nullptr) {\n";
            cu << "        return Status::error(\"" << cls << ": null params\");\n";
            cu << "    }\n";
        } else {
            cu << "    (void)device_params;\n";
        }
        cu << "    const std::size_t hist_bytes =\n";
        cu << "        candidate_count * alphabet_size * sizeof(std::uint32_t);\n";
        cu << "    Status cleared = CudaError::to_status(\n";
        cu << "        cudaMemsetAsync(device_counts, 0, hist_bytes, 0), \"" << cls
           << " clear\");\n";
        cu << "    if (!cleared.ok()) {\n";
        cu << "        return cleared;\n";
        cu << "    }\n";
        cu << "    const dim3 grid(static_cast<unsigned>(candidate_count),\n";
        cu << "                    static_cast<unsigned>(HistFast::tiles_for(token_count)));\n";
        cu << "    " << kern << "<<<grid, HistFast::threads>>>(\n";
        cu << "        device_in, device_params, device_counts, token_count);\n";
        cu << "    Status hist = CudaError::to_status(cudaGetLastError(), \"" << cls
           << " hist\");\n";
        cu << "    if (!hist.ok()) {\n";
        cu << "        return hist;\n";
        cu << "    }\n";
        cu << "    return Chi2BatchScore::finalize_async(device_counts, device_probabilities,\n";
        cu << "                                          device_scores, candidate_count, "
              "token_count);\n";
        cu << "}\n";

        return EmitBundle{Strategy::S1Lut29, Strategy::S1Lut29, true,
                          std::string("S1 LUT-29 emit: ") + select_reason, hdr.str(), cu.str(),
                          kern, plan, std::nullopt};
    }

    [[nodiscard]] static std::string to_pascal(std::string_view snake) {
        std::string out;
        bool upper = true;
        for (char ch : snake) {
            if (ch == '_') {
                upper = true;
                continue;
            }
            if (upper) {
                out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
                upper = false;
            } else {
                out.push_back(ch);
            }
        }
        return out;
    }
};

#endif // THEORY_HIST_CHI2_EMIT_HPP
