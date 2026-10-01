#ifndef THEORY_HIST_CHI2_EMIT_HPP
#define THEORY_HIST_CHI2_EMIT_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_launch_plan.hpp"
#include "parcae/dsl/dsl_optimize.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// AOT fused-χ² hist emit for theory search (docs: theory CUDA throughput climb).
///
/// Skeleton (this commit): strategy selection + caps + S0 bytecode fallback.
/// S1/S2/S3 source emit arrives in follow-up commits; `emit_decrypt_hist` always
/// returns an S0 bundle until specialized emitters land.
///
/// No C++ namespaces. Emitted CUDA (later) uses file-scope `__global__` names —
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
    };

    class Selection {
    public:
        Selection(Strategy strategy, std::string reason)
            : strategy_(strategy), reason_(std::move(reason)) {}

        [[nodiscard]] Strategy strategy() const noexcept { return strategy_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] bool is_specialized() const noexcept {
            return strategy_ != Strategy::S0Bytecode;
        }

    private:
        Strategy strategy_ = Strategy::S0Bytecode;
        std::string reason_;
    };

    /// Emit / artifact wiring result. `specialized()==false` → use bytecode launch.
    class EmitBundle {
    public:
        EmitBundle(Strategy intended, Strategy emitted, bool specialized, std::string reason,
                   std::string header_text, std::string cu_text, std::string kernel_symbol)
            : intended_(intended), emitted_(emitted), specialized_(specialized),
              reason_(std::move(reason)), header_text_(std::move(header_text)),
              cu_text_(std::move(cu_text)), kernel_symbol_(std::move(kernel_symbol)) {}

        [[nodiscard]] Strategy intended_strategy() const noexcept { return intended_; }

        [[nodiscard]] Strategy emitted_strategy() const noexcept { return emitted_; }

        [[nodiscard]] bool specialized() const noexcept { return specialized_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] const std::string& header_text() const noexcept { return header_text_; }

        [[nodiscard]] const std::string& cu_text() const noexcept { return cu_text_; }

        [[nodiscard]] const std::string& kernel_symbol() const noexcept { return kernel_symbol_; }

    private:
        Strategy intended_ = Strategy::S0Bytecode;
        Strategy emitted_ = Strategy::S0Bytecode;
        bool specialized_ = false;
        std::string reason_;
        std::string header_text_;
        std::string cu_text_;
        std::string kernel_symbol_;
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
        }
        return "S0_bytecode";
    }

    /// Classify decrypt HotLoop for fused hist emit (does not emit sources).
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

        const bool uses_i = DslOptimize::depends_on_var(dec, "i");
        const bool uses_cipher = DslOptimize::depends_on_var(dec, cipher_var);
        if (!uses_cipher) {
            return Selection{Strategy::S0Bytecode, "decrypt_step does not reference cipher_var"};
        }

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

    /// Emit fused-hist sources for decrypt search path.
    /// Skeleton: always returns S0 (empty sources); `intended_strategy` reports the
    /// classified target for later specialized emitters.
    [[nodiscard]] static StatusOr<EmitBundle>
    emit_decrypt_hist(const TheoryIr& theory, std::string_view cipher_var = "x",
                      const std::vector<DslOptimize::Hoist>& /*decrypt_hoists*/ = {}) {
        if (theory.name().empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistChi2Emit: theory name must be non-empty")
                .to_status();
        }
        Selection sel = select_strategy(theory, cipher_var);
        if (sel.strategy() == Strategy::S0Bytecode) {
            return EmitBundle{Strategy::S0Bytecode, Strategy::S0Bytecode, false, sel.reason(), "",
                              "", ""};
        }
        // Specialized emit not implemented yet — soft S0 fallback (export stays correct).
        return EmitBundle{sel.strategy(), Strategy::S0Bytecode, false,
                          std::string("skeleton: ") + strategy_str(sel.strategy()) +
                              " classified but emit not implemented; fallback S0 (" + sel.reason() +
                              ")",
                          "", "", ""};
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
};

#endif // THEORY_HIST_CHI2_EMIT_HPP
