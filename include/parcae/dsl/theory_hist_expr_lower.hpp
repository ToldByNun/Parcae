#ifndef THEORY_HIST_EXPR_LOWER_HPP
#define THEORY_HIST_EXPR_LOWER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_emit_cuda.hpp"
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

/// Lower decrypt HotLoop into a bounded S3 scalar-hist program + device fragment.
///
/// Caps are tighter than S0/`TheoryChi2Batch`: programs that exceed them must soft-fall
/// to bytecode (caller). Rejects autokey / prefer_branch / missing `i` (those are S4 / S0 / S1).
///
/// No C++ namespaces. See `docs/architecture/dsl-smart-hist.md`.
class TheoryHistExprLower {
public:
    /// S3 in-lib twin caps (must fit TheoryChi2Batch shared staging).
    static constexpr std::uint32_t kMaxProgramOps = 128;
    static constexpr std::uint16_t kMaxDeviceStack = 16;
    static constexpr std::uint16_t kMaxSlots = 16;

    class Lowered {
    public:
        Lowered(Z29Bytecode::Program program, std::string device_cpp, std::string reason)
            : program_(std::move(program)), device_cpp_(std::move(device_cpp)),
              reason_(std::move(reason)) {}

        [[nodiscard]] const Z29Bytecode::Program& program() const noexcept { return program_; }

        [[nodiscard]] const std::string& device_cpp() const noexcept { return device_cpp_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] std::uint32_t op_count() const noexcept {
            return static_cast<std::uint32_t>(program_.ops.size());
        }

        [[nodiscard]] std::uint16_t max_stack() const noexcept { return program_.max_stack; }

        [[nodiscard]] std::uint16_t slot_count() const noexcept {
            return static_cast<std::uint16_t>(program_.slot_names.size());
        }

        [[nodiscard]] bool binds_index_i() const noexcept { return program_.binds_index_i; }

    private:
        Z29Bytecode::Program program_;
        std::string device_cpp_;
        std::string reason_;
    };

    /// Lower decrypt for S3 scalar hist. Soft-fail reasons are returned as Status.
    [[nodiscard]] static StatusOr<Lowered> lower_decrypt(const TheoryIr& theory,
                                                         std::string_view cipher_var = "x") {
        if (!theory.decrypt_step()) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: missing decrypt_step")
                .to_status();
        }
        if (cipher_var.empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: empty cipher_var")
                .to_status();
        }
        const Z29Expr& dec = *theory.decrypt_step();
        if (has_autokey(dec)) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: autokey is S4/S0, not S3")
                .to_status();
        }
        if (has_prefer_branch(dec)) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: prefer_branch Select is S0-only")
                .to_status();
        }
        if (!DslOptimize::depends_on_var(dec, cipher_var)) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: decrypt does not reference cipher_var")
                .to_status();
        }
        if (!DslOptimize::depends_on_var(dec, "i")) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: no stream index i — use S1/FxOnly")
                .to_status();
        }

        StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt, cipher_var);
        if (!prog.ok()) {
            return prog.status();
        }
        if (!prog.value().binds_index_i) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: compiled program does not bind i")
                .to_status();
        }
        if (contains_autokey_op(prog.value())) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: AutokeyShift op not allowed in S3")
                .to_status();
        }

        Status caps = check_caps(prog.value());
        if (!caps.ok()) {
            return caps;
        }

        StatusOr<std::string> cpp =
            DslEmitCuda::emit_expr(theory.decrypt_step(), cipher_var, "x");
        if (!cpp.ok()) {
            return cpp.status();
        }

        return Lowered{std::move(prog.value()), std::move(cpp.value()),
                       "bounded S3 lower (ops/stack/slots within caps)"};
    }

    [[nodiscard]] static Status check_caps(const Z29Bytecode::Program& prog) {
        if (prog.ops.size() > kMaxProgramOps) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: op_count exceeds S3 kMaxProgramOps")
                .to_status();
        }
        if (prog.max_stack > kMaxDeviceStack) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: max_stack exceeds S3 kMaxDeviceStack")
                .to_status();
        }
        if (prog.slot_names.size() > kMaxSlots) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: slot_count exceeds S3 kMaxSlots")
                .to_status();
        }
        if (prog.ops.size() != prog.imm.size()) {
            return DslDiag::make(DslRuleId::E032_primitive_body,
                                 "TheoryHistExprLower: ops/imm size mismatch")
                .to_status();
        }
        return Status::success();
    }

    [[nodiscard]] static bool within_caps(const Z29Bytecode::Program& prog) noexcept {
        return check_caps(prog).ok();
    }

private:
    TheoryHistExprLower() = delete;

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
        if (expr.kind() == Kind::Select) {
            return (expr.cond() && has_prefer_branch(*expr.cond())) ||
                   (expr.if_true() && has_prefer_branch(*expr.if_true())) ||
                   (expr.if_false() && has_prefer_branch(*expr.if_false()));
        }
        if (expr.kind() == Kind::Call) {
            for (const Z29Expr::Ptr& a : expr.args()) {
                if (a && has_prefer_branch(*a)) {
                    return true;
                }
            }
            return false;
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

    [[nodiscard]] static bool contains_autokey_op(const Z29Bytecode::Program& prog) {
        for (Z29Bytecode::Op op : prog.ops) {
            if (op == Z29Bytecode::Op::AutokeyShift) {
                return true;
            }
        }
        return false;
    }
};

#endif // THEORY_HIST_EXPR_LOWER_HPP
