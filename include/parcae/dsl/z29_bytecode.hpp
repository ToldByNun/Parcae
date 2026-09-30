#ifndef Z29_BYTECODE_HPP
#define Z29_BYTECODE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/matrix_ir.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/math/autokey_ring.hpp"
#include "parcae/transform/transform_buffer.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Stack bytecode for HotLoop `Z29Expr` trees (Theory decrypt/encrypt steps).
///
/// Host eval is the CPU reference for a future fused CUDA χ² batch path.
/// Encoding is post-order; `Select` is **eager** (both arms evaluated) so the
/// same program lowers to a branch-free device mux later. Pure Z29 trees have
/// no side effects — domain errors in a dead arm still fail (unlike short-circuit
/// `Z29Expr::eval`). Prefer `DslOptimize` dead-arm fold before encode when needed.
///
/// Layout is POD-friendly: `ops[]` + `imm[]` (same length), dense `slots[]` at
/// runtime. Slot 0 is always the cipher var; remaining slots are theory params
/// then optional position `i`.
class Z29Bytecode {
public:
    enum class Op : std::uint8_t {
        Const = 0, ///< push imm (0..28)
        Load = 1,  ///< push slots[imm]
        Add = 2,
        Sub = 3,
        Mul = 4,
        Div = 5,
        FloorDiv = 6,
        Mod = 7,
        Pow = 8,
        BitAnd = 9,
        BitOr = 10,
        BitXor = 11,
        LShift = 12,
        RShift = 13,
        Eq = 14,
        Ne = 15,
        Lt = 16,
        Le = 17,
        Gt = 18,
        Ge = 19,
        BoolAnd = 20,
        BoolOr = 21,
        Neg = 22,
        Inv = 23,
        Atbash = 24,
        BitNot = 25,
        BoolNot = 26,
        /// pop f, t, c → push (c != 0 ? t : f)  (eager arms already on stack)
        Select = 27,
        /// pop lag → push AutokeyRing::shift(stream, i, lag)
        AutokeyShift = 28,
    };

    struct Program {
        std::vector<Op> ops;
        std::vector<std::uint8_t> imm; ///< parallel to ops (Const value or Load slot)
        std::vector<std::string> slot_names;
        std::string cipher_var{"x"};
        bool binds_index_i = false; ///< slot for `i` = stream index % 29
        std::uint16_t max_stack = 0;
        std::uint16_t cipher_slot = 0;
        std::uint16_t index_slot = 0; ///< valid when binds_index_i
    };

    /// Compile a HotLoop expression. `param_names` become Load slots (after cipher).
    /// Unbound vars other than `cipher_var` / `i` / listed params → error.
    [[nodiscard]] static StatusOr<Program> compile_expr(const Z29Expr::Ptr& step,
                                                        std::string_view cipher_var,
                                                        const std::vector<std::string>& param_names) {
        if (!step) {
            return fail(DslRuleId::E032_primitive_body, "bytecode: null step");
        }
        if (cipher_var.empty()) {
            return fail(DslRuleId::E032_primitive_body, "bytecode: cipher_var must be non-empty");
        }
        for (const std::string& p : param_names) {
            if (p == cipher_var) {
                return fail(DslRuleId::E032_primitive_body,
                            "bytecode: param '" + p + "' collides with cipher_var");
            }
            if (p == "i") {
                return fail(DslRuleId::E032_primitive_body,
                            "bytecode: param name 'i' reserved for stream index");
            }
        }

        Program prog;
        prog.cipher_var = std::string(cipher_var);
        prog.slot_names.push_back(prog.cipher_var);
        prog.cipher_slot = 0;
        for (const std::string& p : param_names) {
            prog.slot_names.push_back(p);
        }

        EncodeCtx ctx{prog, cipher_var};
        // Reserve index slot if the tree mentions `i` (or always for theory keyed_stream —
        // caller sets binds via compile_theory).
        StatusOr<bool> uses_i = expr_uses_var(step, "i");
        if (!uses_i.ok()) {
            return uses_i.status();
        }
        if (uses_i.value()) {
            prog.binds_index_i = true;
            prog.index_slot = static_cast<std::uint16_t>(prog.slot_names.size());
            prog.slot_names.push_back("i");
        }

        Status st = encode(step, ctx);
        if (!st.ok()) {
            return st;
        }
        if (ctx.stack_depth != 1) {
            return fail(DslRuleId::E032_primitive_body,
                        "bytecode: encode left stack_depth=" + std::to_string(ctx.stack_depth) +
                            " (expected 1)");
        }
        prog.max_stack = ctx.max_stack;
        return prog;
    }

    /// Compile encrypt or decrypt step of a TheoryIr (params in declaration order).
    [[nodiscard]] static StatusOr<Program> compile_theory(const TheoryIr& theory,
                                                          TransformDirection direction,
                                                          std::string_view cipher_var = "x") {
        const Z29Expr::Ptr& step = (direction == TransformDirection::Encrypt)
                                       ? theory.encrypt_step()
                                       : theory.decrypt_step();
        if (!step) {
            return fail(DslRuleId::E032_primitive_body,
                        std::string("bytecode: theory '") + theory.name() + "' missing " +
                            (direction == TransformDirection::Encrypt ? "encrypt_step"
                                                                      : "decrypt_step"));
        }
        std::vector<std::string> names;
        names.reserve(theory.params().size());
        bool i_is_param = false;
        for (const ParamIr& p : theory.params()) {
            if (p.name() == "i") {
                i_is_param = true;
            }
            names.push_back(p.name());
        }
        StatusOr<Program> prog = compile_expr(step, cipher_var, names);
        if (!prog.ok()) {
            return prog.status();
        }
        // Match DslIrApplicator: bind stream index as `i` unless `i` is a theory param.
        if (!i_is_param && !prog.value().binds_index_i) {
            // Still allocate index slot for applicator parity on keyed streams that
            // do not mention `i` (harmless unused slot). Skip — applicator only binds
            // when writing env; bytecode only needs the slot if Load("i") exists.
        }
        if (!i_is_param) {
            // If tree did not reference `i`, leave binds_index_i false (no Load).
            // If it did, compile_expr already set binds_index_i.
        } else {
            prog.value().binds_index_i = false; // `i` is a normal param slot
        }
        return prog;
    }

    /// Load `apply_ir.json` object / TheoryApplyIr schema → compile decrypt (default)
    /// or encrypt step.
    [[nodiscard]] static StatusOr<Program> compile_apply_ir(const nlohmann::json& apply_ir,
                                                            TransformDirection direction =
                                                                TransformDirection::Decrypt) {
        StatusOr<TheoryIr> theory = TheoryApplyIr::from_json(apply_ir);
        if (!theory.ok()) {
            return theory.status();
        }
        std::string cipher = "x";
        if (apply_ir.contains("cipher_var") && apply_ir.at("cipher_var").is_string()) {
            cipher = apply_ir.at("cipher_var").get<std::string>();
        }
        return compile_theory(theory.value(), direction, cipher);
    }

    /// Bind theory JSON params into a dense slot vector matching `prog.slot_names`
    /// (cipher / index slots left as 0 placeholders).
    [[nodiscard]] static StatusOr<std::vector<Index29>>
    bind_theory_slots(const Program& prog, const TheoryIr& theory, const nlohmann::json& params) {
        if (!params.is_object()) {
            return fail(DslRuleId::E040_param_domain, "bytecode: params must be a JSON object");
        }
        if (params.size() != theory.params().size()) {
            return fail(DslRuleId::E040_param_domain,
                        "bytecode: theory expects " + std::to_string(theory.params().size()) +
                            " params, got " + std::to_string(params.size()));
        }
        std::vector<Index29> slots(prog.slot_names.size(), Index29{0});
        for (std::size_t s = 0; s < prog.slot_names.size(); ++s) {
            const std::string& name = prog.slot_names[s];
            if (name == prog.cipher_var) {
                continue;
            }
            if (prog.binds_index_i && name == "i") {
                continue;
            }
            if (!params.contains(name)) {
                return fail(DslRuleId::E040_param_domain, "bytecode: missing param '" + name + "'");
            }
            const nlohmann::json& v = params.at(name);
            if (!v.is_number_integer()) {
                return fail(DslRuleId::E040_param_domain,
                            "bytecode: param '" + name + "' must be an integer");
            }
            const std::int64_t raw = v.get<std::int64_t>();
            if (raw < 0 || raw >= Index29::modulus) {
                return fail(DslRuleId::E040_param_domain,
                            "bytecode: param '" + name + "' outside Index29 domain");
            }
            const Index29 idx{static_cast<std::uint8_t>(raw)};
            for (const ParamIr& p : theory.params()) {
                if (p.name() == name) {
                    Status domain = p.check_value(idx);
                    if (!domain.ok()) {
                        return domain;
                    }
                    break;
                }
            }
            slots[s] = idx;
        }
        return slots;
    }

    /// Evaluate one index with pre-bound param slots (cipher/index overwritten).
    [[nodiscard]] static StatusOr<Index29> eval_at(const Program& prog,
                                                   std::span<Index29> slots,
                                                   std::span<const Index29> stream, std::size_t i) {
        if (prog.ops.size() != prog.imm.size()) {
            return fail(DslRuleId::E032_primitive_body, "bytecode: ops/imm size mismatch");
        }
        if (slots.size() != prog.slot_names.size()) {
            return fail(DslRuleId::E032_primitive_body, "bytecode: slots size mismatch");
        }
        if (i >= stream.size()) {
            return fail(DslRuleId::E032_primitive_body, "bytecode: index out of range");
        }
        slots[prog.cipher_slot] = stream[i];
        if (prog.binds_index_i) {
            slots[prog.index_slot] =
                Index29{static_cast<std::uint8_t>(i % Index29::modulus)};
        }

        std::vector<Index29> stack;
        stack.reserve(prog.max_stack == 0 ? 8 : prog.max_stack);
        for (std::size_t pc = 0; pc < prog.ops.size(); ++pc) {
            const Op op = prog.ops[pc];
            const std::uint8_t imm = prog.imm[pc];
            switch (op) {
            case Op::Const:
                if (imm >= Index29::modulus) {
                    return fail(DslRuleId::E040_param_domain, "bytecode: Const imm out of range");
                }
                stack.push_back(Index29{imm});
                break;
            case Op::Load:
                if (imm >= slots.size()) {
                    return fail(DslRuleId::E032_primitive_body, "bytecode: Load slot OOB");
                }
                stack.push_back(slots[imm]);
                break;
            case Op::AutokeyShift: {
                if (stack.empty()) {
                    return fail(DslRuleId::E032_primitive_body, "bytecode: AutokeyShift stack underrun");
                }
                const Index29 lag = stack.back();
                stack.pop_back();
                stack.push_back(AutokeyRing::shift(stream, i, lag));
                break;
            }
            case Op::Select: {
                if (stack.size() < 3) {
                    return fail(DslRuleId::E032_primitive_body, "bytecode: Select stack underrun");
                }
                const Index29 f = stack.back();
                stack.pop_back();
                const Index29 t = stack.back();
                stack.pop_back();
                const Index29 c = stack.back();
                stack.pop_back();
                stack.push_back(c.value() != 0 ? t : f);
                break;
            }
            default: {
                Status st = eval_op(op, stack);
                if (!st.ok()) {
                    return st;
                }
                break;
            }
            }
        }
        if (stack.size() != 1) {
            return fail(DslRuleId::E032_primitive_body,
                        "bytecode: eval left stack_depth=" + std::to_string(stack.size()));
        }
        return stack.back();
    }

    /// Stream apply matching `DslIrApplicator` interrupt / none_by_design rules when
    /// `theory` is provided; otherwise only empty interrupt is accepted.
    [[nodiscard]] static Status apply_into(const Program& prog, std::span<Index29> slots,
                                           std::span<const Index29> input, std::span<Index29> output,
                                           const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        Status range =
            TransformBuffer::validate_interrupt_range(interrupt, input.size(), "Z29Bytecode");
        if (!range.ok()) {
            return range;
        }
        const auto skips = TransformBuffer::skip_span(interrupt);
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (TransformBuffer::should_skip(skips, i)) {
                output[i] = input[i];
                continue;
            }
            StatusOr<Index29> out = eval_at(prog, slots, input, i);
            if (!out.ok()) {
                return out.status();
            }
            output[i] = out.value();
        }
        return Status::success();
    }

    [[nodiscard]] static Status
    apply_into_theory(const Program& prog, const TheoryIr& theory, const nlohmann::json& params,
                      std::span<const Index29> input, std::span<Index29> output,
                      const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (theory.interrupt_mode() == TheoryIr::InterruptMode::NoneByDesign && !interrupt.empty()) {
            return DslDiag::make(DslRuleId::E030_interrupt_policy,
                                 "theory '" + theory.name() +
                                     "' declares interrupts=none_by_design; non-empty "
                                     "InterruptPolicy rejected (CPU↔CUDA parity)")
                .to_status();
        }
        StatusOr<std::vector<Index29>> slots = bind_theory_slots(prog, theory, params);
        if (!slots.ok()) {
            return slots.status();
        }
        return apply_into(prog, slots.value(), input, output, interrupt);
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>>
    apply_theory(const Program& prog, const TheoryIr& theory, const nlohmann::json& params,
                 std::span<const Index29> input,
                 const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        std::vector<Index29> out(input.size());
        Status st = apply_into_theory(prog, theory, params, input, out, interrupt);
        if (!st.ok()) {
            return st;
        }
        return out;
    }

private:
    Z29Bytecode() = delete;

    struct EncodeCtx {
        Program& prog;
        std::string_view cipher_var;
        std::uint16_t stack_depth = 0;
        std::uint16_t max_stack = 0;

        void push_depth(std::uint16_t delta = 1) {
            stack_depth = static_cast<std::uint16_t>(stack_depth + delta);
            if (stack_depth > max_stack) {
                max_stack = stack_depth;
            }
        }
        void pop_depth(std::uint16_t delta = 1) {
            stack_depth = static_cast<std::uint16_t>(stack_depth - delta);
        }
    };

    [[nodiscard]] static Status fail(std::string_view rule_id, std::string message) {
        return DslDiag::make(rule_id, std::move(message)).to_status();
    }

    [[nodiscard]] static Status emit(EncodeCtx& ctx, Op op, std::uint8_t imm = 0) {
        ctx.prog.ops.push_back(op);
        ctx.prog.imm.push_back(imm);
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::uint8_t> slot_of(const Program& prog,
                                                        std::string_view name) {
        for (std::size_t i = 0; i < prog.slot_names.size(); ++i) {
            if (prog.slot_names[i] == name) {
                return static_cast<std::uint8_t>(i);
            }
        }
        return fail(DslRuleId::E032_primitive_body,
                    "bytecode: unbound variable '" + std::string(name) + "'");
    }

    [[nodiscard]] static StatusOr<bool> expr_uses_var(const Z29Expr::Ptr& expr,
                                                      std::string_view name) {
        if (!expr) {
            return fail(DslRuleId::E032_primitive_body, "bytecode: null expr in uses_var");
        }
        using Kind = Z29Expr::Kind;
        if (expr->kind() == Kind::Var) {
            return expr->name() == name;
        }
        if (expr->kind() == Kind::Const) {
            return false;
        }
        if (expr->kind() == Kind::Call) {
            for (const Z29Expr::Ptr& a : expr->args()) {
                StatusOr<bool> u = expr_uses_var(a, name);
                if (!u.ok()) {
                    return u.status();
                }
                if (u.value()) {
                    return true;
                }
            }
            return false;
        }
        if (expr->kind() == Kind::Select) {
            StatusOr<bool> c = expr_uses_var(expr->cond(), name);
            if (!c.ok()) {
                return c.status();
            }
            if (c.value()) {
                return true;
            }
            StatusOr<bool> t = expr_uses_var(expr->if_true(), name);
            if (!t.ok()) {
                return t.status();
            }
            if (t.value()) {
                return true;
            }
            return expr_uses_var(expr->if_false(), name);
        }
        if (Z29Expr::is_binary(expr->kind())) {
            StatusOr<bool> l = expr_uses_var(expr->left(), name);
            if (!l.ok()) {
                return l.status();
            }
            if (l.value()) {
                return true;
            }
            return expr_uses_var(expr->right(), name);
        }
        if (Z29Expr::is_unary(expr->kind())) {
            return expr_uses_var(expr->arg(), name);
        }
        return false;
    }

    [[nodiscard]] static Status encode(const Z29Expr::Ptr& expr, EncodeCtx& ctx) {
        if (!expr) {
            return fail(DslRuleId::E032_primitive_body, "bytecode: null Z29Expr");
        }
        using Kind = Z29Expr::Kind;
        switch (expr->kind()) {
        case Kind::Const:
            ctx.push_depth();
            return emit(ctx, Op::Const, expr->const_value());
        case Kind::Var: {
            StatusOr<std::uint8_t> slot = slot_of(ctx.prog, expr->name());
            if (!slot.ok()) {
                return slot.status();
            }
            ctx.push_depth();
            return emit(ctx, Op::Load, slot.value());
        }
        case Kind::Call:
            return encode_call(expr, ctx);
        case Kind::Select: {
            Status st = encode(expr->cond(), ctx);
            if (!st.ok()) {
                return st;
            }
            st = encode(expr->if_true(), ctx);
            if (!st.ok()) {
                return st;
            }
            st = encode(expr->if_false(), ctx);
            if (!st.ok()) {
                return st;
            }
            ctx.pop_depth(3);
            ctx.push_depth();
            return emit(ctx, Op::Select);
        }
        default:
            break;
        }
        if (Z29Expr::is_binary(expr->kind())) {
            Status st = encode(expr->left(), ctx);
            if (!st.ok()) {
                return st;
            }
            st = encode(expr->right(), ctx);
            if (!st.ok()) {
                return st;
            }
            ctx.pop_depth(2);
            ctx.push_depth();
            return emit(ctx, op_for_binary(expr->kind()));
        }
        if (Z29Expr::is_unary(expr->kind())) {
            Status st = encode(expr->arg(), ctx);
            if (!st.ok()) {
                return st;
            }
            ctx.pop_depth();
            ctx.push_depth();
            return emit(ctx, op_for_unary(expr->kind()));
        }
        return fail(DslRuleId::E032_primitive_body, "bytecode: unknown Z29Expr kind");
    }

    [[nodiscard]] static Op op_for_binary(Z29Expr::Kind k) {
        using Kind = Z29Expr::Kind;
        switch (k) {
        case Kind::Add:
            return Op::Add;
        case Kind::Sub:
            return Op::Sub;
        case Kind::Mul:
            return Op::Mul;
        case Kind::Div:
            return Op::Div;
        case Kind::FloorDiv:
            return Op::FloorDiv;
        case Kind::Mod:
            return Op::Mod;
        case Kind::Pow:
            return Op::Pow;
        case Kind::BitAnd:
            return Op::BitAnd;
        case Kind::BitOr:
            return Op::BitOr;
        case Kind::BitXor:
            return Op::BitXor;
        case Kind::LShift:
            return Op::LShift;
        case Kind::RShift:
            return Op::RShift;
        case Kind::Eq:
            return Op::Eq;
        case Kind::Ne:
            return Op::Ne;
        case Kind::Lt:
            return Op::Lt;
        case Kind::Le:
            return Op::Le;
        case Kind::Gt:
            return Op::Gt;
        case Kind::Ge:
            return Op::Ge;
        case Kind::BoolAnd:
            return Op::BoolAnd;
        case Kind::BoolOr:
            return Op::BoolOr;
        default:
            return Op::Add;
        }
    }

    [[nodiscard]] static Op op_for_unary(Z29Expr::Kind k) {
        using Kind = Z29Expr::Kind;
        switch (k) {
        case Kind::Neg:
            return Op::Neg;
        case Kind::Inv:
            return Op::Inv;
        case Kind::Atbash:
            return Op::Atbash;
        case Kind::BitNot:
            return Op::BitNot;
        case Kind::BoolNot:
            return Op::BoolNot;
        default:
            return Op::Neg;
        }
    }

    [[nodiscard]] static Status encode_call(const Z29Expr::Ptr& expr, EncodeCtx& ctx) {
        const std::string& n = expr->name();
        const auto& args = expr->args();

        const auto as_bin = [&](Op op) -> Status {
            if (args.size() != 2) {
                return fail(DslRuleId::E032_primitive_body,
                            "bytecode: call '" + n + "' expects 2 args");
            }
            Status st = encode(args[0], ctx);
            if (!st.ok()) {
                return st;
            }
            st = encode(args[1], ctx);
            if (!st.ok()) {
                return st;
            }
            ctx.pop_depth(2);
            ctx.push_depth();
            return emit(ctx, op);
        };
        const auto as_unary = [&](Op op) -> Status {
            if (args.size() != 1) {
                return fail(DslRuleId::E032_primitive_body,
                            "bytecode: call '" + n + "' expects 1 arg");
            }
            Status st = encode(args[0], ctx);
            if (!st.ok()) {
                return st;
            }
            ctx.pop_depth();
            ctx.push_depth();
            return emit(ctx, op);
        };

        if (n == "z29_add") {
            return as_bin(Op::Add);
        }
        if (n == "z29_sub") {
            return as_bin(Op::Sub);
        }
        if (n == "z29_mul") {
            return as_bin(Op::Mul);
        }
        if (n == "z29_div") {
            return as_bin(Op::Div);
        }
        if (n == "z29_floordiv") {
            return as_bin(Op::FloorDiv);
        }
        if (n == "z29_mod") {
            return as_bin(Op::Mod);
        }
        if (n == "z29_pow") {
            return as_bin(Op::Pow);
        }
        if (n == "z29_bit_and") {
            return as_bin(Op::BitAnd);
        }
        if (n == "z29_bit_or") {
            return as_bin(Op::BitOr);
        }
        if (n == "z29_bit_xor") {
            return as_bin(Op::BitXor);
        }
        if (n == "z29_lshift") {
            return as_bin(Op::LShift);
        }
        if (n == "z29_rshift") {
            return as_bin(Op::RShift);
        }
        if (n == "z29_eq") {
            return as_bin(Op::Eq);
        }
        if (n == "z29_ne") {
            return as_bin(Op::Ne);
        }
        if (n == "z29_lt") {
            return as_bin(Op::Lt);
        }
        if (n == "z29_le") {
            return as_bin(Op::Le);
        }
        if (n == "z29_gt") {
            return as_bin(Op::Gt);
        }
        if (n == "z29_ge") {
            return as_bin(Op::Ge);
        }
        if (n == "z29_bool_and") {
            return as_bin(Op::BoolAnd);
        }
        if (n == "z29_bool_or") {
            return as_bin(Op::BoolOr);
        }
        if (n == "z29_inv") {
            return as_unary(Op::Inv);
        }
        if (n == "z29_neg") {
            return as_unary(Op::Neg);
        }
        if (n == "z29_atbash") {
            return as_unary(Op::Atbash);
        }
        if (n == "z29_bit_not") {
            return as_unary(Op::BitNot);
        }
        if (n == "z29_bool_not") {
            return as_unary(Op::BoolNot);
        }
        if (n == "z29_select") {
            if (args.size() != 3) {
                return fail(DslRuleId::E032_primitive_body, "bytecode: z29_select expects 3 args");
            }
            Status st = encode(args[0], ctx);
            if (!st.ok()) {
                return st;
            }
            st = encode(args[1], ctx);
            if (!st.ok()) {
                return st;
            }
            st = encode(args[2], ctx);
            if (!st.ok()) {
                return st;
            }
            ctx.pop_depth(3);
            ctx.push_depth();
            return emit(ctx, Op::Select);
        }
        if (n == "z29_autokey_shift") {
            if (args.size() != 2 || !args[0] || !args[1]) {
                return fail(DslRuleId::E032_primitive_body,
                            "bytecode: z29_autokey_shift expects (stream, lag)");
            }
            if (args[0]->kind() != Z29Expr::Kind::Var || args[0]->name() != ctx.cipher_var) {
                return fail(DslRuleId::E032_primitive_body,
                            "bytecode: z29_autokey_shift stream must be cipher var '" +
                                std::string(ctx.cipher_var) + "'");
            }
            Status st = encode(args[1], ctx);
            if (!st.ok()) {
                return st;
            }
            ctx.pop_depth();
            ctx.push_depth();
            return emit(ctx, Op::AutokeyShift);
        }
        if (n == "z29_det") {
            // Flattened matrix entries (4 or 9) → expand to scalar tree then encode.
            StatusOr<MatrixIr> mat = MatrixIr::make(std::vector<Z29Expr::Ptr>(args.begin(), args.end()));
            if (!mat.ok()) {
                return mat.status();
            }
            return encode(mat.value().det_expr(), ctx);
        }
        if (n == "z29_matmul") {
            // Prefer BuildIr form z29_matmul(...)[i]; accept flattened matrix||vector.
            // 2x2: 4+2=6; 3x3: 9+3=12. Encode row-0 result only if bare call —
            // match CUDA emit: component 0 when not subscripted.
            if (args.size() != 6 && args.size() != 12) {
                return fail(DslRuleId::E032_primitive_body,
                            "bytecode: z29_matmul expects 6 or 12 flattened entries "
                            "(or BuildIr scalar expansion)");
            }
            const std::size_t n_mat = (args.size() == 6) ? 2 : 3;
            std::vector<Z29Expr::Ptr> mentries(args.begin(), args.begin() + static_cast<std::ptrdiff_t>(n_mat * n_mat));
            std::vector<Z29Expr::Ptr> vec(args.begin() + static_cast<std::ptrdiff_t>(n_mat * n_mat),
                                          args.end());
            StatusOr<MatrixIr> mat = MatrixIr::make(std::move(mentries));
            if (!mat.ok()) {
                return mat.status();
            }
            StatusOr<std::vector<Z29Expr::Ptr>> rows = mat.value().mul_vec_exprs(vec);
            if (!rows.ok()) {
                return rows.status();
            }
            return encode(rows.value()[0], ctx);
        }
        return fail(DslRuleId::E032_primitive_body,
                    "bytecode: unknown call '" + n + "' (inline @define_primitive first)");
    }

    /// Binary/unary ops that mutate `stack` in place; Select/Const/Load/Autokey handled above.
    [[nodiscard]] static Status eval_op(Op op, std::vector<Index29>& stack) {
        auto need1 = [&]() -> StatusOr<Index29> {
            if (stack.empty()) {
                return fail(DslRuleId::E032_primitive_body, "bytecode: unary stack underrun");
            }
            const Index29 a = stack.back();
            stack.pop_back();
            return a;
        };
        auto need2 = [&]() -> StatusOr<std::pair<Index29, Index29>> {
            if (stack.size() < 2) {
                return fail(DslRuleId::E032_primitive_body, "bytecode: binary stack underrun");
            }
            const Index29 b = stack.back();
            stack.pop_back();
            const Index29 a = stack.back();
            stack.pop_back();
            return std::make_pair(a, b);
        };

        switch (op) {
        case Op::Add: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::add(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Sub: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::sub(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Mul: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::mul(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Div: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            if (ab.value().second.value() == 0) {
                return fail(DslRuleId::E040_param_domain, "bytecode: div by 0");
            }
            stack.push_back(Z29::mul(ab.value().first, Z29::inv(ab.value().second)));
            return Status::success();
        }
        case Op::FloorDiv: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            if (ab.value().second.value() == 0) {
                return fail(DslRuleId::E040_param_domain, "bytecode: floordiv by 0");
            }
            stack.push_back(Z29::floor_div(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Mod: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            if (ab.value().second.value() == 0) {
                return fail(DslRuleId::E040_param_domain, "bytecode: mod by 0");
            }
            stack.push_back(Index29{
                static_cast<std::uint8_t>(ab.value().first.value() % ab.value().second.value())});
            return Status::success();
        }
        case Op::Pow: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::pow(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::BitAnd: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::bit_and(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::BitOr: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::bit_or(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::BitXor: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::bit_xor(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::LShift: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::lshift(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::RShift: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::rshift(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Eq: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::eq(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Ne: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::ne(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Lt: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::lt(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Le: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::le(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Gt: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::gt(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Ge: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::ge(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::BoolAnd: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::bool_and(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::BoolOr: {
            StatusOr<std::pair<Index29, Index29>> ab = need2();
            if (!ab.ok()) {
                return ab.status();
            }
            stack.push_back(Z29::bool_or(ab.value().first, ab.value().second));
            return Status::success();
        }
        case Op::Neg: {
            StatusOr<Index29> a = need1();
            if (!a.ok()) {
                return a.status();
            }
            stack.push_back(Z29::neg(a.value()));
            return Status::success();
        }
        case Op::Inv: {
            StatusOr<Index29> a = need1();
            if (!a.ok()) {
                return a.status();
            }
            StatusOr<Index29> inv = Z29::try_inv(a.value());
            if (!inv.ok()) {
                return fail(DslRuleId::E040_param_domain, "bytecode: inv(0)");
            }
            stack.push_back(inv.value());
            return Status::success();
        }
        case Op::Atbash: {
            StatusOr<Index29> a = need1();
            if (!a.ok()) {
                return a.status();
            }
            stack.push_back(Z29::atbash(a.value()));
            return Status::success();
        }
        case Op::BitNot: {
            StatusOr<Index29> a = need1();
            if (!a.ok()) {
                return a.status();
            }
            stack.push_back(Z29::bit_not(a.value()));
            return Status::success();
        }
        case Op::BoolNot: {
            StatusOr<Index29> a = need1();
            if (!a.ok()) {
                return a.status();
            }
            stack.push_back(Z29::bool_not(a.value()));
            return Status::success();
        }
        default:
            return fail(DslRuleId::E032_primitive_body, "bytecode: eval_op bad opcode");
        }
    }
};

#endif // Z29_BYTECODE_HPP
