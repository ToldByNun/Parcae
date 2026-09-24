#ifndef Z29_EXPR_HPP
#define Z29_EXPR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

/// Immutable \(\mathbb{Z}_{29}\) expression IR for the theory DSL (docs/spec/dsl.md).
/// Construction builds trees only; host `eval` uses `Z29`, `eval_cuda_mirror`
/// uses `Z29Device` op-sequence semantics (never aborts on `inv(0)` — returns E040).
/// Non-builtin primitive calls fail until a registry binds them (PrimitiveIr + DslBuildIr).
class Z29Expr {
public:
    enum class Kind : std::uint8_t {
        Const = 0,
        Var,
        // Arithmetic (ring)
        Add,
        Sub,
        Mul,
        Div,       // modular: mul(x, inv(y))
        FloorDiv,  // integer // on representatives
        Mod,
        Pow,
        Neg,
        Inv,
        Atbash,
        // Bitwise / shift (representatives, then mod 29)
        BitAnd,
        BitOr,
        BitXor,
        BitNot,  // (~x) mod 29 == 28-x
        LShift,
        RShift,
        // Comparisons → 0/1
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        // Boolean-ish on nonzero → 0/1
        BoolAnd,
        BoolOr,
        BoolNot,
        /// Branch-free mux: `select(cond, t, f)` → `cond != 0 ? t : f` (Z29 0/1).
        Select,
        Call,
    };

    using Ptr = std::shared_ptr<Z29Expr>;
    using Env = std::unordered_map<std::string, Index29>;

    [[nodiscard]] static bool is_binary(Kind k) noexcept {
        switch (k) {
        case Kind::Add:
        case Kind::Sub:
        case Kind::Mul:
        case Kind::Div:
        case Kind::FloorDiv:
        case Kind::Mod:
        case Kind::Pow:
        case Kind::BitAnd:
        case Kind::BitOr:
        case Kind::BitXor:
        case Kind::LShift:
        case Kind::RShift:
        case Kind::Eq:
        case Kind::Ne:
        case Kind::Lt:
        case Kind::Le:
        case Kind::Gt:
        case Kind::Ge:
        case Kind::BoolAnd:
        case Kind::BoolOr:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]] static bool is_unary(Kind k) noexcept {
        switch (k) {
        case Kind::Neg:
        case Kind::Inv:
        case Kind::Atbash:
        case Kind::BitNot:
        case Kind::BoolNot:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]] static bool is_select(Kind k) noexcept {
        return k == Kind::Select;
    }

    [[nodiscard]] static StatusOr<Ptr> constant(std::int64_t value) {
        if (value < 0 || value >= Index29::modulus) {
            return DslDiag::make(
                       DslRuleId::E040_param_domain,
                       "constant " + std::to_string(value) +
                           " is outside Index29 domain 0..28")
                .to_status();
        }
        auto node = std::shared_ptr<Z29Expr>(new Z29Expr(Kind::Const));
        node->const_value_ = static_cast<std::uint8_t>(value);
        return node;
    }

    [[nodiscard]] static Ptr var(std::string name) {
        auto node = std::shared_ptr<Z29Expr>(new Z29Expr(Kind::Var));
        node->name_ = std::move(name);
        return node;
    }

    [[nodiscard]] static Ptr add(Ptr left, Ptr right) {
        return make_bin(Kind::Add, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr sub(Ptr left, Ptr right) {
        return make_bin(Kind::Sub, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr mul(Ptr left, Ptr right) {
        return make_bin(Kind::Mul, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr div(Ptr left, Ptr right) {
        return make_bin(Kind::Div, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr floor_div(Ptr left, Ptr right) {
        return make_bin(Kind::FloorDiv, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr mod(Ptr left, Ptr right) {
        return make_bin(Kind::Mod, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr pow(Ptr left, Ptr right) {
        return make_bin(Kind::Pow, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr neg(Ptr arg) {
        return make_unary(Kind::Neg, std::move(arg));
    }
    [[nodiscard]] static Ptr inv(Ptr arg) {
        return make_unary(Kind::Inv, std::move(arg));
    }
    [[nodiscard]] static Ptr atbash(Ptr arg) {
        return make_unary(Kind::Atbash, std::move(arg));
    }
    [[nodiscard]] static Ptr bit_and(Ptr left, Ptr right) {
        return make_bin(Kind::BitAnd, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr bit_or(Ptr left, Ptr right) {
        return make_bin(Kind::BitOr, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr bit_xor(Ptr left, Ptr right) {
        return make_bin(Kind::BitXor, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr bit_not(Ptr arg) {
        return make_unary(Kind::BitNot, std::move(arg));
    }
    [[nodiscard]] static Ptr lshift(Ptr left, Ptr right) {
        return make_bin(Kind::LShift, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr rshift(Ptr left, Ptr right) {
        return make_bin(Kind::RShift, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr eq(Ptr left, Ptr right) {
        return make_bin(Kind::Eq, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr ne(Ptr left, Ptr right) {
        return make_bin(Kind::Ne, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr lt(Ptr left, Ptr right) {
        return make_bin(Kind::Lt, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr le(Ptr left, Ptr right) {
        return make_bin(Kind::Le, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr gt(Ptr left, Ptr right) {
        return make_bin(Kind::Gt, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr ge(Ptr left, Ptr right) {
        return make_bin(Kind::Ge, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr bool_and(Ptr left, Ptr right) {
        return make_bin(Kind::BoolAnd, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr bool_or(Ptr left, Ptr right) {
        return make_bin(Kind::BoolOr, std::move(left), std::move(right));
    }
    [[nodiscard]] static Ptr bool_not(Ptr arg) {
        return make_unary(Kind::BoolNot, std::move(arg));
    }

    /// `select(cond, t, f)` — nonzero `cond` yields `t`, else `f` (branch-free mux).
    /// When `prefer_branch` is true (honored `#ignore DSL_FLAG:divergent_branch`),
    /// CPU/CUDA emit may use a C++/CUDA conditional that can warp-diverge.
    [[nodiscard]] static Ptr select(
        Ptr cond,
        Ptr if_true,
        Ptr if_false,
        bool prefer_branch = false) {
        return make_select(
            std::move(cond), std::move(if_true), std::move(if_false), prefer_branch);
    }

    [[nodiscard]] static Ptr call(std::string primitive, std::vector<Ptr> args) {
        auto node = std::shared_ptr<Z29Expr>(new Z29Expr(Kind::Call));
        node->name_ = std::move(primitive);
        node->args_ = std::move(args);
        return node;
    }

    /// Rebuild a binary node of the same kind (used by optimize / substitute).
    [[nodiscard]] static Ptr make_binary(Kind kind, Ptr left, Ptr right) {
        return make_bin(kind, std::move(left), std::move(right));
    }

    [[nodiscard]] static Ptr make_unary_kind(Kind kind, Ptr arg) {
        return make_unary(kind, std::move(arg));
    }

    [[nodiscard]] static Ptr make_select(
        Ptr cond,
        Ptr if_true,
        Ptr if_false,
        bool prefer_branch = false) {
        auto node = std::shared_ptr<Z29Expr>(new Z29Expr(Kind::Select));
        node->left_ = std::move(cond);
        node->right_ = std::move(if_true);
        node->alt_ = std::move(if_false);
        node->prefer_branch_ = prefer_branch;
        return node;
    }

    [[nodiscard]] Kind kind() const noexcept {
        return kind_;
    }

    [[nodiscard]] std::uint8_t const_value() const noexcept {
        return const_value_;
    }

    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

    [[nodiscard]] const Ptr& left() const noexcept {
        return left_;
    }

    [[nodiscard]] const Ptr& right() const noexcept {
        return right_;
    }

    [[nodiscard]] const Ptr& arg() const noexcept {
        return left_;
    }

    /// Select: condition (nonzero → true arm).
    [[nodiscard]] const Ptr& cond() const noexcept {
        return left_;
    }

    /// Select: arm taken when `cond != 0`.
    [[nodiscard]] const Ptr& if_true() const noexcept {
        return right_;
    }

    /// Select: arm taken when `cond == 0`.
    [[nodiscard]] const Ptr& if_false() const noexcept {
        return alt_;
    }

    /// When true, emit may use a real conditional (ignored divergent HotLoop if).
    [[nodiscard]] bool prefer_branch() const noexcept {
        return prefer_branch_;
    }

    void set_prefer_branch(bool prefer) noexcept {
        prefer_branch_ = prefer;
    }

    [[nodiscard]] const std::vector<Ptr>& args() const noexcept {
        return args_;
    }

    [[nodiscard]] const std::string& source_path() const noexcept {
        return source_path_;
    }

    [[nodiscard]] std::optional<int> lineno() const noexcept {
        return lineno_;
    }

    [[nodiscard]] std::optional<int> col_offset() const noexcept {
        return col_offset_;
    }

    void set_location(
        std::string path,
        std::optional<int> lineno = std::nullopt,
        std::optional<int> col = std::nullopt) {
        source_path_ = std::move(path);
        lineno_ = lineno;
        col_offset_ = col;
    }

    /// Evaluate against variable bindings. Missing vars and domain errors → Status.
    [[nodiscard]] StatusOr<Index29> eval(const Env& env) const {
        switch (kind_) {
        case Kind::Const:
            return Index29{const_value_};
        case Kind::Var: {
            const auto it = env.find(name_);
            if (it == env.end()) {
                return diag_fail(
                    DslRuleId::E032_primitive_body, "unbound variable '" + name_ + "'");
            }
            return it->second;
        }
        case Kind::Call:
            return eval_call(env);
        case Kind::Select:
            return eval_select(env, /*cuda_mirror=*/false);
        default:
            break;
        }
        if (is_binary(kind_)) {
            if (!left_ || !right_) {
                return diag_fail(DslRuleId::E032_primitive_body, "binary Z29Expr missing operands");
            }
            StatusOr<Index29> l = left_->eval(env);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Index29> r = right_->eval(env);
            if (!r.ok()) {
                return r.status();
            }
            return eval_binary_host(kind_, l.value(), r.value());
        }
        if (is_unary(kind_)) {
            if (!left_) {
                return diag_fail(DslRuleId::E032_primitive_body, "unary Z29Expr missing operand");
            }
            StatusOr<Index29> a = left_->eval(env);
            if (!a.ok()) {
                return a.status();
            }
            return eval_unary_host(kind_, a.value());
        }
        return diag_fail(DslRuleId::E032_primitive_body, "unknown Z29Expr kind");
    }

    /// Host evaluation using the CUDA emit op-sequence (`Z29Device` semantics).
    [[nodiscard]] StatusOr<Index29> eval_cuda_mirror(const Env& env) const {
        switch (kind_) {
        case Kind::Const:
            return Index29{const_value_};
        case Kind::Var: {
            const auto it = env.find(name_);
            if (it == env.end()) {
                return diag_fail(
                    DslRuleId::E032_primitive_body, "unbound variable '" + name_ + "'");
            }
            return it->second;
        }
        case Kind::Call:
            return eval_call_cuda_mirror(env);
        case Kind::Select:
            return eval_select(env, /*cuda_mirror=*/true);
        default:
            break;
        }
        if (is_binary(kind_)) {
            if (!left_ || !right_) {
                return diag_fail(DslRuleId::E032_primitive_body, "binary Z29Expr missing operands");
            }
            StatusOr<Index29> l = left_->eval_cuda_mirror(env);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Index29> r = right_->eval_cuda_mirror(env);
            if (!r.ok()) {
                return r.status();
            }
            return eval_binary_device(kind_, l.value().value(), r.value().value());
        }
        if (is_unary(kind_)) {
            if (!left_) {
                return diag_fail(DslRuleId::E032_primitive_body, "unary Z29Expr missing operand");
            }
            StatusOr<Index29> a = left_->eval_cuda_mirror(env);
            if (!a.ok()) {
                return a.status();
            }
            return eval_unary_device(kind_, a.value().value());
        }
        return diag_fail(DslRuleId::E032_primitive_body, "unknown Z29Expr kind");
    }

    /// Deep-clone with variable remapping (used by `DslFuse` inlining).
    [[nodiscard]] Ptr remap(const std::unordered_map<std::string, Ptr>& mapping) const {
        switch (kind_) {
        case Kind::Const:
            return constant(const_value_).value();
        case Kind::Var: {
            const auto it = mapping.find(name_);
            if (it != mapping.end()) {
                return it->second;
            }
            return var(name_);
        }
        case Kind::Call: {
            std::vector<Ptr> mapped;
            mapped.reserve(args_.size());
            for (const Ptr& a : args_) {
                mapped.push_back(a->remap(mapping));
            }
            return call(name_, std::move(mapped));
        }
        case Kind::Select: {
            auto out = make_select(
                left_->remap(mapping),
                right_->remap(mapping),
                alt_->remap(mapping),
                prefer_branch_);
            return out;
        }
        default:
            break;
        }
        if (is_binary(kind_)) {
            return make_bin(kind_, left_->remap(mapping), right_->remap(mapping));
        }
        if (is_unary(kind_)) {
            return make_unary(kind_, left_->remap(mapping));
        }
        return constant(0).value();
    }

private:
    explicit Z29Expr(Kind kind) : kind_(kind) {}

    [[nodiscard]] StatusOr<Index29> eval_binary_host(Kind k, Index29 l, Index29 r) const {
        switch (k) {
        case Kind::Add:
            return Z29::add(l, r);
        case Kind::Sub:
            return Z29::sub(l, r);
        case Kind::Mul:
            return Z29::mul(l, r);
        case Kind::Div: {
            if (r.value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_div divisor is 0");
            }
            return Z29::mul(l, Z29::inv(r));
        }
        case Kind::FloorDiv: {
            if (r.value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_floordiv divisor is 0");
            }
            return Z29::floor_div(l, r);
        }
        case Kind::Mod: {
            if (r.value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_mod divisor is 0");
            }
            return Index29{static_cast<std::uint8_t>(l.value() % r.value())};
        }
        case Kind::Pow:
            return Z29::pow(l, r);
        case Kind::BitAnd:
            return Z29::bit_and(l, r);
        case Kind::BitOr:
            return Z29::bit_or(l, r);
        case Kind::BitXor:
            return Z29::bit_xor(l, r);
        case Kind::LShift:
            return Z29::lshift(l, r);
        case Kind::RShift:
            return Z29::rshift(l, r);
        case Kind::Eq:
            return Z29::eq(l, r);
        case Kind::Ne:
            return Z29::ne(l, r);
        case Kind::Lt:
            return Z29::lt(l, r);
        case Kind::Le:
            return Z29::le(l, r);
        case Kind::Gt:
            return Z29::gt(l, r);
        case Kind::Ge:
            return Z29::ge(l, r);
        case Kind::BoolAnd:
            return Z29::bool_and(l, r);
        case Kind::BoolOr:
            return Z29::bool_or(l, r);
        default:
            return diag_fail(DslRuleId::E032_primitive_body, "not a binary Z29Expr kind");
        }
    }

    [[nodiscard]] StatusOr<Index29> eval_unary_host(Kind k, Index29 a) const {
        switch (k) {
        case Kind::Neg:
            return Z29::neg(a);
        case Kind::Inv: {
            if (a.value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_inv(0) is undefined");
            }
            return Z29::inv(a);
        }
        case Kind::Atbash:
            return Z29::atbash(a);
        case Kind::BitNot:
            return Z29::bit_not(a);
        case Kind::BoolNot:
            return Z29::bool_not(a);
        default:
            return diag_fail(DslRuleId::E032_primitive_body, "not a unary Z29Expr kind");
        }
    }

    /// Bit-identical to `Z29Device` (Parcae/Parcae/cuda/z29_device.hpp).
    [[nodiscard]] StatusOr<Index29> eval_binary_device(
        Kind k,
        std::uint8_t l,
        std::uint8_t r) const {
        switch (k) {
        case Kind::Add:
            return Index29{device_add(l, r)};
        case Kind::Sub:
            return Index29{device_sub(l, r)};
        case Kind::Mul:
            return Index29{device_mul(l, r)};
        case Kind::Div: {
            if (r == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_div divisor is 0");
            }
            return Index29{device_mul(l, device_inv(r))};
        }
        case Kind::FloorDiv: {
            if (r == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_floordiv divisor is 0");
            }
            return Index29{static_cast<std::uint8_t>(l / r)};
        }
        case Kind::Mod: {
            if (r == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_mod divisor is 0");
            }
            return Index29{static_cast<std::uint8_t>(l % r)};
        }
        case Kind::Pow:
            return Index29{device_pow(l, r)};
        case Kind::BitAnd:
            return Index29{static_cast<std::uint8_t>((l & r) % Index29::modulus)};
        case Kind::BitOr:
            return Index29{static_cast<std::uint8_t>((l | r) % Index29::modulus)};
        case Kind::BitXor:
            return Index29{static_cast<std::uint8_t>((l ^ r) % Index29::modulus)};
        case Kind::LShift: {
            if (r >= 64u) {
                return Index29{0};
            }
            return Index29{static_cast<std::uint8_t>(
                (static_cast<unsigned long long>(l) << r) % Index29::modulus)};
        }
        case Kind::RShift: {
            if (r >= 8u) {
                return Index29{0};
            }
            return Index29{static_cast<std::uint8_t>(l >> r)};
        }
        case Kind::Eq:
            return Index29{static_cast<std::uint8_t>(l == r ? 1u : 0u)};
        case Kind::Ne:
            return Index29{static_cast<std::uint8_t>(l != r ? 1u : 0u)};
        case Kind::Lt:
            return Index29{static_cast<std::uint8_t>(l < r ? 1u : 0u)};
        case Kind::Le:
            return Index29{static_cast<std::uint8_t>(l <= r ? 1u : 0u)};
        case Kind::Gt:
            return Index29{static_cast<std::uint8_t>(l > r ? 1u : 0u)};
        case Kind::Ge:
            return Index29{static_cast<std::uint8_t>(l >= r ? 1u : 0u)};
        case Kind::BoolAnd:
            return Index29{static_cast<std::uint8_t>((l != 0 && r != 0) ? 1u : 0u)};
        case Kind::BoolOr:
            return Index29{static_cast<std::uint8_t>((l != 0 || r != 0) ? 1u : 0u)};
        default:
            return diag_fail(DslRuleId::E032_primitive_body, "not a binary Z29Expr kind");
        }
    }

    [[nodiscard]] StatusOr<Index29> eval_unary_device(Kind k, std::uint8_t a) const {
        switch (k) {
        case Kind::Neg:
            return Index29{device_neg(a)};
        case Kind::Inv: {
            if (a == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_inv(0) is undefined");
            }
            return Index29{device_inv(a)};
        }
        case Kind::Atbash:
        case Kind::BitNot:
            return Index29{device_sub(28, a)};
        case Kind::BoolNot:
            return Index29{static_cast<std::uint8_t>(a == 0 ? 1u : 0u)};
        default:
            return diag_fail(DslRuleId::E032_primitive_body, "not a unary Z29Expr kind");
        }
    }

    [[nodiscard]] static std::uint8_t device_add(std::uint8_t x, std::uint8_t y) noexcept {
        const unsigned s = static_cast<unsigned>(x) + static_cast<unsigned>(y);
        return static_cast<std::uint8_t>(s >= Index29::modulus ? s - Index29::modulus : s);
    }

    [[nodiscard]] static std::uint8_t device_neg(std::uint8_t x) noexcept {
        return x == 0 ? static_cast<std::uint8_t>(0)
                      : static_cast<std::uint8_t>(Index29::modulus - x);
    }

    [[nodiscard]] static std::uint8_t device_sub(std::uint8_t x, std::uint8_t y) noexcept {
        const unsigned s =
            static_cast<unsigned>(x) + Index29::modulus - static_cast<unsigned>(y);
        return static_cast<std::uint8_t>(s >= Index29::modulus ? s - Index29::modulus : s);
    }

    [[nodiscard]] static std::uint8_t device_mul(std::uint8_t x, std::uint8_t y) noexcept {
        return static_cast<std::uint8_t>(
            (static_cast<unsigned>(x) * static_cast<unsigned>(y)) % Index29::modulus);
    }

    [[nodiscard]] static std::uint8_t device_inv(std::uint8_t a) noexcept {
        constexpr std::uint8_t inv_table[Index29::modulus] = {
            0,  1,  15, 10, 22, 6,  5,  25, 11, 13, 3,  8,  17, 9,  27,
            2,  20, 12, 21, 26, 16, 18, 4,  24, 23, 7,  19, 14, 28};
        return inv_table[a];
    }

    [[nodiscard]] static std::uint8_t device_pow(std::uint8_t base, std::uint8_t exp) noexcept {
        std::uint8_t result = 1;
        std::uint8_t b = base;
        std::uint8_t e = exp;
        while (e != 0) {
            if ((e & 1u) != 0) {
                result = static_cast<std::uint8_t>((result * b) % Index29::modulus);
            }
            b = static_cast<std::uint8_t>((b * b) % Index29::modulus);
            e = static_cast<std::uint8_t>(e >> 1);
        }
        return result;
    }

    [[nodiscard]] static Ptr make_bin(Kind kind, Ptr left, Ptr right) {
        auto node = std::shared_ptr<Z29Expr>(new Z29Expr(kind));
        node->left_ = std::move(left);
        node->right_ = std::move(right);
        return node;
    }

    [[nodiscard]] static Ptr make_unary(Kind kind, Ptr arg) {
        auto node = std::shared_ptr<Z29Expr>(new Z29Expr(kind));
        node->left_ = std::move(arg);
        return node;
    }

    [[nodiscard]] Status diag_fail(std::string_view rule_id, std::string message) const {
        return DslDiag::make(rule_id, std::move(message), source_path_, lineno_, col_offset_)
            .to_status();
    }

    [[nodiscard]] StatusOr<Index29> eval_select(const Env& env, bool cuda_mirror) const {
        if (!left_ || !right_ || !alt_) {
            return diag_fail(DslRuleId::E032_primitive_body, "Select Z29Expr missing operands");
        }
        StatusOr<Index29> c = cuda_mirror ? left_->eval_cuda_mirror(env) : left_->eval(env);
        if (!c.ok()) {
            return c.status();
        }
        // Nonzero → true arm (matches Z29 bool-ish / compare 0|1 convention).
        if (c.value().value() != 0) {
            return cuda_mirror ? right_->eval_cuda_mirror(env) : right_->eval(env);
        }
        return cuda_mirror ? alt_->eval_cuda_mirror(env) : alt_->eval(env);
    }

    [[nodiscard]] StatusOr<Index29> eval_call(const Env& env) const {
        return eval_call_dispatch(env, /*cuda_mirror=*/false);
    }

    [[nodiscard]] StatusOr<Index29> eval_call_cuda_mirror(const Env& env) const {
        return eval_call_dispatch(env, /*cuda_mirror=*/true);
    }

    [[nodiscard]] StatusOr<Index29> eval_call_dispatch(const Env& env, bool cuda_mirror) const {
        const auto as_bin = [&](Kind k) -> StatusOr<Index29> {
            if (args_.size() != 2 || !args_[0] || !args_[1]) {
                return diag_fail(
                    DslRuleId::E032_primitive_body, "call '" + name_ + "' expects 2 arguments");
            }
            auto tmp = make_bin(k, args_[0], args_[1]);
            tmp->source_path_ = source_path_;
            tmp->lineno_ = lineno_;
            tmp->col_offset_ = col_offset_;
            return cuda_mirror ? tmp->eval_cuda_mirror(env) : tmp->eval(env);
        };
        const auto as_unary = [&](Kind k) -> StatusOr<Index29> {
            if (args_.size() != 1 || !args_[0]) {
                return diag_fail(
                    DslRuleId::E032_primitive_body, "call '" + name_ + "' expects 1 argument");
            }
            auto tmp = make_unary(k, args_[0]);
            tmp->source_path_ = source_path_;
            tmp->lineno_ = lineno_;
            tmp->col_offset_ = col_offset_;
            return cuda_mirror ? tmp->eval_cuda_mirror(env) : tmp->eval(env);
        };

        if (name_ == "z29_add") {
            return as_bin(Kind::Add);
        }
        if (name_ == "z29_sub") {
            return as_bin(Kind::Sub);
        }
        if (name_ == "z29_mul") {
            return as_bin(Kind::Mul);
        }
        if (name_ == "z29_div") {
            return as_bin(Kind::Div);
        }
        if (name_ == "z29_floordiv") {
            return as_bin(Kind::FloorDiv);
        }
        if (name_ == "z29_mod") {
            return as_bin(Kind::Mod);
        }
        if (name_ == "z29_pow") {
            return as_bin(Kind::Pow);
        }
        if (name_ == "z29_bit_and") {
            return as_bin(Kind::BitAnd);
        }
        if (name_ == "z29_bit_or") {
            return as_bin(Kind::BitOr);
        }
        if (name_ == "z29_bit_xor") {
            return as_bin(Kind::BitXor);
        }
        if (name_ == "z29_lshift") {
            return as_bin(Kind::LShift);
        }
        if (name_ == "z29_rshift") {
            return as_bin(Kind::RShift);
        }
        if (name_ == "z29_eq") {
            return as_bin(Kind::Eq);
        }
        if (name_ == "z29_ne") {
            return as_bin(Kind::Ne);
        }
        if (name_ == "z29_lt") {
            return as_bin(Kind::Lt);
        }
        if (name_ == "z29_le") {
            return as_bin(Kind::Le);
        }
        if (name_ == "z29_gt") {
            return as_bin(Kind::Gt);
        }
        if (name_ == "z29_ge") {
            return as_bin(Kind::Ge);
        }
        if (name_ == "z29_bool_and") {
            return as_bin(Kind::BoolAnd);
        }
        if (name_ == "z29_bool_or") {
            return as_bin(Kind::BoolOr);
        }
        if (name_ == "z29_inv") {
            return as_unary(Kind::Inv);
        }
        if (name_ == "z29_neg") {
            return as_unary(Kind::Neg);
        }
        if (name_ == "z29_atbash") {
            return as_unary(Kind::Atbash);
        }
        if (name_ == "z29_bit_not") {
            return as_unary(Kind::BitNot);
        }
        if (name_ == "z29_bool_not") {
            return as_unary(Kind::BoolNot);
        }
        if (name_ == "z29_select") {
            if (args_.size() != 3 || !args_[0] || !args_[1] || !args_[2]) {
                return diag_fail(
                    DslRuleId::E032_primitive_body, "call 'z29_select' expects 3 arguments");
            }
            auto tmp = make_select(args_[0], args_[1], args_[2]);
            tmp->source_path_ = source_path_;
            tmp->lineno_ = lineno_;
            tmp->col_offset_ = col_offset_;
            return cuda_mirror ? tmp->eval_cuda_mirror(env) : tmp->eval(env);
        }
        return diag_fail(
            DslRuleId::E032_primitive_body,
            "unknown primitive call '" + name_ + "' (not a builtin; registry comes later)");
    }

    Kind kind_ = Kind::Const;
    std::uint8_t const_value_ = 0;
    std::string name_;
    Ptr left_;
    Ptr right_;
    Ptr alt_;  // Select false-arm only
    bool prefer_branch_ = false;
    std::vector<Ptr> args_;
    std::string source_path_;
    std::optional<int> lineno_;
    std::optional<int> col_offset_;
};

#endif // Z29_EXPR_HPP
