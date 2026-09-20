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
        Add,
        Sub,
        Mul,
        Neg,
        Inv,
        Mod,
        Atbash,
        Call,
    };

    using Ptr = std::shared_ptr<Z29Expr>;
    using Env = std::unordered_map<std::string, Index29>;

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

    [[nodiscard]] static Ptr neg(Ptr arg) {
        return make_unary(Kind::Neg, std::move(arg));
    }

    [[nodiscard]] static Ptr inv(Ptr arg) {
        return make_unary(Kind::Inv, std::move(arg));
    }

    [[nodiscard]] static Ptr mod(Ptr left, Ptr right) {
        return make_bin(Kind::Mod, std::move(left), std::move(right));
    }

    [[nodiscard]] static Ptr atbash(Ptr arg) {
        return make_unary(Kind::Atbash, std::move(arg));
    }

    [[nodiscard]] static Ptr call(std::string primitive, std::vector<Ptr> args) {
        auto node = std::shared_ptr<Z29Expr>(new Z29Expr(Kind::Call));
        node->name_ = std::move(primitive);
        node->args_ = std::move(args);
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
        case Kind::Add:
            return eval_bin(env, Z29::add);
        case Kind::Sub:
            return eval_bin(env, Z29::sub);
        case Kind::Mul:
            return eval_bin(env, Z29::mul);
        case Kind::Neg: {
            StatusOr<Index29> a = require_unary(env);
            if (!a.ok()) {
                return a.status();
            }
            return Z29::neg(a.value());
        }
        case Kind::Inv: {
            StatusOr<Index29> a = require_unary(env);
            if (!a.ok()) {
                return a.status();
            }
            if (a.value().value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_inv(0) is undefined");
            }
            return Z29::inv(a.value());
        }
        case Kind::Mod: {
            if (!left_ || !right_) {
                return diag_fail(DslRuleId::E032_primitive_body, "Mod missing operands");
            }
            StatusOr<Index29> l = left_->eval(env);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Index29> r = right_->eval(env);
            if (!r.ok()) {
                return r.status();
            }
            if (r.value().value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_mod divisor is 0");
            }
            return Index29{static_cast<std::uint8_t>(l.value().value() % r.value().value())};
        }
        case Kind::Atbash: {
            StatusOr<Index29> a = require_unary(env);
            if (!a.ok()) {
                return a.status();
            }
            return Z29::atbash(a.value());
        }
        case Kind::Call:
            return eval_call(env);
        }
        return diag_fail(DslRuleId::E032_primitive_body, "unknown Z29Expr kind");
    }

    /// Host evaluation using the CUDA emit op-sequence (`Z29Device` semantics).
    /// Matches `DslEmitCuda` (add/sub/mul/neg/inv; atbash → `sub(28,x)`; mod → `%`).
    /// No device launch — used by `DslVerifier` CPU↔CUDA mirror gate (docs/spec/dsl.md).
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
        case Kind::Add: {
            StatusOr<Index29> l = require_bin_left_mirror(env);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Index29> r = require_bin_right_mirror(env);
            if (!r.ok()) {
                return r.status();
            }
            return Index29{device_add(l.value().value(), r.value().value())};
        }
        case Kind::Sub: {
            StatusOr<Index29> l = require_bin_left_mirror(env);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Index29> r = require_bin_right_mirror(env);
            if (!r.ok()) {
                return r.status();
            }
            return Index29{device_sub(l.value().value(), r.value().value())};
        }
        case Kind::Mul: {
            StatusOr<Index29> l = require_bin_left_mirror(env);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Index29> r = require_bin_right_mirror(env);
            if (!r.ok()) {
                return r.status();
            }
            return Index29{device_mul(l.value().value(), r.value().value())};
        }
        case Kind::Neg: {
            StatusOr<Index29> a = require_unary_mirror(env);
            if (!a.ok()) {
                return a.status();
            }
            return Index29{device_neg(a.value().value())};
        }
        case Kind::Inv: {
            StatusOr<Index29> a = require_unary_mirror(env);
            if (!a.ok()) {
                return a.status();
            }
            if (a.value().value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_inv(0) is undefined");
            }
            return Index29{device_inv(a.value().value())};
        }
        case Kind::Mod: {
            if (!left_ || !right_) {
                return diag_fail(DslRuleId::E032_primitive_body, "Mod missing operands");
            }
            StatusOr<Index29> l = left_->eval_cuda_mirror(env);
            if (!l.ok()) {
                return l.status();
            }
            StatusOr<Index29> r = right_->eval_cuda_mirror(env);
            if (!r.ok()) {
                return r.status();
            }
            if (r.value().value() == 0) {
                return diag_fail(DslRuleId::E040_param_domain, "z29_mod divisor is 0");
            }
            // Same as DslEmitCuda: `static_cast<uint8_t>((l % r))`.
            return Index29{static_cast<std::uint8_t>(l.value().value() % r.value().value())};
        }
        case Kind::Atbash: {
            StatusOr<Index29> a = require_unary_mirror(env);
            if (!a.ok()) {
                return a.status();
            }
            // Z29Device has no atbash — emit uses `Z29Device::sub(28, x)`.
            return Index29{device_sub(28, a.value().value())};
        }
        case Kind::Call:
            return eval_call_cuda_mirror(env);
        }
        return diag_fail(DslRuleId::E032_primitive_body, "unknown Z29Expr kind");
    }

    /// Deep-clone with variable remapping: each `Var` whose name is a key in
    /// `mapping` is replaced by that expression (used by `DslFuse` inlining).
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
        case Kind::Add:
            return add(left_->remap(mapping), right_->remap(mapping));
        case Kind::Sub:
            return sub(left_->remap(mapping), right_->remap(mapping));
        case Kind::Mul:
            return mul(left_->remap(mapping), right_->remap(mapping));
        case Kind::Mod:
            return mod(left_->remap(mapping), right_->remap(mapping));
        case Kind::Neg:
            return neg(left_->remap(mapping));
        case Kind::Inv:
            return inv(left_->remap(mapping));
        case Kind::Atbash:
            return atbash(left_->remap(mapping));
        case Kind::Call: {
            std::vector<Ptr> mapped;
            mapped.reserve(args_.size());
            for (const Ptr& a : args_) {
                mapped.push_back(a->remap(mapping));
            }
            return call(name_, std::move(mapped));
        }
        }
        // Unreachable if Kind is exhaustive; keep a safe leaf.
        return constant(0).value();
    }

private:
    explicit Z29Expr(Kind kind) : kind_(kind) {}

    /// Bit-identical to `Z29Device` host/device ops (Parcae/Parcae/cuda/z29_device.hpp).
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
        // Must match Z29Device::inv / Z29::inv_table for 1..28.
        constexpr std::uint8_t inv_table[Index29::modulus] = {
            0,  1,  15, 10, 22, 6,  5,  25, 11, 13, 3,  8,  17, 9,  27,
            2,  20, 12, 21, 26, 16, 18, 4,  24, 23, 7,  19, 14, 28};
        return inv_table[a];
    }

    [[nodiscard]] StatusOr<Index29> require_bin_left_mirror(const Env& env) const {
        if (!left_ || !right_) {
            return diag_fail(DslRuleId::E032_primitive_body, "binary Z29Expr missing operands");
        }
        return left_->eval_cuda_mirror(env);
    }

    [[nodiscard]] StatusOr<Index29> require_bin_right_mirror(const Env& env) const {
        return right_->eval_cuda_mirror(env);
    }

    [[nodiscard]] StatusOr<Index29> require_unary_mirror(const Env& env) const {
        if (!left_) {
            return diag_fail(DslRuleId::E032_primitive_body, "unary Z29Expr missing operand");
        }
        return left_->eval_cuda_mirror(env);
    }

    [[nodiscard]] StatusOr<Index29> eval_call_cuda_mirror(const Env& env) const {
        if (name_ == "z29_add") {
            return eval_call_as_bin_cuda_mirror(env, Kind::Add);
        }
        if (name_ == "z29_sub") {
            return eval_call_as_bin_cuda_mirror(env, Kind::Sub);
        }
        if (name_ == "z29_mul") {
            return eval_call_as_bin_cuda_mirror(env, Kind::Mul);
        }
        if (name_ == "z29_mod") {
            return eval_call_as_bin_cuda_mirror(env, Kind::Mod);
        }
        if (name_ == "z29_inv") {
            return eval_call_as_unary_cuda_mirror(env, Kind::Inv);
        }
        if (name_ == "z29_neg") {
            return eval_call_as_unary_cuda_mirror(env, Kind::Neg);
        }
        if (name_ == "z29_atbash") {
            return eval_call_as_unary_cuda_mirror(env, Kind::Atbash);
        }
        return diag_fail(
            DslRuleId::E032_primitive_body,
            "unknown primitive call '" + name_ + "' (not a builtin; registry comes later)");
    }

    [[nodiscard]] StatusOr<Index29> eval_call_as_bin_cuda_mirror(
        const Env& env, Kind as_kind) const {
        if (args_.size() != 2 || !args_[0] || !args_[1]) {
            return diag_fail(
                DslRuleId::E032_primitive_body, "call '" + name_ + "' expects 2 arguments");
        }
        auto tmp = make_bin(as_kind, args_[0], args_[1]);
        tmp->source_path_ = source_path_;
        tmp->lineno_ = lineno_;
        tmp->col_offset_ = col_offset_;
        return tmp->eval_cuda_mirror(env);
    }

    [[nodiscard]] StatusOr<Index29> eval_call_as_unary_cuda_mirror(
        const Env& env, Kind as_kind) const {
        if (args_.size() != 1 || !args_[0]) {
            return diag_fail(
                DslRuleId::E032_primitive_body, "call '" + name_ + "' expects 1 argument");
        }
        auto tmp = make_unary(as_kind, args_[0]);
        tmp->source_path_ = source_path_;
        tmp->lineno_ = lineno_;
        tmp->col_offset_ = col_offset_;
        return tmp->eval_cuda_mirror(env);
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

    template <typename Op>
    [[nodiscard]] StatusOr<Index29> eval_bin(const Env& env, Op op) const {
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
        return op(l.value(), r.value());
    }

    [[nodiscard]] StatusOr<Index29> require_unary(const Env& env) const {
        if (!left_) {
            return diag_fail(DslRuleId::E032_primitive_body, "unary Z29Expr missing operand");
        }
        return left_->eval(env);
    }

    [[nodiscard]] StatusOr<Index29> eval_call(const Env& env) const {
        if (name_ == "z29_add") {
            return eval_call_as_bin(env, Kind::Add);
        }
        if (name_ == "z29_sub") {
            return eval_call_as_bin(env, Kind::Sub);
        }
        if (name_ == "z29_mul") {
            return eval_call_as_bin(env, Kind::Mul);
        }
        if (name_ == "z29_mod") {
            return eval_call_as_bin(env, Kind::Mod);
        }
        if (name_ == "z29_inv") {
            return eval_call_as_unary(env, Kind::Inv);
        }
        if (name_ == "z29_neg") {
            return eval_call_as_unary(env, Kind::Neg);
        }
        if (name_ == "z29_atbash") {
            return eval_call_as_unary(env, Kind::Atbash);
        }
        return diag_fail(
            DslRuleId::E032_primitive_body,
            "unknown primitive call '" + name_ + "' (not a builtin; registry comes later)");
    }

    [[nodiscard]] StatusOr<Index29> eval_call_as_bin(const Env& env, Kind as_kind) const {
        if (args_.size() != 2 || !args_[0] || !args_[1]) {
            return diag_fail(
                DslRuleId::E032_primitive_body, "call '" + name_ + "' expects 2 arguments");
        }
        auto tmp = make_bin(as_kind, args_[0], args_[1]);
        tmp->source_path_ = source_path_;
        tmp->lineno_ = lineno_;
        tmp->col_offset_ = col_offset_;
        return tmp->eval(env);
    }

    [[nodiscard]] StatusOr<Index29> eval_call_as_unary(const Env& env, Kind as_kind) const {
        if (args_.size() != 1 || !args_[0]) {
            return diag_fail(
                DslRuleId::E032_primitive_body, "call '" + name_ + "' expects 1 argument");
        }
        auto tmp = make_unary(as_kind, args_[0]);
        tmp->source_path_ = source_path_;
        tmp->lineno_ = lineno_;
        tmp->col_offset_ = col_offset_;
        return tmp->eval(env);
    }

    Kind kind_;
    std::uint8_t const_value_ = 0;
    std::string name_;
    Ptr left_;
    Ptr right_;
    std::vector<Ptr> args_;
    std::string source_path_;
    std::optional<int> lineno_;
    std::optional<int> col_offset_;
};

#endif // Z29_EXPR_HPP
