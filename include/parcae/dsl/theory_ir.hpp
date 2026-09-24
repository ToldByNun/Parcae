#ifndef THEORY_IR_HPP
#define THEORY_IR_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Compiled `@Theory` record (docs/spec/dsl.md). Does not lower AST yet.
class TheoryIr {
public:
    enum class Family : std::uint8_t {
        Elementwise = 0,
        KeyedStream,
        KeyedPermutation,
        Compose,
    };

    enum class Tier : std::uint8_t {
        A = 0,
        B,
        C,
    };

    enum class InterruptMode : std::uint8_t {
        /// Family elementwise may omit interrupt_policy.
        ElementwiseDefault = 0,
        /// Explicit `@Theory(interrupts="none_by_design")`.
        NoneByDesign,
        /// `interrupt_policy` method present (predicate IR later).
        PolicyMethod,
    };

    [[nodiscard]] static StatusOr<Family> parse_family(std::string_view text) {
        if (text == "elementwise") {
            return Family::Elementwise;
        }
        if (text == "keyed_stream") {
            return Family::KeyedStream;
        }
        if (text == "keyed_permutation") {
            return Family::KeyedPermutation;
        }
        if (text == "compose") {
            return Family::Compose;
        }
        return Status::error("unknown theory family '" + std::string(text) + "'");
    }

    [[nodiscard]] static StatusOr<Tier> parse_tier(std::string_view text) {
        if (text == "A") {
            return Tier::A;
        }
        if (text == "B") {
            return Tier::B;
        }
        if (text == "C") {
            return Tier::C;
        }
        return Status::error("unknown theory tier '" + std::string(text) + "'");
    }

    [[nodiscard]] static const char* family_str(Family f) noexcept {
        switch (f) {
        case Family::Elementwise:
            return "elementwise";
        case Family::KeyedStream:
            return "keyed_stream";
        case Family::KeyedPermutation:
            return "keyed_permutation";
        case Family::Compose:
            return "compose";
        }
        return "unknown";
    }

    [[nodiscard]] static const char* tier_str(Tier t) noexcept {
        switch (t) {
        case Tier::A:
            return "A";
        case Tier::B:
            return "B";
        case Tier::C:
            return "C";
        }
        return "?";
    }

    [[nodiscard]] static const char* interrupt_mode_str(InterruptMode m) noexcept {
        switch (m) {
        case InterruptMode::ElementwiseDefault:
            return "elementwise_default";
        case InterruptMode::NoneByDesign:
            return "none_by_design";
        case InterruptMode::PolicyMethod:
            return "policy_method";
        }
        return "unknown";
    }

    [[nodiscard]] static StatusOr<TheoryIr>
    make(std::string name, Family family, Tier tier, InterruptMode interrupts,
         std::vector<ParamIr> params, Z29Expr::Ptr encrypt_step = {},
         Z29Expr::Ptr decrypt_step = {}, std::optional<std::string> structural_claim = std::nullopt,
         std::string source_path = {}, std::optional<int> lineno = std::nullopt,
         std::optional<int> col = std::nullopt) {
        if (name.empty()) {
            return DslDiag::make(DslRuleId::E032_primitive_body, "theory name must be non-empty",
                                 std::move(source_path), lineno, col)
                .to_status();
        }
        TheoryIr ir{
            std::move(name),
            family,
            tier,
            interrupts,
            std::move(params),
            std::move(encrypt_step),
            std::move(decrypt_step),
            std::move(structural_claim),
            std::move(source_path),
            lineno,
            col,
        };
        Status st = ir.validate();
        if (!st.ok()) {
            return st;
        }
        return ir;
    }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    [[nodiscard]] Family family() const noexcept { return family_; }

    [[nodiscard]] Tier tier() const noexcept { return tier_; }

    [[nodiscard]] InterruptMode interrupt_mode() const noexcept { return interrupt_mode_; }

    [[nodiscard]] const std::vector<ParamIr>& params() const noexcept { return params_; }

    [[nodiscard]] const Z29Expr::Ptr& encrypt_step() const noexcept { return encrypt_step_; }

    [[nodiscard]] const Z29Expr::Ptr& decrypt_step() const noexcept { return decrypt_step_; }

    [[nodiscard]] const std::optional<std::string>& structural_claim() const noexcept {
        return structural_claim_;
    }

    [[nodiscard]] Status validate() const {
        Status st = validate_tier_claim();
        if (!st.ok()) {
            return st;
        }
        st = validate_interrupt();
        if (!st.ok()) {
            return st;
        }
        return validate_params();
    }

    [[nodiscard]] Status validate_tier_claim() const {
        if (tier_ == Tier::A) {
            return Status::success();
        }
        if (!structural_claim_.has_value() || structural_claim_->empty()) {
            return DslDiag::make(DslRuleId::E013_tier_structural_claim,
                                 std::string("tier ") + tier_str(tier_) +
                                     " requires structural_claim()",
                                 source_path_, lineno_, col_)
                .to_status();
        }
        return Status::success();
    }

    [[nodiscard]] Status validate_interrupt() const {
        if (interrupt_mode_ == InterruptMode::NoneByDesign ||
            interrupt_mode_ == InterruptMode::PolicyMethod) {
            return Status::success();
        }
        // ElementwiseDefault
        if (family_ == Family::Elementwise) {
            return Status::success();
        }
        return DslDiag::make(DslRuleId::E030_interrupt_policy,
                             std::string("interrupt_policy missing; family=") +
                                 family_str(family_) +
                                 "; set interrupts='none_by_design' if intentional",
                             source_path_, lineno_, col_)
            .to_status();
    }

    [[nodiscard]] Status validate_params() const {
        for (const ParamIr& p : params_) {
            // ParamIr::make already gated bounds; re-check name uniqueness.
            (void)p;
        }
        for (std::size_t i = 0; i < params_.size(); ++i) {
            for (std::size_t j = i + 1; j < params_.size(); ++j) {
                if (params_[i].name() == params_[j].name()) {
                    return DslDiag::make(DslRuleId::E040_param_domain,
                                         "duplicate param name '" + params_[i].name() + "'",
                                         source_path_, lineno_, col_)
                        .to_status();
                }
            }
        }
        return Status::success();
    }

private:
    TheoryIr(std::string name, Family family, Tier tier, InterruptMode interrupts,
             std::vector<ParamIr> params, Z29Expr::Ptr encrypt_step, Z29Expr::Ptr decrypt_step,
             std::optional<std::string> structural_claim, std::string source_path,
             std::optional<int> lineno, std::optional<int> col)
        : name_(std::move(name)), family_(family), tier_(tier), interrupt_mode_(interrupts),
          params_(std::move(params)), encrypt_step_(std::move(encrypt_step)),
          decrypt_step_(std::move(decrypt_step)), structural_claim_(std::move(structural_claim)),
          source_path_(std::move(source_path)), lineno_(lineno), col_(col) {}

    std::string name_;
    Family family_ = Family::Elementwise;
    Tier tier_ = Tier::A;
    InterruptMode interrupt_mode_ = InterruptMode::ElementwiseDefault;
    std::vector<ParamIr> params_;
    Z29Expr::Ptr encrypt_step_;
    Z29Expr::Ptr decrypt_step_;
    std::optional<std::string> structural_claim_;
    std::string source_path_;
    std::optional<int> lineno_;
    std::optional<int> col_;
};

#endif // THEORY_IR_HPP
