#ifndef COMPOSE_IR_HPP
#define COMPOSE_IR_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/// Compiled `@ComposedTheory` record: ordered steps + step_params bindings.
/// Fusion / staged fallback is `DslFuse` (later).
class ComposeIr {
public:
    /// One binding from `step_params()`: stage id → param name → value ref
    /// (theory field name for now; literal/expr lowering comes with DslBuildIr).
    class StepParamBinding {
    public:
        StepParamBinding(std::string step_id, std::string param_name, std::string value_ref)
            : step_id_(std::move(step_id)),
              param_name_(std::move(param_name)),
              value_ref_(std::move(value_ref)) {}

        [[nodiscard]] const std::string& step_id() const noexcept {
            return step_id_;
        }

        [[nodiscard]] const std::string& param_name() const noexcept {
            return param_name_;
        }

        [[nodiscard]] const std::string& value_ref() const noexcept {
            return value_ref_;
        }

    private:
        std::string step_id_;
        std::string param_name_;
        std::string value_ref_;
    };

    [[nodiscard]] static StatusOr<ComposeIr> make(
        std::string name,
        TheoryIr::Tier tier,
        std::vector<std::string> steps,
        std::vector<ParamIr> params = {},
        std::vector<StepParamBinding> step_params = {},
        std::optional<std::string> structural_claim = std::nullopt,
        std::string source_path = {},
        std::optional<int> lineno = std::nullopt,
        std::optional<int> col = std::nullopt) {
        if (name.empty()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "composed theory name must be non-empty",
                       std::move(source_path),
                       lineno,
                       col)
                .to_status();
        }
        if (steps.empty()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "composed theory '" + name + "' steps must be non-empty",
                       std::move(source_path),
                       lineno,
                       col)
                .to_status();
        }
        for (const std::string& step : steps) {
            if (step.empty()) {
                return DslDiag::make(
                           DslRuleId::E032_primitive_body,
                           "composed theory step id must be non-empty",
                           std::move(source_path),
                           lineno,
                           col)
                    .to_status();
            }
        }
        ComposeIr ir{
            std::move(name),
            tier,
            std::move(steps),
            std::move(params),
            std::move(step_params),
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

    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

    [[nodiscard]] TheoryIr::Tier tier() const noexcept {
        return tier_;
    }

    [[nodiscard]] const std::vector<std::string>& steps() const noexcept {
        return steps_;
    }

    [[nodiscard]] const std::vector<ParamIr>& params() const noexcept {
        return params_;
    }

    [[nodiscard]] const std::vector<StepParamBinding>& step_params() const noexcept {
        return step_params_;
    }

    [[nodiscard]] const std::optional<std::string>& structural_claim() const noexcept {
        return structural_claim_;
    }

    [[nodiscard]] Status validate() const {
        Status st = validate_tier_claim();
        if (!st.ok()) {
            return st;
        }
        return validate_step_params();
    }

    [[nodiscard]] Status validate_tier_claim() const {
        if (tier_ == TheoryIr::Tier::A) {
            return Status::success();
        }
        if (!structural_claim_.has_value() || structural_claim_->empty()) {
            return DslDiag::make(
                       DslRuleId::E013_tier_structural_claim,
                       std::string("tier ") + TheoryIr::tier_str(tier_) +
                           " requires structural_claim()",
                       source_path_,
                       lineno_,
                       col_)
                .to_status();
        }
        return Status::success();
    }

    [[nodiscard]] Status validate_step_params() const {
        for (const StepParamBinding& b : step_params_) {
            bool step_known = false;
            for (const std::string& s : steps_) {
                if (s == b.step_id()) {
                    step_known = true;
                    break;
                }
            }
            if (!step_known) {
                return DslDiag::make(
                           DslRuleId::E032_primitive_body,
                           "step_params references unknown step '" + b.step_id() + "'",
                           source_path_,
                           lineno_,
                           col_)
                    .to_status();
            }
            if (b.param_name().empty() || b.value_ref().empty()) {
                return DslDiag::make(
                           DslRuleId::E032_primitive_body,
                           "step_params entry missing param_name or value_ref",
                           source_path_,
                           lineno_,
                           col_)
                    .to_status();
            }
        }
        for (std::size_t i = 0; i < params_.size(); ++i) {
            for (std::size_t j = i + 1; j < params_.size(); ++j) {
                if (params_[i].name() == params_[j].name()) {
                    return DslDiag::make(
                               DslRuleId::E040_param_domain,
                               "duplicate param name '" + params_[i].name() + "'",
                               source_path_,
                               lineno_,
                               col_)
                        .to_status();
                }
            }
        }
        return Status::success();
    }

private:
    ComposeIr(
        std::string name,
        TheoryIr::Tier tier,
        std::vector<std::string> steps,
        std::vector<ParamIr> params,
        std::vector<StepParamBinding> step_params,
        std::optional<std::string> structural_claim,
        std::string source_path,
        std::optional<int> lineno,
        std::optional<int> col)
        : name_(std::move(name)),
          tier_(tier),
          steps_(std::move(steps)),
          params_(std::move(params)),
          step_params_(std::move(step_params)),
          structural_claim_(std::move(structural_claim)),
          source_path_(std::move(source_path)),
          lineno_(lineno),
          col_(col) {}

    std::string name_;
    TheoryIr::Tier tier_ = TheoryIr::Tier::A;
    std::vector<std::string> steps_;
    std::vector<ParamIr> params_;
    std::vector<StepParamBinding> step_params_;
    std::optional<std::string> structural_claim_;
    std::string source_path_;
    std::optional<int> lineno_;
    std::optional<int> col_;
};

#endif // COMPOSE_IR_HPP
