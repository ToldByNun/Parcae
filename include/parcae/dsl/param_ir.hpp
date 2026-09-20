#ifndef PARAM_IR_HPP
#define PARAM_IR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

/// Theory / compose parameter domain (`Param[int] = Param(min=…, max=…)`).
/// Bounds MUST lie in `0..28` with `min <= max` (docs/spec/dsl.md).
class ParamIr {
public:
    [[nodiscard]] static StatusOr<ParamIr> make(
        std::string name,
        std::int64_t min,
        std::int64_t max,
        std::string source_path = {},
        std::optional<int> lineno = std::nullopt,
        std::optional<int> col = std::nullopt) {
        if (name.empty()) {
            return fail(
                "param name must be non-empty",
                std::move(source_path),
                lineno,
                col);
        }
        if (min < 0 || max < 0 || min >= Index29::modulus || max >= Index29::modulus) {
            return fail(
                "param '" + name + "' bounds must lie in 0..28 (got min=" +
                    std::to_string(min) + " max=" + std::to_string(max) + ")",
                std::move(source_path),
                lineno,
                col);
        }
        if (min > max) {
            return fail(
                "param '" + name + "' has min > max",
                std::move(source_path),
                lineno,
                col);
        }
        return ParamIr{
            std::move(name),
            static_cast<std::uint8_t>(min),
            static_cast<std::uint8_t>(max),
            std::move(source_path),
            lineno,
            col,
        };
    }

    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

    [[nodiscard]] std::uint8_t min() const noexcept {
        return min_;
    }

    [[nodiscard]] std::uint8_t max() const noexcept {
        return max_;
    }

    [[nodiscard]] std::size_t cardinality() const noexcept {
        return static_cast<std::size_t>(max_ - min_) + 1;
    }

    [[nodiscard]] bool contains(Index29 value) const noexcept {
        return value.value() >= min_ && value.value() <= max_;
    }

    [[nodiscard]] Status check_value(Index29 value) const {
        if (!contains(value)) {
            return DslDiag::make(
                       DslRuleId::E040_param_domain,
                       "value " + std::to_string(value.value()) + " outside param '" + name_ +
                           "' domain [" + std::to_string(min_) + ".." + std::to_string(max_) +
                           "]",
                       source_path_,
                       lineno_,
                       col_)
                .to_status();
        }
        return Status::success();
    }

private:
    ParamIr(
        std::string name,
        std::uint8_t min,
        std::uint8_t max,
        std::string source_path,
        std::optional<int> lineno,
        std::optional<int> col)
        : name_(std::move(name)),
          min_(min),
          max_(max),
          source_path_(std::move(source_path)),
          lineno_(lineno),
          col_(col) {}

    [[nodiscard]] static Status fail(
        std::string message,
        std::string path,
        std::optional<int> lineno,
        std::optional<int> col) {
        return DslDiag::make(
                   DslRuleId::E040_param_domain,
                   std::move(message),
                   std::move(path),
                   lineno,
                   col)
            .to_status();
    }

    std::string name_;
    std::uint8_t min_ = 0;
    std::uint8_t max_ = 0;
    std::string source_path_;
    std::optional<int> lineno_;
    std::optional<int> col_;
};

#endif // PARAM_IR_HPP
