#ifndef DSL_VERIFIER_HPP
#define DSL_VERIFIER_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/primitive_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

/// Compile-time verification gates for DSL IR (docs/spec/dsl.md).
/// E20: exhaustive enumeration for arity ≤ 4 (totality + determinism).
/// Fuzz mode and full CPU↔CUDA mirror land in later commits.
class DslVerifier {
public:
    static constexpr std::size_t max_exhaustive_arity = 4;
    static constexpr std::uint8_t modulus = Index29::modulus;
    /// 29^4 — documented ceiling for exhaustive mode.
    static constexpr std::size_t max_exhaustive_samples = 29u * 29u * 29u * 29u; // 707281

    enum class Mode : std::uint8_t {
        Exhaustive = 0,
        Fuzz, // reserved (E21)
    };

    class Report {
    public:
        Report() = default;

        [[nodiscard]] Mode mode() const noexcept {
            return mode_;
        }

        [[nodiscard]] bool passed() const noexcept {
            return passed_;
        }

        [[nodiscard]] std::size_t samples_checked() const noexcept {
            return samples_checked_;
        }

        [[nodiscard]] const std::string& primitive_name() const noexcept {
            return primitive_name_;
        }

        [[nodiscard]] const std::string& detail() const noexcept {
            return detail_;
        }

    private:
        friend class DslVerifier;

        Mode mode_ = Mode::Exhaustive;
        bool passed_ = false;
        std::size_t samples_checked_ = 0;
        std::string primitive_name_;
        std::string detail_;
    };

    /// Exhaustive gate over `0..28^arity`. Arity 0 checks a single empty eval.
    /// Arity > 4 → E050 (fuzz required; not implemented in this commit).
    [[nodiscard]] static StatusOr<Report> verify_primitive_exhaustive(const PrimitiveIr& primitive) {
        Status st = primitive.validate();
        if (!st.ok()) {
            return st;
        }

        const std::size_t arity = primitive.arity();
        if (arity > max_exhaustive_arity) {
            return fail_verify(
                primitive,
                "arity " + std::to_string(arity) + " > " +
                    std::to_string(max_exhaustive_arity) +
                    "; exhaustive mode unsupported (use fuzz seed 0xC1CADA)");
        }

        Report report;
        report.mode_ = Mode::Exhaustive;
        report.primitive_name_ = primitive.name();

        std::array<Index29, max_exhaustive_arity> args{};
        std::array<std::uint8_t, max_exhaustive_arity> digits{};
        digits.fill(0);

        const std::size_t total = pow29(arity);
        for (std::size_t sample = 0; sample < total; ++sample) {
            for (std::size_t i = 0; i < arity; ++i) {
                args[i] = Index29{digits[i]};
            }

            const std::span<const Index29> arg_span(args.data(), arity);
            StatusOr<Index29> first = primitive.eval(arg_span);
            if (!first.ok()) {
                return fail_verify(
                    primitive,
                    "totality failed at " + format_args(primitive, arg_span) + ": " +
                        first.status().message());
            }
            // Index29 construction already enforces 0..28; re-check value for the gate.
            if (first.value().value() >= modulus) {
                return fail_verify(
                    primitive,
                    "totality failed: result out of Index29 domain at " +
                        format_args(primitive, arg_span));
            }

            StatusOr<Index29> second = primitive.eval(arg_span);
            if (!second.ok()) {
                return fail_verify(
                    primitive,
                    "determinism failed (second eval error) at " +
                        format_args(primitive, arg_span) + ": " + second.status().message());
            }
            if (first.value() != second.value()) {
                return fail_verify(
                    primitive,
                    "determinism failed at " + format_args(primitive, arg_span) +
                        ": first=" + std::to_string(first.value().value()) +
                        " second=" + std::to_string(second.value().value()));
            }

            // CPU self-mirror: re-eval body env independently (CUDA mirror in E22).
            Z29Expr::Env env;
            for (std::size_t i = 0; i < arity; ++i) {
                env.emplace(primitive.param_names()[i], args[i]);
            }
            StatusOr<Index29> mirror = primitive.body()->eval(env);
            if (!mirror.ok() || mirror.value() != first.value()) {
                return fail_verify(
                    primitive,
                    "cpu mirror failed at " + format_args(primitive, arg_span));
            }

            ++report.samples_checked_;
            if (!increment_digits(digits, arity)) {
                break;
            }
        }

        if (report.samples_checked_ != total) {
            return fail_verify(
                primitive,
                "internal exhaustive sample count mismatch (got " +
                    std::to_string(report.samples_checked_) + ", expected " +
                    std::to_string(total) + ")");
        }

        report.passed_ = true;
        report.detail_ = "exhaustive ok; samples=" + std::to_string(report.samples_checked_);
        return report;
    }

private:
    DslVerifier() = delete;

    [[nodiscard]] static std::size_t pow29(std::size_t arity) noexcept {
        std::size_t n = 1;
        for (std::size_t i = 0; i < arity; ++i) {
            n *= static_cast<std::size_t>(modulus);
        }
        return n;
    }

    /// Returns false when the counter wraps past the last sample.
    [[nodiscard]] static bool increment_digits(
        std::array<std::uint8_t, max_exhaustive_arity>& digits,
        std::size_t arity) noexcept {
        if (arity == 0) {
            return false;
        }
        for (std::size_t i = 0; i < arity; ++i) {
            ++digits[i];
            if (digits[i] < modulus) {
                return true;
            }
            digits[i] = 0;
        }
        return false;
    }

    [[nodiscard]] static std::string format_args(
        const PrimitiveIr& primitive,
        std::span<const Index29> args) {
        std::ostringstream oss;
        oss << primitive.name() << "(";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            if (i < primitive.param_names().size()) {
                oss << primitive.param_names()[i] << "=";
            }
            oss << static_cast<int>(args[i].value());
        }
        oss << ")";
        return oss.str();
    }

    [[nodiscard]] static Status fail_verify(const PrimitiveIr& primitive, std::string message) {
        return DslDiag::make(
                   DslRuleId::E050_verify_failed,
                   "primitive '" + primitive.name() + "': " + std::move(message),
                   /*path=*/{},
                   std::nullopt,
                   std::nullopt)
            .to_status();
    }
};

#endif // DSL_VERIFIER_HPP
