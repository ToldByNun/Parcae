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
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Compile-time verification gates for DSL IR (docs/spec/dsl.md).
/// Exhaustive for arity ≤ 4; seeded fuzz (`0xC1CADA`) for larger arity.
/// Each sample: totality + determinism + CPU↔CUDA op-sequence mirror.
class DslVerifier {
public:
    static constexpr std::size_t max_exhaustive_arity = 4;
    static constexpr std::uint8_t modulus = Index29::modulus;
    /// 29^4 — documented ceiling for exhaustive mode.
    static constexpr std::size_t max_exhaustive_samples = 29u * 29u * 29u * 29u; // 707281

    /// Normative v0 fuzz seed (docs/spec/dsl.md, theory-artifact.md).
    static constexpr std::uint32_t default_fuzz_seed = 0xC1CADAu;
    static constexpr std::size_t default_fuzz_samples = 8192;
    /// Safety cap on arity for fuzz (args buffer).
    static constexpr std::size_t max_fuzz_arity = 32;

    enum class Mode : std::uint8_t {
        Exhaustive = 0,
        Fuzz,
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

        [[nodiscard]] std::optional<std::uint32_t> seed() const noexcept {
            return seed_;
        }

    private:
        friend class DslVerifier;

        Mode mode_ = Mode::Exhaustive;
        bool passed_ = false;
        std::size_t samples_checked_ = 0;
        std::string primitive_name_;
        std::string detail_;
        std::optional<std::uint32_t> seed_;
    };

    /// Auto: exhaustive if arity ≤ 4, else fuzz with `default_fuzz_seed`.
    [[nodiscard]] static StatusOr<Report> verify_primitive(const PrimitiveIr& primitive) {
        if (primitive.arity() <= max_exhaustive_arity) {
            return verify_primitive_exhaustive(primitive);
        }
        return verify_primitive_fuzz(primitive);
    }

    /// Exhaustive gate over `0..28^arity`. Arity 0 checks a single empty eval.
    /// Arity > 4 → E050 (use fuzz).
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
                    "; exhaustive mode unsupported (use verify_primitive_fuzz / seed 0xC1CADA)");
        }

        Report report;
        report.mode_ = Mode::Exhaustive;
        report.primitive_name_ = primitive.name();
        report.seed_ = std::nullopt;

        std::array<Index29, max_exhaustive_arity> args{};
        std::array<std::uint8_t, max_exhaustive_arity> digits{};
        digits.fill(0);

        const std::size_t total = pow29(arity);
        for (std::size_t sample = 0; sample < total; ++sample) {
            for (std::size_t i = 0; i < arity; ++i) {
                args[i] = Index29{digits[i]};
            }

            const std::span<const Index29> arg_span(args.data(), arity);
            Status sample_st = check_sample(primitive, arg_span);
            if (!sample_st.ok()) {
                return sample_st;
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
        report.detail_ = "exhaustive ok; samples=" + std::to_string(report.samples_checked_) +
                         "; gates=totality+determinism+cuda_mirror";
        return report;
    }

    /// Seeded property fuzz (totality + determinism + CPU↔CUDA mirror).
    /// Intended for arity > 4; also usable on smaller arities.
    [[nodiscard]] static StatusOr<Report> verify_primitive_fuzz(
        const PrimitiveIr& primitive,
        std::uint32_t seed = default_fuzz_seed,
        std::size_t sample_count = default_fuzz_samples) {
        Status st = primitive.validate();
        if (!st.ok()) {
            return st;
        }
        if (sample_count == 0) {
            return fail_verify(primitive, "fuzz sample_count must be > 0");
        }

        const std::size_t arity = primitive.arity();
        if (arity > max_fuzz_arity) {
            return fail_verify(
                primitive,
                "arity " + std::to_string(arity) + " > max_fuzz_arity " +
                    std::to_string(max_fuzz_arity));
        }

        Report report;
        report.mode_ = Mode::Fuzz;
        report.primitive_name_ = primitive.name();
        report.seed_ = seed;

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, static_cast<int>(modulus - 1));
        std::vector<Index29> args(arity);

        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            for (std::size_t i = 0; i < arity; ++i) {
                args[i] = Index29{static_cast<std::uint8_t>(dist(rng))};
            }
            Status sample_st = check_sample(primitive, args);
            if (!sample_st.ok()) {
                return sample_st;
            }
            ++report.samples_checked_;
        }

        report.passed_ = true;
        report.detail_ = "fuzz ok; seed=0x" + to_hex32(seed) +
                         "; samples=" + std::to_string(report.samples_checked_) +
                         "; gates=totality+determinism+cuda_mirror";
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

    [[nodiscard]] static std::string to_hex32(std::uint32_t v) {
        static constexpr char kHex[] = "0123456789ABCDEF";
        std::string out(8, '0');
        for (int i = 7; i >= 0; --i) {
            out[static_cast<std::size_t>(i)] = kHex[v & 0xFu];
            v >>= 4;
        }
        return out;
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

    [[nodiscard]] static Status check_sample(
        const PrimitiveIr& primitive,
        std::span<const Index29> args) {
        StatusOr<Index29> first = primitive.eval(args);
        if (!first.ok()) {
            return fail_verify(
                primitive,
                "totality failed at " + format_args(primitive, args) + ": " +
                    first.status().message());
        }
        if (first.value().value() >= modulus) {
            return fail_verify(
                primitive,
                "totality failed: result out of Index29 domain at " +
                    format_args(primitive, args));
        }

        StatusOr<Index29> second = primitive.eval(args);
        if (!second.ok()) {
            return fail_verify(
                primitive,
                "determinism failed (second eval error) at " + format_args(primitive, args) +
                    ": " + second.status().message());
        }
        if (first.value() != second.value()) {
            return fail_verify(
                primitive,
                "determinism failed at " + format_args(primitive, args) +
                    ": first=" + std::to_string(first.value().value()) +
                    " second=" + std::to_string(second.value().value()));
        }

        // CPU ↔ CUDA op-sequence mirror (Z29Expr::eval_cuda_mirror / DslEmitCuda).
        Z29Expr::Env env;
        for (std::size_t i = 0; i < args.size(); ++i) {
            env.emplace(primitive.param_names()[i], args[i]);
        }
        StatusOr<Index29> cuda_mirror = primitive.body()->eval_cuda_mirror(env);
        if (!cuda_mirror.ok()) {
            return fail_verify(
                primitive,
                "cuda mirror failed at " + format_args(primitive, args) + ": " +
                    cuda_mirror.status().message());
        }
        if (cuda_mirror.value() != first.value()) {
            return fail_verify(
                primitive,
                "cpu/cuda mirror mismatch at " + format_args(primitive, args) +
                    ": cpu=" + std::to_string(first.value().value()) +
                    " cuda_mirror=" + std::to_string(cuda_mirror.value().value()));
        }
        return Status::success();
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
