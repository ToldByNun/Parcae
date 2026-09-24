#ifndef DSL_PEAK_SANITY_HPP
#define DSL_PEAK_SANITY_HPP

#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/theory_ir.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

/// Peak / SLO sanity gates aligned with `BenchTierSpec` (cuda-throughput.md).
/// Host-only: mirrors `estimated_peak` + `pass_tier` without requiring CUDA.
/// Use for artifact claims and measured CUDA rps — not for CPU `DslFuse::bench_cpu`
/// rates (those are a different metric).
///
/// **Sync rule:** `estimated_peak` / `slo_floor` / `pass_tier` MUST delegate to
/// `BenchTierSpec` (same source as `ThroughputTiers` and docs/architecture/cuda-throughput.md).
class DslPeakSanity {
public:
    /// Same ≥90% band as `BenchTierSpec::pass_tier` (89.5% raw → printed 90%).
    static constexpr double peak_band_pct = BenchTierSpec::peak_band_pct;
    static constexpr double peak_band_raw = BenchTierSpec::peak_band_raw;

    enum class Verdict : std::uint8_t {
        Pass = 0,
        FailSlo,
        FailPeakBand,
        ImplausibleAbovePeak,
        UnknownTier,
    };

    class Report {
    public:
        Report(
            std::string tier,
            double rps,
            double slo_min,
            double peak,
            double percent_peak,
            Verdict verdict,
            std::string detail)
            : tier_(std::move(tier)),
              rps_(rps),
              slo_min_(slo_min),
              peak_(peak),
              percent_peak_(percent_peak),
              verdict_(verdict),
              detail_(std::move(detail)) {}

        [[nodiscard]] const std::string& tier() const noexcept {
            return tier_;
        }

        [[nodiscard]] double rps() const noexcept {
            return rps_;
        }

        [[nodiscard]] double slo_min() const noexcept {
            return slo_min_;
        }

        [[nodiscard]] double peak() const noexcept {
            return peak_;
        }

        [[nodiscard]] double percent_peak() const noexcept {
            return percent_peak_;
        }

        [[nodiscard]] Verdict verdict() const noexcept {
            return verdict_;
        }

        [[nodiscard]] bool passed() const noexcept {
            return verdict_ == Verdict::Pass;
        }

        [[nodiscard]] const std::string& detail() const noexcept {
            return detail_;
        }

    private:
        std::string tier_;
        double rps_ = 0;
        double slo_min_ = 0;
        double peak_ = 0;
        double percent_peak_ = 0;
        Verdict verdict_ = Verdict::UnknownTier;
        std::string detail_;
    };

    /// Practical ceilings (runes/s) — delegated to `BenchTierSpec`.
    [[nodiscard]] static double estimated_peak(std::string_view tier) noexcept {
        return BenchTierSpec::estimated_peak(tier);
    }

    /// SLO floors — delegated to `BenchTierSpec`.
    [[nodiscard]] static double slo_floor(std::string_view tier) noexcept {
        return BenchTierSpec::slo_floor(tier);
    }

    [[nodiscard]] static bool known_tier(std::string_view tier) noexcept {
        return BenchTierSpec::known_tier(tier);
    }

    /// Identical rule to `BenchTierSpec::pass_tier` / `ThroughputTiers::pass_tier`.
    [[nodiscard]] static bool pass_tier(double rps, double slo_min, double peak) noexcept {
        return BenchTierSpec::pass_tier(rps, slo_min, peak);
    }

    [[nodiscard]] static double percent_peak(double rps, double peak) noexcept {
        return BenchTierSpec::percent_peak(rps, peak);
    }

    /// Full gate: unknown tier / above peak / SLO / 90% band.
    [[nodiscard]] static Report check(std::string_view tier, double rps) {
        const std::string tier_s(tier);
        const double peak = estimated_peak(tier);
        const double slo = slo_floor(tier);
        const double pct = percent_peak(rps, peak);

        if (peak <= 0.0) {
            return Report{
                tier_s,
                rps,
                slo,
                peak,
                pct,
                Verdict::UnknownTier,
                "unknown throughput tier '" + tier_s + "'"};
        }

        // Claims / measurements must not exceed the practical ceiling (stale peak → recalibrate).
        constexpr double kEps = 1.0e-9;
        if (rps > peak * (1.0 + kEps)) {
            std::ostringstream d;
            d << "implausible rps above peak: tier=" << tier_s << " rps=" << rps
              << " peak=" << peak << " pct=" << pct;
            return Report{
                tier_s, rps, slo, peak, pct, Verdict::ImplausibleAbovePeak, d.str()};
        }

        if (rps < slo) {
            std::ostringstream d;
            d << "below SLO floor: tier=" << tier_s << " rps=" << rps << " slo=" << slo;
            return Report{tier_s, rps, slo, peak, pct, Verdict::FailSlo, d.str()};
        }

        if (!pass_tier(rps, slo, peak)) {
            std::ostringstream d;
            d << "below " << peak_band_pct << "% peak band: tier=" << tier_s << " pct=" << pct;
            return Report{tier_s, rps, slo, peak, pct, Verdict::FailPeakBand, d.str()};
        }

        std::ostringstream d;
        d << "ok tier=" << tier_s << " pct=" << pct;
        return Report{tier_s, rps, slo, peak, pct, Verdict::Pass, d.str()};
    }

    /// Soft Status wrapper (Pass → success; else E050-style compile abort for claims).
    [[nodiscard]] static Status check_status(std::string_view tier, double rps) {
        const Report r = check(tier, rps);
        if (r.passed()) {
            return Status::success();
        }
        return DslDiag::make(
                   DslRuleId::E050_verify_failed,
                   std::string("peak sanity: ") + r.detail())
            .to_status();
    }

    /// Only the >peak implausibility gate (does not require ≥90% band).
    [[nodiscard]] static Status check_not_above_peak(std::string_view tier, double rps) {
        const double peak = estimated_peak(tier);
        if (peak <= 0.0) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "peak sanity: unknown tier '" + std::string(tier) + "'")
                .to_status();
        }
        constexpr double kEps = 1.0e-9;
        if (rps > peak * (1.0 + kEps)) {
            return DslDiag::make(
                       DslRuleId::E050_verify_failed,
                       "peak sanity: rps exceeds BenchTierSpec ceiling for '" +
                           std::string(tier) + "'")
                .to_status();
        }
        return Status::success();
    }

    /// Best-effort tier id for DSL theories (compose / known elementwise names).
    [[nodiscard]] static std::string suggest_tier(const TheoryIr& theory) {
        const std::string& n = theory.name();
        if (n.find("koan") != std::string::npos || n.find("atbash_then_caesar") != std::string::npos ||
            n == "koan1_style") {
            return "C.koan1_fused";
        }
        if (n == "atbash" || n.find("atbash") != std::string::npos) {
            return "F.atbash";
        }
        if (n.find("affine") != std::string::npos) {
            return "F.affine";
        }
        if (n.find("vigenere") != std::string::npos) {
            return "F.vigenere";
        }
        if (n.find("beaufort") != std::string::npos) {
            return "F.beaufort";
        }
        if (n.find("totient") != std::string::npos) {
            return "F.totient";
        }
        if (n.find("caesar") != std::string::npos) {
            return "T1";
        }
        return {};
    }

    [[nodiscard]] static const char* verdict_str(Verdict v) noexcept {
        switch (v) {
        case Verdict::Pass:
            return "pass";
        case Verdict::FailSlo:
            return "fail_slo";
        case Verdict::FailPeakBand:
            return "fail_peak_band";
        case Verdict::ImplausibleAbovePeak:
            return "implausible_above_peak";
        case Verdict::UnknownTier:
            return "unknown_tier";
        }
        return "unknown";
    }

private:
    DslPeakSanity() = delete;
};

#endif // DSL_PEAK_SANITY_HPP
