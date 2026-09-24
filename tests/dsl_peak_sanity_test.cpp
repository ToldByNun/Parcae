#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/dsl/dsl_peak_sanity.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

#if defined(PARCAE_HAS_CUDA)
#include <parcae/run/throughput_tiers.hpp>
#endif

TEST_CASE("DslPeakSanity peaks match cuda-throughput reference plateaus", "[dsl][peak]") {
    REQUIRE(DslPeakSanity::estimated_peak("T1") == 392.0e9);
    REQUIRE(DslPeakSanity::estimated_peak("T2") == 402.0e9);
    REQUIRE(DslPeakSanity::estimated_peak("T3") == 55.0e9);
    REQUIRE(DslPeakSanity::estimated_peak("F.atbash") == 550.0e9);
    REQUIRE(DslPeakSanity::estimated_peak("F.affine") == 473.0e9);
    REQUIRE(DslPeakSanity::estimated_peak("C.koan1_fused") == 372.0e9);
    REQUIRE(DslPeakSanity::estimated_peak("C.koan1_stages") == 322.0e9);
    REQUIRE(DslPeakSanity::estimated_peak("nope") == 0.0);
}

TEST_CASE("DslPeakSanity delegates peak and pass rules to BenchTierSpec", "[dsl][peak]") {
    REQUIRE(DslPeakSanity::peak_band_pct == BenchTierSpec::peak_band_pct);
    REQUIRE(DslPeakSanity::peak_band_raw == BenchTierSpec::peak_band_raw);
    for (const char* tier :
         {"T1",
          "T2",
          "T3",
          "F.atbash",
          "F.affine",
          "F.vigenere",
          "F.beaufort",
          "F.totient",
          "C.koan1_fused",
          "C.koan1_stages"}) {
        REQUIRE(DslPeakSanity::estimated_peak(tier) == BenchTierSpec::estimated_peak(tier));
        REQUIRE(DslPeakSanity::slo_floor(tier) == BenchTierSpec::slo_floor(tier));
        REQUIRE(DslPeakSanity::known_tier(tier) == BenchTierSpec::known_tier(tier));
        const double peak = BenchTierSpec::estimated_peak(tier);
        const double slo = BenchTierSpec::slo_floor(tier);
        REQUIRE(DslPeakSanity::pass_tier(0.91 * peak, slo, peak) ==
                BenchTierSpec::pass_tier(0.91 * peak, slo, peak));
        REQUIRE(DslPeakSanity::percent_peak(0.5 * peak, peak) ==
                BenchTierSpec::percent_peak(0.5 * peak, peak));
    }
}

TEST_CASE("DslPeakSanity slo floors match cuda-throughput.md", "[dsl][peak]") {
    REQUIRE(DslPeakSanity::slo_floor("T1") == 15.0e9);
    REQUIRE(DslPeakSanity::slo_floor("T2") == 3.0e9);
    REQUIRE(DslPeakSanity::slo_floor("T3") == 1.0e9);
    REQUIRE(DslPeakSanity::slo_floor("F.atbash") == 15.0e9);
    REQUIRE(DslPeakSanity::slo_floor("F.vigenere") == 3.0e9);
    REQUIRE(DslPeakSanity::slo_floor("C.koan1_fused") == 15.0e9);
}

TEST_CASE("DslPeakSanity pass_tier mirrors ThroughputTiers band", "[dsl][peak]") {
    const double peak = DslPeakSanity::estimated_peak("T1");
    const double slo = DslPeakSanity::slo_floor("T1");
    REQUIRE(DslPeakSanity::pass_tier(0.91 * peak, slo, peak));
    REQUIRE_FALSE(DslPeakSanity::pass_tier(0.80 * peak, slo, peak));
    REQUIRE_FALSE(DslPeakSanity::pass_tier(slo * 0.5, slo, peak));
    // Boundary: pct + 0.5 >= 90  ⇒  pct >= 89.5
    REQUIRE(DslPeakSanity::pass_tier(0.895 * peak, slo, peak));
    REQUIRE_FALSE(DslPeakSanity::pass_tier(0.894 * peak, slo, peak));
}

TEST_CASE("DslPeakSanity check pass / fail paths", "[dsl][peak]") {
    const double peak = DslPeakSanity::estimated_peak("C.koan1_fused");
    {
        const DslPeakSanity::Report r = DslPeakSanity::check("C.koan1_fused", 0.95 * peak);
        REQUIRE(r.passed());
        REQUIRE(r.verdict() == DslPeakSanity::Verdict::Pass);
        REQUIRE(DslPeakSanity::check_status("C.koan1_fused", 0.95 * peak).ok());
    }
    {
        const DslPeakSanity::Report r = DslPeakSanity::check("C.koan1_fused", 1.0e9);
        REQUIRE_FALSE(r.passed());
        REQUIRE(r.verdict() == DslPeakSanity::Verdict::FailSlo);
    }
    {
        const DslPeakSanity::Report r = DslPeakSanity::check("C.koan1_fused", 0.85 * peak);
        REQUIRE_FALSE(r.passed());
        REQUIRE(r.verdict() == DslPeakSanity::Verdict::FailPeakBand);
    }
    {
        const DslPeakSanity::Report r = DslPeakSanity::check("C.koan1_fused", 1.05 * peak);
        REQUIRE_FALSE(r.passed());
        REQUIRE(r.verdict() == DslPeakSanity::Verdict::ImplausibleAbovePeak);
        REQUIRE_FALSE(DslPeakSanity::check_not_above_peak("C.koan1_fused", 1.05 * peak).ok());
        REQUIRE(DslPeakSanity::check_not_above_peak("C.koan1_fused", 0.99 * peak).ok());
    }
    {
        const DslPeakSanity::Report r = DslPeakSanity::check("not_a_tier", 1.0e12);
        REQUIRE(r.verdict() == DslPeakSanity::Verdict::UnknownTier);
    }
}

TEST_CASE("DslPeakSanity suggest_tier for DSL theory names", "[dsl][peak]") {
    const StatusOr<TheoryIr> koan = TheoryIr::make(
        "koan1_style",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {},
        Z29Expr::var("x"),
        Z29Expr::var("x"));
    REQUIRE(koan.ok());
    REQUIRE(DslPeakSanity::suggest_tier(koan.value()) == "C.koan1_fused");

    const StatusOr<TheoryIr> atb = TheoryIr::make(
        "atbash",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {},
        Z29Expr::atbash(Z29Expr::var("x")),
        Z29Expr::atbash(Z29Expr::var("x")));
    REQUIRE(atb.ok());
    REQUIRE(DslPeakSanity::suggest_tier(atb.value()) == "F.atbash");

    const StatusOr<TheoryIr> other = TheoryIr::make(
        "poly2_stream",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {});
    REQUIRE(other.ok());
    REQUIRE(DslPeakSanity::suggest_tier(other.value()).empty());
}

TEST_CASE("DslPeakSanity verdict_str", "[dsl][peak]") {
    REQUIRE(std::string(DslPeakSanity::verdict_str(DslPeakSanity::Verdict::Pass)) == "pass");
    REQUIRE(
        std::string(DslPeakSanity::verdict_str(DslPeakSanity::Verdict::ImplausibleAbovePeak)) ==
        "implausible_above_peak");
}

#if defined(PARCAE_HAS_CUDA)
TEST_CASE("DslPeakSanity and ThroughputTiers both follow BenchTierSpec", "[dsl][peak][cuda]") {
    for (const char* tier :
         {"T1",
          "T2",
          "T3",
          "F.atbash",
          "F.affine",
          "F.vigenere",
          "F.beaufort",
          "F.totient",
          "C.koan1_fused",
          "C.koan1_stages"}) {
        REQUIRE(DslPeakSanity::estimated_peak(tier) == BenchTierSpec::estimated_peak(tier));
        REQUIRE(ThroughputTiers::estimated_peak(tier) == BenchTierSpec::estimated_peak(tier));
        REQUIRE(DslPeakSanity::estimated_peak(tier) == ThroughputTiers::estimated_peak(tier));
        REQUIRE(DslPeakSanity::pass_tier(0.95 * DslPeakSanity::estimated_peak(tier), 1.0, DslPeakSanity::estimated_peak(tier)) ==
                ThroughputTiers::pass_tier(
                    0.95 * ThroughputTiers::estimated_peak(tier),
                    1.0,
                    ThroughputTiers::estimated_peak(tier)));
    }

    // Primary SLO configs used by ThroughputTiers::tier1/2/3 (wired to Spec).
    REQUIRE(BenchTierSpec::t1.repeats == 64u);
    REQUIRE(BenchTierSpec::t2.repeats == 8u);
    REQUIRE(BenchTierSpec::t3.repeats == 8u);
    REQUIRE(BenchTierSpec::t1.tokens == 1048576u);
    REQUIRE(BenchTierSpec::t2.tokens == 262144u);
    REQUIRE(BenchTierSpec::t3.tokens == 262144u);
}
#endif
