#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_peak_sanity.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string_view>

TEST_CASE("BenchTierSpec T1 config matches canonical SLO table", "[bench][spec]") {
    REQUIRE(std::string_view(BenchTierSpec::t1.id) == "T1");
    REQUIRE(BenchTierSpec::t1.candidates == Index29::modulus);
    REQUIRE(BenchTierSpec::t1.candidates == 29u);
    REQUIRE(BenchTierSpec::t1.tokens == 1048576u);
    REQUIRE(BenchTierSpec::t1.repeats == 64u);
    REQUIRE(BenchTierSpec::t1.slo_min == 15.0e9);
    REQUIRE(BenchTierSpec::t1.slo_max == 35.0e9);
    REQUIRE(BenchTierSpec::t1.estimated_peak == 392.0e9);
}

TEST_CASE("BenchTierSpec T2 config matches canonical SLO table", "[bench][spec]") {
    REQUIRE(std::string_view(BenchTierSpec::t2.id) == "T2");
    REQUIRE(BenchTierSpec::t2.candidates == 4096u);
    REQUIRE(BenchTierSpec::t2.tokens == 262144u);
    REQUIRE(BenchTierSpec::t2.repeats == 8u);
    REQUIRE(BenchTierSpec::t2.slo_min == 3.0e9);
    REQUIRE(BenchTierSpec::t2.slo_max == 10.0e9);
    REQUIRE(BenchTierSpec::t2.estimated_peak == 402.0e9);
}

TEST_CASE("BenchTierSpec T3 config matches canonical SLO table", "[bench][spec]") {
    REQUIRE(std::string_view(BenchTierSpec::t3.id) == "T3");
    REQUIRE(BenchTierSpec::t3.candidates == 512u);
    REQUIRE(BenchTierSpec::t3.tokens == 262144u);
    REQUIRE(BenchTierSpec::t3.repeats == 8u);
    REQUIRE(BenchTierSpec::t3.slo_min == 1.0e9);
    REQUIRE(BenchTierSpec::t3.slo_max == 0.0);
    REQUIRE(BenchTierSpec::t3.estimated_peak == 55.0e9);
}

TEST_CASE("BenchTierSpec primary iteration order is T1 T2 T3", "[bench][spec]") {
    REQUIRE(BenchTierSpec::primary_tier_count == 3u);
    REQUIRE(std::string_view(BenchTierSpec::tier_at(0).id) == "T1");
    REQUIRE(std::string_view(BenchTierSpec::tier_at(1).id) == "T2");
    REQUIRE(std::string_view(BenchTierSpec::tier_at(2).id) == "T3");
    REQUIRE(BenchTierSpec::find_primary("T2") == &BenchTierSpec::t2);
    REQUIRE(BenchTierSpec::find_primary("nope") == nullptr);
}

TEST_CASE("BenchTierSpec peaks match DslPeakSanity and cuda-throughput plateaus", "[bench][spec]") {
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
        REQUIRE(BenchTierSpec::estimated_peak(tier) == DslPeakSanity::estimated_peak(tier));
        REQUIRE(BenchTierSpec::slo_floor(tier) == DslPeakSanity::slo_floor(tier));
        REQUIRE(BenchTierSpec::known_tier(tier));
    }
    REQUIRE(BenchTierSpec::estimated_peak("nope") == 0.0);
    REQUIRE_FALSE(BenchTierSpec::known_tier("nope"));
}

TEST_CASE("BenchTierSpec pass_tier mirrors DslPeakSanity band", "[bench][spec]") {
    const double peak = BenchTierSpec::t1.estimated_peak;
    const double slo = BenchTierSpec::t1.slo_min;
    REQUIRE(BenchTierSpec::pass_tier(0.91 * peak, slo, peak));
    REQUIRE_FALSE(BenchTierSpec::pass_tier(0.80 * peak, slo, peak));
    REQUIRE_FALSE(BenchTierSpec::pass_tier(slo * 0.5, slo, peak));
    // Boundary: pct + 0.5 >= 90  ⇒  pct >= 89.5
    REQUIRE(BenchTierSpec::pass_tier(0.895 * peak, slo, peak));
    REQUIRE_FALSE(BenchTierSpec::pass_tier(0.894 * peak, slo, peak));
    REQUIRE(BenchTierSpec::pass_tier(0.91 * peak, slo, peak) ==
            DslPeakSanity::pass_tier(0.91 * peak, slo, peak));
}

TEST_CASE("BenchTierSpec percent_peak", "[bench][spec]") {
    REQUIRE(BenchTierSpec::percent_peak(196.0e9, 392.0e9) == 50.0);
    REQUIRE(BenchTierSpec::percent_peak(1.0, 0.0) == 0.0);
    REQUIRE(std::isfinite(BenchTierSpec::percent_peak(BenchTierSpec::t3.slo_min,
                                                       BenchTierSpec::t3.estimated_peak)));
}
