#include <parcae/bench/bench_metric.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

TEST_CASE("BenchMetric runes_per_sec and keys_per_sec formulas", "[bench][metric]") {
    // 2 reps × 10 C × 100 T / 0.5 s = 4000 runes/s; keys/s = 2×10/0.5 = 40
    REQUIRE(BenchMetric::runes_per_sec(2, 10, 100, 0.5) == 4000.0);
    REQUIRE(BenchMetric::keys_per_sec(2, 10, 0.5) == 40.0);
    REQUIRE(BenchMetric::runes_per_sec(1, 1, 1, 0.0) == 0.0);
    REQUIRE(BenchMetric::keys_per_sec(1, 1, -1.0) == 0.0);
}

TEST_CASE("BenchMetric Sample::from_elapsed", "[bench][metric]") {
    const BenchMetric::Sample s = BenchMetric::Sample::from_elapsed(64, 29, 1048576, 2.0);
    REQUIRE(s.wall_seconds() == 2.0);
    const double expected_rps =
        BenchMetric::runes_per_sec(64, 29, 1048576, 2.0);
    const double expected_kps = BenchMetric::keys_per_sec(64, 29, 2.0);
    REQUIRE(s.runes_per_sec() == expected_rps);
    REQUIRE(s.keys_per_sec() == expected_kps);
    REQUIRE(std::isfinite(s.runes_per_sec()));
}

TEST_CASE("BenchMetric median3 sort network", "[bench][metric]") {
    REQUIRE(BenchMetric::median3(1.0, 2.0, 3.0) == 2.0);
    REQUIRE(BenchMetric::median3(3.0, 1.0, 2.0) == 2.0);
    REQUIRE(BenchMetric::median3(2.0, 3.0, 1.0) == 2.0);
    REQUIRE(BenchMetric::median3(5.0, 5.0, 5.0) == 5.0);
    REQUIRE(BenchMetric::median3(9.0, 1.0, 1.0) == 1.0);
}

TEST_CASE("BenchMetric median3_sample picks middle runes_per_sec triple", "[bench][metric]") {
    const BenchMetric::Sample lo{1.0, 10.0, 1.0};
    const BenchMetric::Sample mid{2.0, 20.0, 2.0};
    const BenchMetric::Sample hi{3.0, 30.0, 3.0};
    const BenchMetric::Sample picked = BenchMetric::median3_sample(hi, lo, mid);
    REQUIRE(picked.runes_per_sec() == 20.0);
    REQUIRE(picked.wall_seconds() == 2.0);
    REQUIRE(picked.keys_per_sec() == 2.0);
}

TEST_CASE("BenchMetric kind_str", "[bench][metric]") {
    REQUIRE(std::string(BenchMetric::kind_str(BenchMetric::Kind::RunesPerSec)) == "runes/s");
    REQUIRE(std::string(BenchMetric::kind_str(BenchMetric::Kind::KeysPerSec)) == "keys/s");
}
