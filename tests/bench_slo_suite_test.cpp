#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <parcae/bench/bench_formatter.hpp>
#include <parcae/bench/bench_slo_suite.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/tool/context.hpp>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#define PARCAE_TEST_DATA_DIR ""
#endif

TEST_CASE("BenchSloSuite Options default extended off", "[bench][slo]") {
    const BenchSloSuite::Options opts;
    REQUIRE_FALSE(opts.extended());
    BenchSloSuite::Options on(true);
    REQUIRE(on.extended());
    on.set_extended(false);
    REQUIRE_FALSE(on.extended());
}

TEST_CASE("BenchSloSuite make_measured_row derives keys and wall", "[bench][slo]") {
    // runes/s = 512; C=4, T=16, reps=2 → keys/s = 512/16 = 32
    // wall = reps*C*T / rps = 2*4*16 / 512 = 0.25
    const BenchReport::Row row = BenchSloSuite::make_measured_row(
        "T1", "test", 512.0, BenchTierSpec::t1.slo_min, BenchTierSpec::t1.slo_max, true, 4, 16, 2);
    REQUIRE(row.name() == "T1");
    REQUIRE(row.suite() == BenchReport::Suite::Slo);
    REQUIRE(row.backend() == BenchReport::Backend::Cuda);
    REQUIRE(row.passed());
    REQUIRE(row.runes_per_sec() == 512.0);
    REQUIRE(row.keys_per_sec() == 32.0);
    REQUIRE(row.wall_seconds() == 0.25);
    REQUIRE(row.estimated_peak() == BenchTierSpec::t1.estimated_peak);
    REQUIRE(row.candidates() == 4u);
    REQUIRE(row.tokens() == 16u);
    REQUIRE(row.repeats() == 2u);
}

TEST_CASE("BenchSloSuite make_measured_row zero rps keeps keys and wall zero", "[bench][slo]") {
    const BenchReport::Row row =
        BenchSloSuite::make_measured_row("T2", "x", 0.0, 3.0e9, 10.0e9, false, 4096, 262144, 8);
    REQUIRE_FALSE(row.passed());
    REQUIRE(row.keys_per_sec() == 0.0);
    REQUIRE(row.wall_seconds() == 0.0);
    REQUIRE(row.estimated_peak() == BenchTierSpec::t2.estimated_peak);
}

#if !defined(PARCAE_HAS_CUDA)
TEST_CASE("BenchSloSuite run errors without CUDA", "[bench][slo]") {
    std::array<double, 29> probs{};
    for (double& p : probs) {
        p = 1.0 / 29.0;
    }
    std::array<std::uint64_t, 29> counts{};
    counts.fill(1);
    const ExpectedFrequencyTable freqs("synthetic", probs, counts, 29, "none", {});
    StatusOr<BenchReport::Document> doc = BenchSloSuite::run(freqs);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("CUDA") != std::string::npos);
}
#endif

#if defined(PARCAE_HAS_CUDA)
TEST_CASE("BenchSloSuite run primary T1-T3 on CUDA", "[bench][slo][cuda]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    BenchSloSuite::Options opts(false);
    StatusOr<BenchReport::Document> doc = BenchSloSuite::run(freqs.value(), opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().suite() == BenchReport::Suite::Slo);
    REQUIRE(doc.value().rows().size() == 3u);
    REQUIRE(doc.value().rows()[0].name() == "T1");
    REQUIRE(doc.value().rows()[1].name() == "T2");
    REQUIRE(doc.value().rows()[2].name() == "T3");
    REQUIRE(doc.value().rows()[0].repeats() == BenchTierSpec::t1.repeats);
    REQUIRE(doc.value().rows()[1].repeats() == BenchTierSpec::t2.repeats);
    REQUIRE(doc.value().rows()[2].repeats() == BenchTierSpec::t3.repeats);

    const std::string human = BenchFormatter::format(doc.value());
    REQUIRE(human.find("SLO TIERS") != std::string::npos);
    REQUIRE(human.find("TRANSFORM FAMILIES") == std::string::npos);

    const nlohmann::json j = doc.value().to_json(true);
    REQUIRE(j.at("omit_timing").get<bool>());
    REQUIRE(j.at("row_count").get<std::size_t>() == 3u);
    REQUIRE_FALSE(j.at("rows").at(0).contains("runes_per_sec"));
}

TEST_CASE("BenchSloSuite run extended includes F and C rows", "[bench][slo][cuda]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    BenchSloSuite::Options opts(true);
    StatusOr<BenchReport::Document> doc = BenchSloSuite::run(freqs.value(), opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().rows().size() > 3u);

    bool saw_family = false;
    bool saw_compose = false;
    for (const BenchReport::Row& row : doc.value().rows()) {
        if (row.name().size() >= 2 && row.name()[0] == 'F' && row.name()[1] == '.') {
            saw_family = true;
        }
        if (row.name().size() >= 2 && row.name()[0] == 'C' && row.name()[1] == '.') {
            saw_compose = true;
        }
    }
    REQUIRE(saw_family);
    REQUIRE(saw_compose);

    const std::string human = BenchFormatter::format(doc.value());
    REQUIRE(human.find("TRANSFORM FAMILIES") != std::string::npos);
    REQUIRE(human.find("COMPOSE") != std::string::npos);
}
#endif
