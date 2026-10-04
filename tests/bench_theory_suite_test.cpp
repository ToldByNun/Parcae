#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <parcae/bench/bench_theory_suite.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/tool/context.hpp>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#define PARCAE_TEST_DATA_DIR ""
#endif

TEST_CASE("BenchTheorySuite Options defaults", "[bench][theory]") {
    const BenchTheorySuite::Options opts;
    REQUIRE(opts.tokens() == 0u);
    REQUIRE(opts.candidates() == 0u);
    REQUIRE(opts.repeats() == 0u);
    REQUIRE_FALSE(opts.campaign_grid());
    REQUIRE(opts.compare_catalog());

    BenchTheorySuite::Options tuned;
    tuned.set_tokens(262);
    tuned.set_candidates(64);
    tuned.set_repeats(2);
    tuned.set_campaign_grid(true);
    tuned.set_compare_catalog(false);
    REQUIRE(tuned.tokens() == 262u);
    REQUIRE(tuned.candidates() == 64u);
    REQUIRE(tuned.repeats() == 2u);
    REQUIRE(tuned.campaign_grid());
    REQUIRE_FALSE(tuned.compare_catalog());
}

TEST_CASE("BenchTheorySuite make_measured_row derives rates", "[bench][theory]") {
    const BenchReport::Row row = BenchTheorySuite::make_measured_row(
        "T.theory.caesar_bytecode", "test", 512.0, 32.0, 0.25, true, 4, 16, 2, "S0_bytecode",
        BenchTierSpec::theory_s0_caesar.estimated_peak, BenchTierSpec::theory_s0_caesar.slo_min);
    REQUIRE(row.name() == "T.theory.caesar_bytecode");
    REQUIRE(row.suite() == BenchReport::Suite::Theory);
    REQUIRE(row.backend() == BenchReport::Backend::Cuda);
    REQUIRE(row.passed());
    REQUIRE(row.runes_per_sec() == 512.0);
    REQUIRE(row.keys_per_sec() == 32.0);
    REQUIRE(row.wall_seconds() == 0.25);
    REQUIRE(row.estimated_peak() == BenchTierSpec::theory_s0_caesar.estimated_peak);
    REQUIRE(row.detail() == "S0_bytecode");
}

#if !defined(PARCAE_HAS_CUDA)
TEST_CASE("BenchTheorySuite run errors without CUDA", "[bench][theory]") {
    std::array<double, 29> probs{};
    for (double& p : probs) {
        p = 1.0 / 29.0;
    }
    std::array<std::uint64_t, 29> counts{};
    counts.fill(1);
    const ExpectedFrequencyTable freqs("synthetic", probs, counts, 29, "none", {});
    StatusOr<BenchReport::Document> doc = BenchTheorySuite::run(freqs);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("CUDA") != std::string::npos);
}
#endif

#if defined(PARCAE_HAS_CUDA)
TEST_CASE("BenchTheorySuite run fair microbench on CUDA", "[bench][theory][cuda]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    // Short T for Catch2; operators use default fair T via CLI.
    BenchTheorySuite::Options opts;
    opts.set_tokens(4096);
    opts.set_repeats(2);
    StatusOr<BenchReport::Document> doc = BenchTheorySuite::run(freqs.value(), opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().suite() == BenchReport::Suite::Theory);
    REQUIRE(doc.value().rows().size() == 4u);
    REQUIRE(doc.value().rows()[0].name() == "T.theory.caesar_bytecode");
    REQUIRE(doc.value().rows()[1].name() == "T.theory.compare_caesar");
    REQUIRE(doc.value().rows()[2].name() == "T.theory.s1_lut29");
    REQUIRE(doc.value().rows()[3].name() == "T.theory.progressive");
    REQUIRE(doc.value().rows()[0].tokens() == 4096u);
    REQUIRE(doc.value().rows()[0].candidates() == BenchTierSpec::t1.candidates);
    REQUIRE(doc.value().rows()[0].estimated_peak() ==
            BenchTierSpec::theory_s0_caesar.estimated_peak);
    REQUIRE(doc.value().rows()[2].estimated_peak() ==
            BenchTierSpec::theory_s1_lut29.estimated_peak);
    REQUIRE(doc.value().rows()[3].estimated_peak() ==
            BenchTierSpec::theory_s2_linear.estimated_peak);
    REQUIRE(doc.value().rows()[0].detail().find("underfill_not_slo_gate") != std::string::npos);
    // Caesar fair row prefers ShapeInline (or S1 soft) after emit classify.
    REQUIRE((doc.value().rows()[0].detail().find("specialize_ShapeInline") != std::string::npos ||
             doc.value().rows()[0].detail().find("specialize_S1") != std::string::npos));
    REQUIRE(doc.value().rows()[2].detail().find("S1_device_bake") != std::string::npos);
    REQUIRE(doc.value().rows()[0].runes_per_sec() > 0.0);
    REQUIRE(doc.value().rows()[1].runes_per_sec() > 0.0);
    REQUIRE(doc.value().rows()[2].runes_per_sec() > 0.0);
    REQUIRE(doc.value().all_pass());
}

TEST_CASE("BenchTheorySuite campaign grid adds underfill row", "[bench][theory][cuda]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    BenchTheorySuite::Options opts;
    opts.set_tokens(1024);
    opts.set_repeats(1);
    opts.set_campaign_grid(true);
    opts.set_compare_catalog(false);
    StatusOr<BenchReport::Document> doc = BenchTheorySuite::run(freqs.value(), opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().rows().size() == 4u);
    REQUIRE(doc.value().rows()[0].name() == "T.theory.caesar_bytecode");
    REQUIRE(doc.value().rows()[1].name() == "T.theory.s1_lut29");
    REQUIRE(doc.value().rows()[2].name() == "T.theory.progressive");
    REQUIRE(doc.value().rows()[3].name() == "T.theory.caesar_campaign");
    REQUIRE(doc.value().rows()[3].candidates() == 16384u);
    REQUIRE(doc.value().rows()[3].tokens() == 262u);
    REQUIRE(doc.value().rows()[3].detail().find("underfill_not_slo_gate") != std::string::npos);
}
#endif
