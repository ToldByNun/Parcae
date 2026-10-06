#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <parcae/bench/bench_dsl_smart_suite.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/tool/context.hpp>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#define PARCAE_TEST_DATA_DIR ""
#endif

TEST_CASE("BenchDslSmartSuite Options defaults", "[bench][dsl_smart]") {
    const BenchDslSmartSuite::Options opts;
    REQUIRE(opts.tokens() == 0u);
    REQUIRE(opts.repeats() == 0u);
    REQUIRE(opts.compare_catalog());

    BenchDslSmartSuite::Options tuned;
    tuned.set_tokens(4096);
    tuned.set_repeats(2);
    tuned.set_compare_catalog(false);
    REQUIRE(tuned.tokens() == 4096u);
    REQUIRE(tuned.repeats() == 2u);
    REQUIRE_FALSE(tuned.compare_catalog());
}

TEST_CASE("BenchTierSpec dsl_smart tiers share fair Kernel SLO roof", "[bench][dsl_smart]") {
    REQUIRE(BenchTierSpec::dsl_smart_atbash.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_compare_atbash.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_caesar.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_compare_caesar.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_affine.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_compare_affine.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_linear.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_compare_linear.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_autokey.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::dsl_smart_compare_autokey.tokens == BenchTierSpec::fair_gate_tokens());

    REQUIRE(BenchTierSpec::dsl_smart_atbash.candidates == 512u);
    REQUIRE(BenchTierSpec::dsl_smart_compare_atbash.candidates == 512u);
    REQUIRE(BenchTierSpec::dsl_smart_caesar.candidates == 29u);
    REQUIRE(BenchTierSpec::dsl_smart_compare_caesar.candidates == 29u);
    REQUIRE(BenchTierSpec::dsl_smart_affine.candidates == 812u);
    REQUIRE(BenchTierSpec::dsl_smart_compare_affine.candidates == 812u);
    REQUIRE(BenchTierSpec::dsl_smart_linear.candidates == 841u);
    REQUIRE(BenchTierSpec::dsl_smart_compare_linear.candidates == 841u);
    REQUIRE(BenchTierSpec::dsl_smart_autokey.candidates == 28u);
    REQUIRE(BenchTierSpec::dsl_smart_compare_autokey.candidates == 28u);

    REQUIRE(BenchTierSpec::dsl_smart_atbash.estimated_peak ==
            BenchTierSpec::kSharedCipherComputeRoofRps);
    REQUIRE(BenchTierSpec::dsl_smart_linear.estimated_peak ==
            BenchTierSpec::kDramRooflineHistPeak);
    REQUIRE(BenchTierSpec::dsl_smart_autokey.estimated_peak ==
            BenchTierSpec::kDramRooflineHistPeak);
    REQUIRE(BenchTierSpec::find_dsl_smart("T.dsl_smart.custom_atbash") ==
            &BenchTierSpec::dsl_smart_atbash);
    REQUIRE(BenchTierSpec::find_dsl_smart("T.dsl_smart.compare_Fatbash") ==
            &BenchTierSpec::dsl_smart_compare_atbash);
    REQUIRE(BenchTierSpec::find_dsl_smart("T.dsl_smart.custom_linear") ==
            &BenchTierSpec::dsl_smart_linear);
    REQUIRE(BenchTierSpec::find_dsl_smart("T.dsl_smart.compare_S2") ==
            &BenchTierSpec::dsl_smart_compare_linear);
    REQUIRE(BenchTierSpec::find_dsl_smart("T.dsl_smart.custom_autokey") ==
            &BenchTierSpec::dsl_smart_autokey);
    REQUIRE(BenchTierSpec::find_dsl_smart("T.dsl_smart.compare_S4") ==
            &BenchTierSpec::dsl_smart_compare_autokey);
}

#if !defined(PARCAE_HAS_CUDA)
TEST_CASE("BenchDslSmartSuite run errors without CUDA", "[bench][dsl_smart]") {
    std::array<double, 29> probs{};
    for (double& p : probs) {
        p = 1.0 / 29.0;
    }
    std::array<std::uint64_t, 29> counts{};
    counts.fill(1);
    const ExpectedFrequencyTable freqs("synthetic", probs, counts, 29, "none", {});
    StatusOr<BenchReport::Document> doc = BenchDslSmartSuite::run(freqs);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("CUDA") != std::string::npos);
}
#endif

#if defined(PARCAE_HAS_CUDA)
TEST_CASE("BenchDslSmartSuite short-T customs vs catalog twins", "[bench][dsl_smart][cuda]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    // Short T for Catch2; fair T≥2^20 is the CLI / profiles/dsl_smart gate.
    BenchDslSmartSuite::Options opts;
    opts.set_tokens(4096);
    opts.set_repeats(2);
    StatusOr<BenchReport::Document> doc = BenchDslSmartSuite::run(freqs.value(), opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().suite() == BenchReport::Suite::DslSmart);
    REQUIRE(doc.value().rows().size() == 10u);

    REQUIRE(doc.value().rows()[0].name() == "T.dsl_smart.custom_atbash");
    REQUIRE(doc.value().rows()[1].name() == "T.dsl_smart.compare_Fatbash");
    REQUIRE(doc.value().rows()[2].name() == "T.dsl_smart.custom_caesar");
    REQUIRE(doc.value().rows()[3].name() == "T.dsl_smart.compare_caesar");
    REQUIRE(doc.value().rows()[4].name() == "T.dsl_smart.custom_affine");
    REQUIRE(doc.value().rows()[5].name() == "T.dsl_smart.compare_Faffine");
    REQUIRE(doc.value().rows()[6].name() == "T.dsl_smart.custom_linear");
    REQUIRE(doc.value().rows()[7].name() == "T.dsl_smart.compare_S2");
    REQUIRE(doc.value().rows()[8].name() == "T.dsl_smart.custom_autokey");
    REQUIRE(doc.value().rows()[9].name() == "T.dsl_smart.compare_S4");

    REQUIRE(doc.value().rows()[0].candidates() == 512u);
    REQUIRE(doc.value().rows()[1].candidates() == 512u);
    REQUIRE(doc.value().rows()[2].candidates() == 29u);
    REQUIRE(doc.value().rows()[3].candidates() == 29u);
    REQUIRE(doc.value().rows()[4].candidates() == 812u);
    REQUIRE(doc.value().rows()[5].candidates() == 812u);
    REQUIRE(doc.value().rows()[6].candidates() == 841u);
    REQUIRE(doc.value().rows()[7].candidates() == 841u);
    REQUIRE(doc.value().rows()[8].candidates() == 28u);
    REQUIRE(doc.value().rows()[9].candidates() == 28u);

    for (const BenchReport::Row& row : doc.value().rows()) {
        REQUIRE(row.tokens() == 4096u);
        REQUIRE(row.suite() == BenchReport::Suite::DslSmart);
        REQUIRE(row.detail().find("underfill_not_slo_gate") != std::string::npos);
        REQUIRE(row.runes_per_sec() > 0.0);
    }

    REQUIRE(doc.value().rows()[0].estimated_peak() ==
            BenchTierSpec::kSharedCipherComputeRoofRps);
    REQUIRE(doc.value().rows()[2].estimated_peak() == BenchTierSpec::kDramRooflineHistPeak);
    REQUIRE(doc.value().rows()[6].estimated_peak() == BenchTierSpec::kDramRooflineHistPeak);
    REQUIRE(doc.value().rows()[8].estimated_peak() == BenchTierSpec::kDramRooflineHistPeak);

    REQUIRE(doc.value().rows()[0].detail().find("ShapeInline_atbash") != std::string::npos);
    REQUIRE(doc.value().rows()[1].detail().find("catalog_F.atbash") != std::string::npos);
    REQUIRE(doc.value().rows()[2].detail().find("ShapeInline_caesar") != std::string::npos);
    REQUIRE(doc.value().rows()[3].detail().find("catalog_CaesarChi2") != std::string::npos);
    REQUIRE(doc.value().rows()[4].detail().find("ShapeInline_affine") != std::string::npos);
    REQUIRE(doc.value().rows()[5].detail().find("catalog_F.affine") != std::string::npos);
    REQUIRE(doc.value().rows()[6].detail().find("S2_linear") != std::string::npos);
    REQUIRE(doc.value().rows()[7].detail().find("catalog_S2_linear") != std::string::npos);
    REQUIRE(doc.value().rows()[8].detail().find("S4_autokey") != std::string::npos);
    REQUIRE(doc.value().rows()[9].detail().find("catalog_S4_autokey") != std::string::npos);
    REQUIRE(doc.value().all_pass());
}

TEST_CASE("BenchDslSmartSuite can skip catalog twins", "[bench][dsl_smart][cuda]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    BenchDslSmartSuite::Options opts;
    opts.set_tokens(1024);
    opts.set_repeats(1);
    opts.set_compare_catalog(false);
    StatusOr<BenchReport::Document> doc = BenchDslSmartSuite::run(freqs.value(), opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().rows().size() == 5u);
    REQUIRE(doc.value().rows()[0].name() == "T.dsl_smart.custom_atbash");
    REQUIRE(doc.value().rows()[1].name() == "T.dsl_smart.custom_caesar");
    REQUIRE(doc.value().rows()[2].name() == "T.dsl_smart.custom_affine");
    REQUIRE(doc.value().rows()[3].name() == "T.dsl_smart.custom_linear");
    REQUIRE(doc.value().rows()[4].name() == "T.dsl_smart.custom_autokey");
}
#endif
