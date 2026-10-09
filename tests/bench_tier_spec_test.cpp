#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_peak_sanity.hpp>
#include <string_view>

using Catch::Approx;

TEST_CASE("BenchTierSpec DRAM roofline is physical BW / 1 B cipher/rune", "[bench][spec]") {
    REQUIRE(BenchTierSpec::kDramBandwidthBytesPerSec == 896.0e9);
    REQUIRE(BenchTierSpec::kHistCipherBytesPerRune == 1.0);
    REQUIRE(BenchTierSpec::dram_roofline_hist_peak() == 896.0e9);
}

TEST_CASE("BenchTierSpec shared-cipher occupancy DRAM peak is diary traffic model",
          "[bench][spec][traffic]") {
    // atbash ncu: 1.490688e6 / (512*262144) ≈ 0.01111 B/rune → ~80.7 TB runes/s
    REQUIRE(BenchTierSpec::kHistBytesPerRuneSharedCipherOccupancy == 0.01111);
    const double diary = BenchTierSpec::dram_roofline_shared_cipher_occupancy_peak();
    REQUIRE(diary ==
            Approx(BenchTierSpec::kDramBandwidthBytesPerSec /
                   BenchTierSpec::kHistBytesPerRuneSharedCipherOccupancy)
                .epsilon(1e-12));
    REQUIRE(diary > BenchTierSpec::dram_roofline_hist_peak());
    // Diary is not the Done gate — estimated_peak uses compute roof.
    REQUIRE(BenchTierSpec::estimated_peak("F.atbash") != diary);
    REQUIRE(BenchTierSpec::estimated_peak("F.totient") != diary);
}

TEST_CASE("BenchTierSpec shared-cipher compute roof gates Atbash/totient Done",
          "[bench][spec][traffic][compute_roof]") {
    REQUIRE(BenchTierSpec::kSharedCipherComputeRoofRps == 2.0e12);
    const double roof = BenchTierSpec::shared_cipher_compute_roof_rps();
    REQUIRE(roof == BenchTierSpec::kSharedCipherComputeRoofRps);
    // Below DRAM diary; above unique-key 896B DRAM roof.
    REQUIRE(roof < BenchTierSpec::dram_roofline_shared_cipher_occupancy_peak());
    REQUIRE(roof > BenchTierSpec::dram_roofline_hist_peak());
    REQUIRE(BenchTierSpec::estimated_peak("F.atbash") == roof);
    REQUIRE(BenchTierSpec::estimated_peak("F.totient") == roof);
    REQUIRE(BenchTierSpec::estimated_peak("T.dsl_smart.custom_atbash") == roof);
    REQUIRE(BenchTierSpec::estimated_peak("T.dsl_smart.compare_Fatbash") == roof);
    REQUIRE(BenchTierSpec::dsl_smart_atbash.estimated_peak == roof);
    REQUIRE(BenchTierSpec::dsl_smart_compare_atbash.estimated_peak == roof);
    // Quiet Atbash-class RPS stays ≤100 under compute roof (not DRAM diary).
    REQUIRE(BenchTierSpec::percent_peak(1678.0e9, roof) < 100.0);
    REQUIRE(BenchTierSpec::percent_peak(1600.0e9, roof) < 100.0);
    REQUIRE(BenchTierSpec::percent_peak(1109.0e9, roof) < 100.0);
    // Column remap families use column Remap roof (not 896B DRAM).
    REQUIRE(BenchTierSpec::estimated_peak("F.vigenere") ==
            BenchTierSpec::column_remap_hist_roof_rps());
}

TEST_CASE("BenchTierSpec Remap roofs gate production hist Done", "[bench][spec][remap]") {
    REQUIRE(BenchTierSpec::kAlphabetRemapHistRoofRps == 2.0e12);
    REQUIRE(BenchTierSpec::kColumnRemapHistRoofRps == 2.0e12);
    REQUIRE(BenchTierSpec::kLagRemapHistRoofRps == 2.0e12);
    REQUIRE(BenchTierSpec::kBigramRemapHistRoofRps == 1.0e12);
    REQUIRE(BenchTierSpec::alphabet_remap_hist_roof_rps() ==
            BenchTierSpec::kAlphabetRemapHistRoofRps);
    REQUIRE(BenchTierSpec::column_remap_hist_roof_rps() == BenchTierSpec::kColumnRemapHistRoofRps);
    REQUIRE(BenchTierSpec::lag_remap_hist_roof_rps() == BenchTierSpec::kLagRemapHistRoofRps);
    REQUIRE(BenchTierSpec::bigram_remap_hist_roof_rps() == BenchTierSpec::kBigramRemapHistRoofRps);

    // Above unique-key 896B DRAM diary; Remap Done is not 1 B/rune.
    REQUIRE(BenchTierSpec::alphabet_remap_hist_roof_rps() >
            BenchTierSpec::dram_roofline_hist_peak());
    REQUIRE(BenchTierSpec::estimated_peak("T1") == BenchTierSpec::alphabet_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T2") == BenchTierSpec::lag_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T3") == BenchTierSpec::bigram_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.theory.s1_lut29") ==
            BenchTierSpec::alphabet_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.theory.s2_linear") ==
            BenchTierSpec::column_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.theory.s4_autokey") ==
            BenchTierSpec::lag_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.theory.s5_poly") ==
            BenchTierSpec::column_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("F.beaufort") ==
            BenchTierSpec::column_remap_hist_roof_rps());
    // Legacy decode stages stay on 896B.
    REQUIRE(BenchTierSpec::estimated_peak("C.koan1_fused") == 896.0e9);
}

TEST_CASE("BenchTierSpec Affine diary DRAM peak is not production Done",
          "[bench][spec][traffic][affine]") {
    // affine ncu cache_bound 2026-10-06: ~4.0566e6 / (812*262144) ≈ 0.01906 B/rune
    REQUIRE(BenchTierSpec::kHistBytesPerRuneAffineSharedCipher == 0.01906);
    const double diary = BenchTierSpec::dram_roofline_affine_shared_cipher_peak();
    REQUIRE(diary ==
            Approx(BenchTierSpec::kDramBandwidthBytesPerSec /
                   BenchTierSpec::kHistBytesPerRuneAffineSharedCipher)
                .epsilon(1e-12));
    REQUIRE(diary > BenchTierSpec::dram_roofline_hist_peak());
    REQUIRE(diary < BenchTierSpec::dram_roofline_shared_cipher_occupancy_peak());
    // Production Affine remap Done = alphabet Remap roof, not diary DRAM.
    const double peak = BenchTierSpec::alphabet_remap_hist_roof_rps();
    REQUIRE(BenchTierSpec::estimated_peak("F.affine") == peak);
    REQUIRE(BenchTierSpec::estimated_peak("T.dsl_smart.custom_affine") == peak);
    REQUIRE(BenchTierSpec::estimated_peak("T.dsl_smart.compare_Faffine") == peak);
    REQUIRE(BenchTierSpec::dsl_smart_affine.estimated_peak == peak);
    REQUIRE(BenchTierSpec::dsl_smart_compare_affine.estimated_peak == peak);
    REQUIRE(BenchTierSpec::estimated_peak("F.affine") != diary);
    REQUIRE(BenchTierSpec::percent_peak(1142.0e9, peak) < 100.0);
    REQUIRE(BenchTierSpec::percent_peak(885.0e9, peak) < 100.0);
}

TEST_CASE("BenchTierSpec T1 config matches canonical SLO table", "[bench][spec]") {
    REQUIRE(std::string_view(BenchTierSpec::t1.id) == "T1");
    REQUIRE(BenchTierSpec::t1.candidates == Index29::modulus);
    REQUIRE(BenchTierSpec::t1.candidates == 29u);
    REQUIRE(BenchTierSpec::t1.tokens == 1048576u);
    REQUIRE(BenchTierSpec::t1.repeats == 64u);
    REQUIRE(BenchTierSpec::t1.slo_min == 15.0e9);
    REQUIRE(BenchTierSpec::t1.slo_max == 35.0e9);
    REQUIRE(BenchTierSpec::t1.estimated_peak == BenchTierSpec::alphabet_remap_hist_roof_rps());
}

TEST_CASE("BenchTierSpec T2 config matches canonical SLO table", "[bench][spec]") {
    REQUIRE(std::string_view(BenchTierSpec::t2.id) == "T2");
    REQUIRE(BenchTierSpec::t2.candidates == 4096u);
    REQUIRE(BenchTierSpec::t2.tokens == 262144u);
    REQUIRE(BenchTierSpec::t2.repeats == 8u);
    REQUIRE(BenchTierSpec::t2.slo_min == 3.0e9);
    REQUIRE(BenchTierSpec::t2.slo_max == 10.0e9);
    REQUIRE(BenchTierSpec::t2.estimated_peak == BenchTierSpec::lag_remap_hist_roof_rps());
}

TEST_CASE("BenchTierSpec T3 config matches canonical SLO table", "[bench][spec]") {
    REQUIRE(std::string_view(BenchTierSpec::t3.id) == "T3");
    REQUIRE(BenchTierSpec::t3.candidates == 512u);
    REQUIRE(BenchTierSpec::t3.tokens == 262144u);
    REQUIRE(BenchTierSpec::t3.repeats == 8u);
    REQUIRE(BenchTierSpec::t3.slo_min == 1.0e9);
    REQUIRE(BenchTierSpec::t3.slo_max == 0.0);
    REQUIRE(BenchTierSpec::t3.estimated_peak == BenchTierSpec::bigram_remap_hist_roof_rps());
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
    for (const char* tier : {"T1", "T2", "T3", "F.atbash", "F.affine", "F.vigenere", "F.beaufort",
                             "F.totient", "C.koan1_fused", "C.koan1_stages"}) {
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
    REQUIRE(BenchTierSpec::percent_peak(448.0e9, 896.0e9) == 50.0);
    REQUIRE(BenchTierSpec::percent_peak(1.0, 0.0) == 0.0);
    REQUIRE(std::isfinite(
        BenchTierSpec::percent_peak(BenchTierSpec::t3.slo_min, BenchTierSpec::t3.estimated_peak)));
}

TEST_CASE("BenchTierSpec primary reps match wired ThroughputTiers contract", "[bench][spec]") {
    // ThroughputTiers::tier1/2/3 read these constexpr fields (reps 64/8/8).
    REQUIRE(BenchTierSpec::t1.repeats == 64u);
    REQUIRE(BenchTierSpec::t2.repeats == 8u);
    REQUIRE(BenchTierSpec::t3.repeats == 8u);
    REQUIRE(BenchTierSpec::t1.candidates == 29u);
    REQUIRE(BenchTierSpec::t2.candidates == 4096u);
    REQUIRE(BenchTierSpec::t3.candidates == 512u);
}

TEST_CASE("BenchTierSpec theory rows use Remap roofs", "[bench][spec][remap]") {
    REQUIRE(std::string_view(BenchTierSpec::theory_s0_caesar.id) == "T.theory.caesar_bytecode");
    REQUIRE(BenchTierSpec::theory_s0_caesar.estimated_peak ==
            BenchTierSpec::alphabet_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::theory_s1_lut29.estimated_peak ==
            BenchTierSpec::alphabet_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::theory_s2_linear.estimated_peak ==
            BenchTierSpec::column_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::theory_s4_autokey.estimated_peak ==
            BenchTierSpec::lag_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::theory_s5_poly.estimated_peak ==
            BenchTierSpec::column_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::theory_s0_caesar.tokens == BenchTierSpec::fair_gate_tokens());
    REQUIRE(BenchTierSpec::is_fair_gate_tokens(1048576u));
    REQUIRE_FALSE(BenchTierSpec::is_fair_gate_tokens(4096u));

    REQUIRE(BenchTierSpec::find_theory("T.theory.s0") == &BenchTierSpec::theory_s0_caesar);
    REQUIRE(BenchTierSpec::find_theory("T.theory.s1") == &BenchTierSpec::theory_s1_lut29);
    REQUIRE(BenchTierSpec::find_theory("T.theory.progressive") == &BenchTierSpec::theory_s2_linear);
    REQUIRE(BenchTierSpec::find_theory("T.theory.s4_autokey") == &BenchTierSpec::theory_s4_autokey);
    REQUIRE(BenchTierSpec::find_theory("T.theory.s5") == &BenchTierSpec::theory_s5_poly);
    REQUIRE(BenchTierSpec::find_theory("nope") == nullptr);

    REQUIRE(BenchTierSpec::estimated_peak("T.theory.caesar_bytecode") ==
            BenchTierSpec::alphabet_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.theory.s1_lut29") ==
            BenchTierSpec::alphabet_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.theory.s4_autokey") ==
            BenchTierSpec::lag_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.theory.s5_poly") ==
            BenchTierSpec::column_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("F.atbash") ==
            BenchTierSpec::shared_cipher_compute_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.dsl_smart.custom_atbash") ==
            BenchTierSpec::shared_cipher_compute_roof_rps());
    REQUIRE(BenchTierSpec::estimated_peak("T.dsl_smart.compare_caesar") ==
            BenchTierSpec::alphabet_remap_hist_roof_rps());
    REQUIRE(BenchTierSpec::slo_floor("T.theory.s1_lut29") == 15.0e9);
    REQUIRE(BenchTierSpec::slo_floor("T.dsl_smart.custom_affine") == 15.0e9);
    REQUIRE(BenchTierSpec::find_dsl_smart("T.dsl_smart.custom_caesar") ==
            &BenchTierSpec::dsl_smart_caesar);
    REQUIRE(BenchTierSpec::find_dsl_smart("nope") == nullptr);

    REQUIRE(BenchTierSpec::checkpoint_50B_applicable(896.0e9));
    REQUIRE(BenchTierSpec::checkpoint_50B_applicable(
        BenchTierSpec::alphabet_remap_hist_roof_rps()));
    REQUIRE(BenchTierSpec::checkpoint_50B_hit(55.0e9, 896.0e9));
    REQUIRE_FALSE(BenchTierSpec::checkpoint_50B_hit(40.0e9, 896.0e9));

    const double peak = BenchTierSpec::theory_s1_lut29.estimated_peak;
    const double slo = BenchTierSpec::theory_s1_lut29.slo_min;
    REQUIRE(BenchTierSpec::pass_tier(0.91 * peak, slo, peak));
    REQUIRE_FALSE(BenchTierSpec::pass_tier(0.80 * peak, slo, peak));
}
