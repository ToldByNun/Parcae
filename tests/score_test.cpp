#include <parcae/core/index29.hpp>
#include <parcae/corpus/fixture_loader.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/gematria/latin_codec.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/score/ic_mod29.hpp>
#include <parcae/score/score_id.hpp>
#include <parcae/score/score_order.hpp>
#include <parcae/score/self_repeat_rate.hpp>
#include <parcae/validate/plaintext_normalizer.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] GematriaProfile load_profile() {
    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/gematria/gematria-primus-v0.json");
    REQUIRE(profile.ok());
    return profile.value();
}

[[nodiscard]] std::vector<Index29> plaintext_indices_of(const std::string& fixture_id) {
    const GematriaProfile profile = load_profile();
    LatinCodec codec(profile);
    PlaintextNormalizer normalizer(codec);

    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/" + fixture_id);
    REQUIRE(fixture.ok());
    REQUIRE(fixture.value().verification_status() == "locked");

    StatusOr<std::string> normalized = normalizer.normalize(fixture.value().plaintext());
    REQUIRE(normalized.ok());
    StatusOr<std::vector<Index29>> indices = codec.delatinize(normalized.value());
    REQUIRE(indices.ok());
    return indices.value();
}

[[nodiscard]] std::array<std::uint64_t, 29> count_symbols(const std::vector<Index29>& indices) {
    std::array<std::uint64_t, 29> counts{};
    for (const Index29 idx : indices) {
        ++counts[idx.value()];
    }
    return counts;
}

}  // namespace

TEST_CASE("ScoreId parses Tier A ids", "[score]") {
    REQUIRE(ScoreId::from_string("ic_mod29").value() == ScoreId::ic_mod29());
    REQUIRE(ScoreId::from_string("chi2_english_gp_v0").value() == ScoreId::chi2_english_gp_v0());
    REQUIRE(ScoreId::from_string("self_repeat_rate").value() == ScoreId::self_repeat_rate());
    REQUIRE_FALSE(ScoreId::from_string("nope").ok());
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::ic_mod29()) == ScoreOrder::Desc);
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::chi2_english_gp_v0()) == ScoreOrder::Asc);
    // Spec: neither assumed globally — raw report only (Asc used as neutral default).
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::self_repeat_rate()) == ScoreOrder::Asc);
}

TEST_CASE("SelfRepeatRate hand vectors", "[score][self-repeat]") {
    SECTION("rejects N < 2") {
        REQUIRE_FALSE(SelfRepeatRate::score({}).ok());
        REQUIRE_FALSE(SelfRepeatRate::score({I(0)}).ok());
    }

    SECTION("all identical → rate = 1") {
        const std::vector<Index29> xs = {I(5), I(5), I(5), I(5)};
        StatusOr<double> rate = SelfRepeatRate::score(xs);
        REQUIRE(rate.ok());
        REQUIRE(rate.value() == Catch::Approx(1.0).margin(0.0));
    }

    SECTION("no adjacent equals → rate = 0") {
        const std::vector<Index29> xs = {I(0), I(1), I(0), I(1)};
        StatusOr<double> rate = SelfRepeatRate::score(xs);
        REQUIRE(rate.ok());
        REQUIRE(rate.value() == Catch::Approx(0.0).margin(0.0));
    }

    SECTION("two of three adjacent pairs repeat → 2/3") {
        // pairs: (0,0) yes, (0,1) no, (1,1) yes → 2/3
        const std::vector<Index29> xs = {I(0), I(0), I(1), I(1)};
        StatusOr<double> rate = SelfRepeatRate::score(xs);
        REQUIRE(rate.ok());
        REQUIRE(rate.value() == Catch::Approx(2.0 / 3.0).epsilon(1e-15));
    }

    SECTION("exactly one diagonal in five symbols → 1/4") {
        const std::vector<Index29> xs = {I(2), I(7), I(7), I(3), I(9)};
        StatusOr<double> rate = SelfRepeatRate::score(xs);
        REQUIRE(rate.ok());
        REQUIRE(rate.value() == Catch::Approx(0.25).epsilon(1e-15));
    }
}

TEST_CASE("SelfRepeatRate on solved plaintext is in (0,1) and below chance 1/29 floor is not assumed",
          "[score][self-repeat]") {
    // Chance under independent uniform draws is 1/29 ≈ 0.0345; language may differ.
    // We only assert a well-formed rate — no Tier C external target.
    const std::vector<Index29> plain = plaintext_indices_of("welcome");
    REQUIRE(plain.size() >= 2);

    StatusOr<double> rate = SelfRepeatRate::score(plain);
    REQUIRE(rate.ok());
    REQUIRE(rate.value() >= 0.0);
    REQUIRE(rate.value() <= 1.0);
}

TEST_CASE("IcMod29 hand vectors", "[score][ic]") {
    SECTION("rejects N < 2") {
        REQUIRE_FALSE(IcMod29::score({}).ok());
        REQUIRE_FALSE(IcMod29::score({I(0)}).ok());
    }

    SECTION("all identical symbols → IC = 1") {
        const std::vector<Index29> xs = {I(3), I(3), I(3), I(3)};
        StatusOr<double> ic = IcMod29::score(xs);
        REQUIRE(ic.ok());
        REQUIRE(ic.value() == Catch::Approx(1.0).margin(0.0));
    }

    SECTION("two pairs → IC = 1/3") {
        // n0=2, n1=2 → numerator 2+2=4; N=4 → 4/(4*3)=1/3
        const std::vector<Index29> xs = {I(0), I(0), I(1), I(1)};
        StatusOr<double> ic = IcMod29::score(xs);
        REQUIRE(ic.ok());
        REQUIRE(ic.value() == Catch::Approx(1.0 / 3.0).epsilon(1e-15));
    }

    SECTION("uniform over 29 (one each) → IC = 0") {
        std::vector<Index29> xs;
        xs.reserve(29);
        for (std::uint8_t v = 0; v < 29; ++v) {
            xs.push_back(I(v));
        }
        StatusOr<double> ic = IcMod29::score(xs);
        REQUIRE(ic.ok());
        REQUIRE(ic.value() == Catch::Approx(0.0).margin(0.0));
    }
}

TEST_CASE("IcMod29 solved plaintext exceeds random baseline", "[score][ic]") {
    const std::vector<Index29> plain = plaintext_indices_of("a-warning");
    REQUIRE(plain.size() >= 2);

    StatusOr<double> ic_plain = IcMod29::score(plain);
    REQUIRE(ic_plain.ok());

    // Flat random expectation for 29 symbols is 1/29.
    constexpr double random_ic = 1.0 / 29.0;
    REQUIRE(ic_plain.value() > random_ic);
}

TEST_CASE("Chi2EnglishGp synthetic table + hand vector", "[score][chi2]") {
    std::array<std::uint64_t, 29> counts{};
    counts[0] = 10;
    counts[1] = 5;
    for (std::size_t i = 2; i < 29; ++i) {
        counts[i] = 1;
    }
    StatusOr<ExpectedFrequencyTable> table =
        ExpectedFrequencyTable::from_raw_counts("synthetic", counts, {"synthetic"});
    REQUIRE(table.ok());

    SECTION("empty rejected") {
        REQUIRE_FALSE(Chi2EnglishGp::score({}, table.value()).ok());
    }

    SECTION("perfect match to expected proportions → low chi2") {
        // Build an observed sequence that mirrors smoothed probabilities closely
        // by repeating the raw count multiset.
        std::vector<Index29> observed;
        for (std::uint8_t v = 0; v < 29; ++v) {
            for (std::uint64_t k = 0; k < counts[v]; ++k) {
                observed.push_back(I(v));
            }
        }
        StatusOr<double> chi2 = Chi2EnglishGp::score(observed, table.value());
        REQUIRE(chi2.ok());
        // Add-one smoothing means raw counts are not exact MLE → small but >0.
        REQUIRE(chi2.value() >= 0.0);
        REQUIRE(chi2.value() < 5.0);
    }

    SECTION("single-symbol spike → large chi2") {
        std::vector<Index29> spike(50, I(0));
        StatusOr<double> chi2 = Chi2EnglishGp::score(spike, table.value());
        REQUIRE(chi2.ok());
        REQUIRE(chi2.value() > 50.0);
    }
}

TEST_CASE("Chi2EnglishGp vs locked english-gp-expected-v0", "[score][chi2]") {
    StatusOr<ExpectedFrequencyTable> table = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(table.ok());
    REQUIRE(table.value().id() == "english-gp-expected-v0");
    REQUIRE(table.value().smoothing() == "add_one");
    REQUIRE(table.value().raw_total() > 0);

    const std::vector<Index29> warning = plaintext_indices_of("a-warning");
    const std::vector<Index29> welcome = plaintext_indices_of("welcome");

    StatusOr<double> chi_warning = Chi2EnglishGp::score(warning, table.value());
    StatusOr<double> chi_welcome = Chi2EnglishGp::score(welcome, table.value());
    REQUIRE(chi_warning.ok());
    REQUIRE(chi_welcome.ok());

    // Uniform random sequence should fit English-GP worse than real plaintext.
    std::vector<Index29> uniform;
    uniform.reserve(welcome.size());
    for (std::size_t i = 0; i < welcome.size(); ++i) {
        uniform.push_back(I(static_cast<std::uint8_t>(i % 29)));
    }
    StatusOr<double> chi_uniform = Chi2EnglishGp::score(uniform, table.value());
    REQUIRE(chi_uniform.ok());
    REQUIRE(chi_welcome.value() < chi_uniform.value());
    REQUIRE(chi_warning.value() < chi_uniform.value());
}

TEST_CASE("Dump empirical English-GP counts from locked fixtures", "[.][freqdump]") {
    const char* ids[] = {
        "a-warning",
        "some-wisdom",
        "loss-of-divinity",
        "an-instruction",
        "koan-1",
        "welcome",
        "koan-2",
        "an-end",
        "lp2-57-identity",
    };

    std::array<std::uint64_t, 29> total_counts{};
    std::uint64_t total = 0;
    for (const char* id : ids) {
        const std::vector<Index29> indices = plaintext_indices_of(id);
        const auto counts = count_symbols(indices);
        for (std::size_t i = 0; i < 29; ++i) {
            total_counts[i] += counts[i];
            total += counts[i];
        }
        std::cout << id << " n=" << indices.size() << '\n';
    }

    std::cout << "raw_total=" << total << '\n';
    std::cout << "raw_counts=[";
    for (std::size_t i = 0; i < 29; ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << total_counts[i];
    }
    std::cout << "]\n";

    const double denom = static_cast<double>(total) + 29.0;
    std::cout << "probabilities=[";
    for (std::size_t i = 0; i < 29; ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        const double p = (static_cast<double>(total_counts[i]) + 1.0) / denom;
        std::cout.precision(17);
        std::cout << p;
    }
    std::cout << "]\n";
}
