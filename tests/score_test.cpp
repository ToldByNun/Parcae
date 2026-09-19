#include <parcae/core/index29.hpp>
#include <parcae/corpus/fixture_loader.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/gematria/latin_codec.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/exact_match.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/score/hamming_agreement.hpp>
#include <parcae/score/ic_mod29.hpp>
#include <parcae/score/score_catalog_entry.hpp>
#include <parcae/score/score_id.hpp>
#include <parcae/score/score_order.hpp>
#include <parcae/score/score_registry.hpp>
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

/// Deterministic Index29 noise (LCG). Same length as `n`; independent of plaintext.
[[nodiscard]] std::vector<Index29> lcg_noise(std::size_t n, std::uint64_t seed) {
    std::vector<Index29> out;
    out.reserve(n);
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1ULL;
        out.push_back(Index29{static_cast<std::uint8_t>((state >> 33) % 29)});
    }
    return out;
}

}  // namespace

TEST_CASE("ScoreId parses Tier A ids", "[score]") {
    REQUIRE(ScoreId::from_string("exact_match").value() == ScoreId::exact_match());
    REQUIRE(ScoreId::from_string("hamming_agreement").value() == ScoreId::hamming_agreement());
    REQUIRE(ScoreId::from_string("ic_mod29").value() == ScoreId::ic_mod29());
    REQUIRE(ScoreId::from_string("chi2_english_gp_v0").value() == ScoreId::chi2_english_gp_v0());
    REQUIRE(ScoreId::from_string("self_repeat_rate").value() == ScoreId::self_repeat_rate());
    REQUIRE_FALSE(ScoreId::from_string("nope").ok());
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::exact_match()) == ScoreOrder::Desc);
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::hamming_agreement()) == ScoreOrder::Desc);
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::ic_mod29()) == ScoreOrder::Desc);
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::chi2_english_gp_v0()) == ScoreOrder::Asc);
    // Spec: neither assumed globally — raw report only (Asc used as neutral default).
    REQUIRE(ScoreOrderUtil::for_score_id(ScoreId::self_repeat_rate()) == ScoreOrder::Asc);
}

TEST_CASE("ScoreRegistry catalogs Tier A ids for tool API", "[score][registry]") {
    const std::vector<ScoreCatalogEntry> entries = ScoreRegistry::catalog();
    REQUIRE(entries.size() == 5);

    const std::vector<std::string> ids = ScoreRegistry::known_ids();
    REQUIRE(ids == std::vector<std::string>{
        "exact_match",
        "hamming_agreement",
        "ic_mod29",
        "chi2_english_gp_v0",
        "self_repeat_rate",
    });

    REQUIRE(ScoreRegistry::is_known("ic_mod29"));
    REQUIRE_FALSE(ScoreRegistry::is_known("nope"));

    REQUIRE(ScoreRegistry::order_of("ic_mod29").value() == ScoreOrder::Desc);
    REQUIRE(ScoreRegistry::order_of("chi2_english_gp_v0").value() == ScoreOrder::Asc);
    REQUIRE_FALSE(ScoreRegistry::order_of("nope").ok());

    REQUIRE(entries[0].arity() == ScoreCatalogEntry::Arity::Pairwise);
    REQUIRE(entries[2].arity() == ScoreCatalogEntry::Arity::Unary);
    REQUIRE(entries[3].arity() == ScoreCatalogEntry::Arity::UnaryWithTable);
    REQUIRE(ScoreCatalogEntry::arity_string(entries[3].arity()) == "unary_with_table");
}

TEST_CASE("ScoreRegistry dispatches by string id", "[score][registry]") {
    const std::vector<Index29> xs = {I(0), I(0), I(1), I(1)};

    SECTION("rejects unknown id and version") {
        REQUIRE_FALSE(ScoreRegistry::score("nope", xs).ok());
        REQUIRE_FALSE(ScoreRegistry::score("ic_mod29", xs, "v1").ok());
    }

    SECTION("ic_mod29 and self_repeat_rate unary") {
        StatusOr<double> ic = ScoreRegistry::score("ic_mod29", xs);
        REQUIRE(ic.ok());
        REQUIRE(ic.value() == Catch::Approx(IcMod29::score(xs).value()).margin(0.0));

        StatusOr<double> rep = ScoreRegistry::score("self_repeat_rate", xs);
        REQUIRE(rep.ok());
        REQUIRE(rep.value() == Catch::Approx(SelfRepeatRate::score(xs).value()).margin(0.0));
    }

    SECTION("exact_match via ScoreRequest.reference") {
        ScoreRequest req;
        req.reference = std::span<const Index29>(xs);
        StatusOr<double> hit = ScoreRegistry::score("exact_match", xs, "v0", {}, req);
        REQUIRE(hit.ok());
        REQUIRE(hit.value() == 1.0);

        const std::vector<Index29> other = {I(0), I(0), I(1), I(2)};
        req.reference = std::span<const Index29>(other);
        StatusOr<double> miss = ScoreRegistry::score("exact_match", xs, "v0", {}, req);
        REQUIRE(miss.ok());
        REQUIRE(miss.value() == 0.0);
    }

    SECTION("hamming_agreement via params.reference JSON") {
        const nlohmann::json params = {{"reference", {0, 0, 1, 1}}};
        StatusOr<double> full = ScoreRegistry::score("hamming_agreement", xs, "v0", params);
        REQUIRE(full.ok());
        REQUIRE(full.value() == Catch::Approx(1.0).margin(0.0));

        const nlohmann::json partial = {{"reference", {0, 0, 1, 9}}};
        StatusOr<double> three = ScoreRegistry::score("hamming_agreement", xs, "v0", partial);
        REQUIRE(three.ok());
        REQUIRE(three.value() == Catch::Approx(0.75).epsilon(1e-15));

        REQUIRE_FALSE(ScoreRegistry::score("hamming_agreement", xs).ok());
    }

    SECTION("chi2_english_gp_v0 requires expected table") {
        REQUIRE_FALSE(ScoreRegistry::score("chi2_english_gp_v0", xs).ok());

        StatusOr<ExpectedFrequencyTable> table = ExpectedFrequencyLoader::load_from_file(
            std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
        REQUIRE(table.ok());

        ScoreRequest req;
        req.expected_frequencies = &table.value();
        StatusOr<double> via_registry =
            ScoreRegistry::score("chi2_english_gp_v0", xs, "v0", {}, req);
        StatusOr<double> direct = Chi2EnglishGp::score(xs, table.value());
        REQUIRE(via_registry.ok());
        REQUIRE(direct.ok());
        REQUIRE(via_registry.value() == Catch::Approx(direct.value()).margin(0.0));
    }
}

TEST_CASE("ScoreRegistry catalog entries expose order and arity JSON", "[score][catalog]") {
    const std::vector<ScoreCatalogEntry> entries = ScoreRegistry::catalog();
    REQUIRE(entries.size() == 5);

    bool saw_chi2 = false;
    bool saw_exact = false;
    for (const ScoreCatalogEntry& entry : entries) {
        const nlohmann::json row = entry.to_json();
        REQUIRE(row.contains("score_id"));
        REQUIRE(row.contains("score_version"));
        REQUIRE(row.contains("order"));
        REQUIRE(row.contains("arity"));
        REQUIRE(row.at("score_version").get<std::string>() == "v0");
        if (entry.id() == "chi2_english_gp_v0") {
            saw_chi2 = true;
            REQUIRE(row.at("order").get<std::string>() == "asc");
            REQUIRE(row.at("arity").get<std::string>() == "unary_with_table");
        }
        if (entry.id() == "exact_match") {
            saw_exact = true;
            REQUIRE(row.at("order").get<std::string>() == "desc");
            REQUIRE(row.at("arity").get<std::string>() == "pairwise");
        }
    }
    REQUIRE(saw_chi2);
    REQUIRE(saw_exact);
}

TEST_CASE("ExactMatch hand vectors", "[score][exact]") {
    SECTION("identical sequences → 1") {
        const std::vector<Index29> xs = {I(1), I(2), I(3)};
        StatusOr<double> s = ExactMatch::score(xs, xs);
        REQUIRE(s.ok());
        REQUIRE(s.value() == 1.0);
    }

    SECTION("empty ≡ empty → 1") {
        StatusOr<double> s = ExactMatch::score({}, {});
        REQUIRE(s.ok());
        REQUIRE(s.value() == 1.0);
    }

    SECTION("length mismatch → 0") {
        StatusOr<double> s = ExactMatch::score({I(0)}, {I(0), I(1)});
        REQUIRE(s.ok());
        REQUIRE(s.value() == 0.0);
    }

    SECTION("same length, one mismatch → 0") {
        StatusOr<double> s = ExactMatch::score({I(0), I(1), I(2)}, {I(0), I(9), I(2)});
        REQUIRE(s.ok());
        REQUIRE(s.value() == 0.0);
    }
}

TEST_CASE("HammingAgreement hand vectors", "[score][hamming]") {
    SECTION("rejects empty and length mismatch") {
        REQUIRE_FALSE(HammingAgreement::score({}, {}).ok());
        REQUIRE_FALSE(HammingAgreement::score({I(0)}, {I(0), I(1)}).ok());
    }

    SECTION("identical → 1") {
        const std::vector<Index29> xs = {I(4), I(5), I(6), I(7)};
        StatusOr<double> s = HammingAgreement::score(xs, xs);
        REQUIRE(s.ok());
        REQUIRE(s.value() == Catch::Approx(1.0).margin(0.0));
    }

    SECTION("three of four match → 0.75") {
        StatusOr<double> s =
            HammingAgreement::score({I(0), I(1), I(2), I(3)}, {I(0), I(1), I(9), I(3)});
        REQUIRE(s.ok());
        REQUIRE(s.value() == Catch::Approx(0.75).epsilon(1e-15));
    }

    SECTION("no matches → 0") {
        StatusOr<double> s = HammingAgreement::score({I(0), I(1)}, {I(2), I(3)});
        REQUIRE(s.ok());
        REQUIRE(s.value() == Catch::Approx(0.0).margin(0.0));
    }
}

TEST_CASE("ExactMatch and HammingAgreement on fixture plaintext", "[score][exact][hamming]") {
    const std::vector<Index29> plain = plaintext_indices_of("a-warning");
    REQUIRE_FALSE(plain.empty());

    StatusOr<double> exact = ExactMatch::score(plain, plain);
    REQUIRE(exact.ok());
    REQUIRE(exact.value() == 1.0);

    StatusOr<double> hamm = HammingAgreement::score(plain, plain);
    REQUIRE(hamm.ok());
    REQUIRE(hamm.value() == Catch::Approx(1.0).margin(0.0));

    std::vector<Index29> flipped = plain;
    flipped[0] = Index29{static_cast<std::uint8_t>((flipped[0].value() + 1) % 29)};
    REQUIRE(ExactMatch::score(flipped, plain).value() == 0.0);
    StatusOr<double> partial = HammingAgreement::score(flipped, plain);
    REQUIRE(partial.ok());
    REQUIRE(partial.value() == Catch::Approx(
        static_cast<double>(plain.size() - 1) / static_cast<double>(plain.size()))
                                    .epsilon(1e-12));
}

TEST_CASE("Scores on plaintext vs random Index29 noise separate cleanly", "[score][noise]") {
    const std::vector<Index29> plain = plaintext_indices_of("welcome");
    REQUIRE(plain.size() >= 64);

    const std::vector<Index29> noise = lcg_noise(plain.size(), /*seed=*/0xC1CADAu);
    REQUIRE(noise.size() == plain.size());
    // Sanity: noise is not accidentally identical to plaintext.
    REQUIRE(ExactMatch::score(noise, plain).value() == 0.0);

    StatusOr<ExpectedFrequencyTable> table = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(table.ok());

    // Pairwise / fixture scores
    REQUIRE(ExactMatch::score(plain, plain).value() == 1.0);
    REQUIRE(ExactMatch::score(noise, plain).value() == 0.0);

    StatusOr<double> hamm_plain = HammingAgreement::score(plain, plain);
    StatusOr<double> hamm_noise = HammingAgreement::score(noise, plain);
    REQUIRE(hamm_plain.ok());
    REQUIRE(hamm_noise.ok());
    REQUIRE(hamm_plain.value() == Catch::Approx(1.0).margin(0.0));
    // Chance agreement ≈ 1/29; require a clear gap below language identity.
    REQUIRE(hamm_noise.value() < 0.15);
    REQUIRE(hamm_noise.value() < hamm_plain.value());

    // Univariate language scores: plaintext beats flat noise.
    StatusOr<double> ic_plain = IcMod29::score(plain);
    StatusOr<double> ic_noise = IcMod29::score(noise);
    REQUIRE(ic_plain.ok());
    REQUIRE(ic_noise.ok());
    REQUIRE(ic_plain.value() > ic_noise.value());
    REQUIRE(ic_plain.value() > 1.0 / 29.0);

    StatusOr<double> chi_plain = Chi2EnglishGp::score(plain, table.value());
    StatusOr<double> chi_noise = Chi2EnglishGp::score(noise, table.value());
    REQUIRE(chi_plain.ok());
    REQUIRE(chi_noise.ok());
    REQUIRE(chi_plain.value() < chi_noise.value());
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
