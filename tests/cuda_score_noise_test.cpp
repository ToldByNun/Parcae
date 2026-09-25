#include "parcae/core/index29.hpp"
#include "parcae/corpus/fixture.hpp"
#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/gematria/gematria_profile_loader.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/validate/plaintext_normalizer.hpp"

#include "cuda_score.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

/// Helpers for CUDA score suite / noise separation (class style — no namespaces).
class CudaScoreNoiseSuite {
public:
    [[nodiscard]] static GematriaProfile load_profile() {
        StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_file(
            std::string(PARCAE_TEST_DATA_DIR) + "/profiles/gematria/gematria-primus-v0.json");
        REQUIRE(profile.ok());
        return profile.value();
    }

    [[nodiscard]] static std::vector<Index29> plaintext_indices_of(const std::string& fixture_id) {
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

    /// Deterministic Index29 noise (LCG). Same length as `n`; independent of plaintext.
    [[nodiscard]] static std::vector<Index29> lcg_noise(std::size_t n, std::uint64_t seed) {
        std::vector<Index29> out;
        out.reserve(n);
        std::uint64_t state = seed;
        for (std::size_t i = 0; i < n; ++i) {
            state = state * 6364136223846793005ULL + 1ULL;
            out.push_back(Index29{static_cast<std::uint8_t>((state >> 33) % 29)});
        }
        return out;
    }

private:
    CudaScoreNoiseSuite() = delete;
};

TEST_CASE("CUDA score suite: CudaScore catalogs registry score ids", "[cuda][score][suite]") {
    const auto ids = CudaScore::known_ids();
    REQUIRE(ids.size() == 6);
    REQUIRE(CudaScore::is_known("exact_match"));
    REQUIRE(CudaScore::is_known("hamming_agreement"));
    REQUIRE(CudaScore::is_known("ic_mod29"));
    REQUIRE(CudaScore::is_known("self_repeat_rate"));
    REQUIRE(CudaScore::is_known("chi2_english_gp_v0"));
    REQUIRE(CudaScore::is_known("log_bigram_gp_v0"));
    REQUIRE(CudaScore::catalog().size() == ScoreRegistry::catalog().size());
}

#if defined(PARCAE_HAS_CUDA)

TEST_CASE("CUDA scores on plaintext vs random Index29 noise separate cleanly",
          "[cuda][score][noise][suite]") {
    if (!CudaScore::available()) {
        SUCCEED("CUDA linked but device unavailable; skipping noise separation");
        return;
    }

    const std::vector<Index29> plain = CudaScoreNoiseSuite::plaintext_indices_of("welcome");
    REQUIRE(plain.size() >= 64);

    const std::vector<Index29> noise =
        CudaScoreNoiseSuite::lcg_noise(plain.size(), /*seed=*/0xC1CADAu);
    REQUIRE(noise.size() == plain.size());

    StatusOr<ExpectedFrequencyTable> table = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(table.ok());

    ScoreRequest pairwise_plain;
    pairwise_plain.reference = std::span<const Index29>(plain.data(), plain.size());

    ScoreRequest chi2_req;
    chi2_req.expected_frequencies = &table.value();

    // Pairwise / fixture scores via CudaScore dispatch
    StatusOr<double> exact_pp =
        CudaScore::score("exact_match", plain, "v0", nlohmann::json::object(), pairwise_plain);
    StatusOr<double> exact_np =
        CudaScore::score("exact_match", noise, "v0", nlohmann::json::object(), pairwise_plain);
    REQUIRE(exact_pp.ok());
    REQUIRE(exact_np.ok());
    REQUIRE(exact_pp.value() == 1.0);
    REQUIRE(exact_np.value() == 0.0);

    StatusOr<double> hamm_plain = CudaScore::score("hamming_agreement", plain, "v0",
                                                   nlohmann::json::object(), pairwise_plain);
    StatusOr<double> hamm_noise = CudaScore::score("hamming_agreement", noise, "v0",
                                                   nlohmann::json::object(), pairwise_plain);
    REQUIRE(hamm_plain.ok());
    REQUIRE(hamm_noise.ok());
    REQUIRE(hamm_plain.value() == Catch::Approx(1.0).margin(0.0));
    REQUIRE(hamm_noise.value() < 0.15);
    REQUIRE(hamm_noise.value() < hamm_plain.value());

    // Univariate language scores: plaintext beats flat noise.
    StatusOr<double> ic_plain = CudaScore::score("ic_mod29", plain);
    StatusOr<double> ic_noise = CudaScore::score("ic_mod29", noise);
    REQUIRE(ic_plain.ok());
    REQUIRE(ic_noise.ok());
    REQUIRE(ic_plain.value() > ic_noise.value());
    REQUIRE(ic_plain.value() > 1.0 / 29.0);

    StatusOr<double> chi_plain =
        CudaScore::score("chi2_english_gp_v0", plain, "v0", nlohmann::json::object(), chi2_req);
    StatusOr<double> chi_noise =
        CudaScore::score("chi2_english_gp_v0", noise, "v0", nlohmann::json::object(), chi2_req);
    REQUIRE(chi_plain.ok());
    REQUIRE(chi_noise.ok());
    REQUIRE(chi_plain.value() < chi_noise.value());

    // Bit-identical to CPU ScoreRegistry on the same streams (suite gate).
    REQUIRE(CudaScore::score("ic_mod29", plain).value() ==
            ScoreRegistry::score("ic_mod29", plain).value());
    REQUIRE(CudaScore::score("chi2_english_gp_v0", plain, "v0", nlohmann::json::object(), chi2_req)
                .value() == ScoreRegistry::score("chi2_english_gp_v0", plain, "v0",
                                                 nlohmann::json::object(), chi2_req)
                                .value());
    REQUIRE(
        CudaScore::score("hamming_agreement", noise, "v0", nlohmann::json::object(), pairwise_plain)
            .value() == ScoreRegistry::score("hamming_agreement", noise, "v0",
                                             nlohmann::json::object(), pairwise_plain)
                            .value());
}

#else

TEST_CASE("CUDA score noise separation skipped (PARCAE_HAS_CUDA unset)",
          "[cuda][score][noise][suite]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CudaScore noise separation");
}

#endif
