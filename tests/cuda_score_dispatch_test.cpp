#include "cuda_score.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

static bool message_contains(const Status& status, const char* fragment) {
    return status.message().find(fragment) != std::string::npos;
}

TEST_CASE("CudaScore catalog mirrors ScoreRegistry", "[cuda][score][dispatch]") {
    REQUIRE(CudaScore::catalog().size() == ScoreRegistry::catalog().size());
    REQUIRE(CudaScore::known_ids() == ScoreRegistry::known_ids());
    REQUIRE(CudaScore::is_known("ic_mod29"));
    REQUIRE_FALSE(CudaScore::is_known("not_a_score"));

    StatusOr<ScoreOrder> order = CudaScore::order_of("chi2_english_gp_v0");
    REQUIRE(order.ok());
    REQUIRE(order.value() == ScoreOrder::Asc);
}

TEST_CASE("CudaScore rejects bad version and unknown id", "[cuda][score][dispatch]") {
    const std::vector<Index29> xs{Index29{1}, Index29{2}};

    StatusOr<double> bad_ver = CudaScore::score("ic_mod29", xs, "v1");
    REQUIRE_FALSE(bad_ver.ok());
    REQUIRE(message_contains(bad_ver.status(), "score_version"));

    StatusOr<double> unknown = CudaScore::score("nope", xs);
    REQUIRE_FALSE(unknown.ok());
}

#if defined(PARCAE_HAS_CUDA)

TEST_CASE("CudaScore dispatch matches ScoreRegistry for all Tier A ids", "[cuda][score][dispatch]") {
    if (!CudaScore::available()) {
        SUCCEED("CUDA linked but device unavailable; skipping dispatch apply");
        return;
    }

    std::mt19937 rng(0x00D15A71u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> candidate;
    std::vector<Index29> reference;
    candidate.reserve(64);
    reference.reserve(64);
    for (std::size_t i = 0; i < 64; ++i) {
        candidate.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        reference.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }

    StatusOr<ExpectedFrequencyTable> table = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(table.ok());

    ScoreRequest pairwise;
    pairwise.reference = std::span<const Index29>(reference.data(), reference.size());

    ScoreRequest chi2_req;
    chi2_req.expected_frequencies = &table.value();

    const auto check = [&](const char* id, const ScoreRequest& req = {}) {
        StatusOr<double> cpu = ScoreRegistry::score(id, candidate, "v0", nlohmann::json::object(), req);
        StatusOr<double> cuda = CudaScore::score(id, candidate, "v0", nlohmann::json::object(), req);
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        REQUIRE(cuda.value() == cpu.value());
    };

    check("ic_mod29");
    check("self_repeat_rate");
    check("chi2_english_gp_v0", chi2_req);
    check("exact_match", pairwise);
    check("hamming_agreement", pairwise);
}

TEST_CASE("CudaScore pairwise via params.reference JSON", "[cuda][score][dispatch]") {
    if (!CudaScore::available()) {
        SUCCEED("CUDA linked but device unavailable; skipping params.reference path");
        return;
    }

    const std::vector<Index29> cand{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
    const nlohmann::json params{{"reference", {0, 9, 2, 8}}};

    StatusOr<double> cpu =
        ScoreRegistry::score("hamming_agreement", cand, "v0", params);
    StatusOr<double> cuda = CudaScore::score("hamming_agreement", cand, "v0", params);
    REQUIRE(cpu.ok());
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value() == cpu.value());
    REQUIRE(cuda.value() == 0.5);
}

#else

TEST_CASE("CudaScore reports unavailable without PARCAE_HAS_CUDA", "[cuda][score][dispatch]") {
    REQUIRE_FALSE(CudaScore::available());
    const std::vector<Index29> xs{Index29{1}, Index29{2}};
    StatusOr<double> scored = CudaScore::score("ic_mod29", xs);
    REQUIRE_FALSE(scored.ok());
    REQUIRE(message_contains(scored.status(), "not available"));
}

#endif
