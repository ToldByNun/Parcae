#include "parcae/core/index29.hpp"
#include "parcae/score/bigram_model_loader.hpp"
#include "parcae/score/log_bigram_gp.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"

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

#if defined(PARCAE_HAS_CUDA)

#include "log_bigram_gp_score.hpp"

TEST_CASE("LogBigramGpScore host twin matches CPU LogBigramGp", "[cuda][score][bigram]") {
    StatusOr<BigramModelTable> table = BigramModelLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-bigram-v0.json");
    REQUIRE(table.ok());

    const std::vector<Index29> xs{Index29{1}, Index29{2}, Index29{4}, Index29{0}};
    std::vector<std::uint8_t> bytes;
    bytes.reserve(xs.size());
    for (const Index29 idx : xs) {
        bytes.push_back(idx.value());
    }

    StatusOr<double> cpu = LogBigramGp::score(xs, table.value());
    StatusOr<double> twin = LogBigramGpScore::score_host(bytes, table.value().log_probs());
    REQUIRE(cpu.ok());
    REQUIRE(twin.ok());
    REQUIRE(twin.value() == Catch::Approx(cpu.value()).margin(0.0));

    ScoreRequest req;
    req.bigram_model = &table.value();
    if (CudaScore::available()) {
        StatusOr<double> via_dispatch =
            CudaScore::score("log_bigram_gp_v0", xs, "v0", nlohmann::json::object(), req);
        REQUIRE(via_dispatch.ok());
        REQUIRE(via_dispatch.value() == Catch::Approx(cpu.value()).margin(0.0));
        REQUIRE(ScoreRegistry::score("log_bigram_gp_v0", xs, "v0", nlohmann::json::object(), req)
                    .value() == via_dispatch.value());
    }

    REQUIRE_FALSE(LogBigramGpScore::score_host({}, table.value().log_probs()).ok());
    REQUIRE_FALSE(
        LogBigramGpScore::score_host(std::vector<std::uint8_t>{0}, table.value().log_probs())
            .ok());
}

#else

TEST_CASE("LogBigramGpScore twin skipped without PARCAE_HAS_CUDA", "[cuda][score][bigram]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise LogBigramGpScore");
}

#endif
