#include <catch2/catch_test_macros.hpp>

#include "parcae/run/search_run.hpp"
#include "parcae/run/search_run_console.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("SearchRun CPU caesar sweep + fixture eval", "[run][search]") {
    const parcae::tool::Context ctx{PARCAE_TEST_DATA_DIR};

    SearchRun::Options options;
    options.backend = parcae::tool::Backend::Cpu;
    options.family = "caesar";
    options.seed = 2109016688u;
    options.stream_length = 256;
    options.throughput_repeats = 2;
    options.compare_cpu_cuda = false;

    StatusOr<SearchRunMetrics> metrics = SearchRun::run(ctx, options);
    REQUIRE(metrics.ok());
    REQUIRE(metrics.value().transform_id() == "caesar");
    REQUIRE(metrics.value().seed() == 2109016688u);
    REQUIRE(metrics.value().steps().size() == 29);
    REQUIRE(metrics.value().tok_per_sec() > 0.0);
    REQUIRE(metrics.value().eval_total() == 9);
    REQUIRE(metrics.value().eval_passed() == 9);
    REQUIRE(metrics.value().eval_set_pass_rate() == 1.0);
    REQUIRE_FALSE(metrics.value().cpu_cuda_pass().has_value());

    REQUIRE(metrics.value().steps()[0].params().at("shift").get<int>() == 0);
    REQUIRE(metrics.value().steps()[11].params().at("shift").get<int>() == 11);
    REQUIRE(metrics.value().steps()[11].param_hash().size() == 64);

    const nlohmann::json with_timing = metrics.value().to_json(false);
    REQUIRE(with_timing.contains("tok_per_sec"));
    REQUIRE(with_timing.at("steps").at(3).at("params").at("shift").get<int>() == 3);

    const nlohmann::json omit = metrics.value().to_json(true);
    REQUIRE_FALSE(omit.contains("tok_per_sec"));
    REQUIRE(omit.at("steps").size() == 29);
    REQUIRE(omit.at("seed").get<std::uint32_t>() == 2109016688u);

    const std::string report = SearchRunConsole::format(metrics.value());
    REQUIRE(report.find("PARCAE - SEARCH RUN") != std::string::npos);
    REQUIRE(report.find("Transform:      Caesar") != std::string::npos);
    REQUIRE(report.find("Parameters:     shift 0-28") != std::string::npos);
    REQUIRE(report.find("Seed:           2109016688") != std::string::npos);
    REQUIRE(report.find("Throughput") != std::string::npos);
    REQUIRE(report.find('#') != std::string::npos);
    REQUIRE(report.find("Fixture Eval    9 / 9") != std::string::npos);
}

#if defined(PARCAE_HAS_CUDA)
TEST_CASE("SearchRun CUDA caesar sweep + CPU↔CUDA parity", "[run][search][cuda]") {
    const parcae::tool::Context ctx{PARCAE_TEST_DATA_DIR};

    SearchRun::Options options;
    options.backend = parcae::tool::Backend::Cuda;
    options.family = "caesar";
    options.seed = 2109016688u;
    options.stream_length = 256;
    options.throughput_repeats = 2;
    options.compare_cpu_cuda = true;

    StatusOr<SearchRunMetrics> metrics = SearchRun::run(ctx, options);
    REQUIRE(metrics.ok());
    REQUIRE(metrics.value().backend() == "cuda");
    REQUIRE(metrics.value().steps().size() == 29);
    REQUIRE(metrics.value().tok_per_sec() > 0.0);
    REQUIRE(metrics.value().eval_passed() == 9);
    REQUIRE(metrics.value().cpu_cuda_pass().has_value());
    REQUIRE(metrics.value().cpu_cuda_pass().value());

    const std::string report = SearchRunConsole::format(metrics.value());
    REQUIRE(report.find("PARCAE - CUDA SEARCH RUN") != std::string::npos);
    REQUIRE(report.find("CPU <-> CUDA      PASS") != std::string::npos);
}

TEST_CASE("SearchRun CUDA all families parity", "[run][search][cuda][families]") {
    const parcae::tool::Context ctx{PARCAE_TEST_DATA_DIR};
    const std::vector<std::string> families = {
        "caesar", "atbash", "atbash_caesar", "affine", "vigenere"};
    for (const std::string& family : families) {
        SearchRun::Options options;
        options.backend = parcae::tool::Backend::Cuda;
        options.family = family;
        options.seed = 42u;
        options.stream_length = 128;
        options.throughput_repeats = 1;
        options.compare_cpu_cuda = true;
        StatusOr<SearchRunMetrics> metrics = SearchRun::run(ctx, options);
        INFO(family);
        REQUIRE(metrics.ok());
        REQUIRE(metrics.value().tok_per_sec() > 0.0);
        REQUIRE(metrics.value().eval_passed() == 9);
        REQUIRE(metrics.value().cpu_cuda_pass().has_value());
        REQUIRE(metrics.value().cpu_cuda_pass().value());
    }
}
#endif
