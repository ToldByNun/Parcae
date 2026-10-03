#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/batch/batch_runner.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/theory_export_cache.hpp>
#include <parcae/search/theory_export_pipeline.hpp>
#include <parcae/transform/transform_direction.hpp>

#include "cuda_stream_pair.hpp"
#include "parcae_cuda.hpp"
#include "theory_device_scratch.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif
#ifndef PARCAE_EXAMPLES_DIR
#error "PARCAE_EXAMPLES_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_EXE
#error "PARCAE_PYTHON_EXE must be defined"
#endif
#ifndef PARCAE_PYTHON_DIR
#error "PARCAE_PYTHON_DIR must be defined"
#endif

namespace {

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

struct PipelineFixture {
    std::filesystem::path root;
    std::filesystem::path theories;
    std::string uri;
    std::vector<Index29> cipher;
    std::vector<nlohmann::json> params_a;
    std::vector<nlohmann::json> params_b;
    ExpectedFrequencyTable freqs;
};

[[nodiscard]] PipelineFixture make_fixture() {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_theory_export_pipeline";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const auto src = std::filesystem::path(PARCAE_EXAMPLES_DIR) / "new_math_example.py";
    StatusOr<DslCompile::Result> compiled =
        DslCompile::compile_file(src, root / "theories", compile_options());
    REQUIRE(compiled.ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    return PipelineFixture{
        root,
        root / "theories",
        "parcae://theories/quadratic_polynomial_stream@1",
        {Index29{3}, Index29{10}, Index29{28}, Index29{0}, Index29{7}, Index29{15}, Index29{22},
         Index29{4}, Index29{11}, Index29{19}},
        {
            nlohmann::json{{"c2", 1}, {"c1", 0}, {"c0", 0}},
            nlohmann::json{{"c2", 0}, {"c1", 1}, {"c0", 0}},
            nlohmann::json{{"c2", 2}, {"c1", 3}, {"c0", 5}},
        },
        {
            nlohmann::json{{"c2", 0}, {"c1", 0}, {"c0", 14}},
            nlohmann::json{{"c2", 4}, {"c1", 7}, {"c0", 1}},
        },
        std::move(freqs.value()),
    };
}

} // namespace

TEST_CASE("TheoryExportPipeline ping-pong matches sequential scores",
          "[search][export][theory][pipeline][cuda]") {
    REQUIRE(ParcaeCuda::available());
    PipelineFixture fx = make_fixture();
    TheoryExportCache cache;
    TheoryDeviceScratch scratch;
    CudaStreamPair streams = CudaStreamPair::create_or_legacy();
    REQUIRE(streams.uses_dedicated_streams());

    TheoryExportPipeline pipe(cache, scratch, streams);
    const std::vector<std::vector<nlohmann::json>> chunks{fx.params_a, fx.params_b};
    StatusOr<std::vector<std::vector<double>>> pipelined =
        pipe.run_chunks(fx.cipher, fx.freqs, fx.theories, fx.uri, chunks);
    REQUIRE(pipelined.ok());
    REQUIRE(pipelined.value().size() == 2);
    REQUIRE(scratch.cipher_upload_count() == 1);
    REQUIRE(scratch.probs_upload_count() == 1);

    StatusOr<std::vector<double>> seq_a = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_a);
    StatusOr<std::vector<double>> seq_b = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_b);
    REQUIRE(seq_a.ok());
    REQUIRE(seq_b.ok());
    REQUIRE(pipelined.value()[0].size() == seq_a.value().size());
    REQUIRE(pipelined.value()[1].size() == seq_b.value().size());
    for (std::size_t i = 0; i < seq_a.value().size(); ++i) {
        REQUIRE(std::abs(pipelined.value()[0][i] - seq_a.value()[i]) < 1e-12);
    }
    for (std::size_t i = 0; i < seq_b.value().size(); ++i) {
        REQUIRE(std::abs(pipelined.value()[1][i] - seq_b.value()[i]) < 1e-12);
    }

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

#else

TEST_CASE("TheoryExportPipeline skipped (PARCAE_HAS_CUDA unset)",
          "[search][export][theory][pipeline]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise TheoryExportPipeline");
}

#endif
