#include <catch2/catch_test_macros.hpp>

#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/generate/theory_explicit_params_candidate_generator.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/search/cpu_candidate_export.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>

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

struct TheoryParityFixture {
    std::filesystem::path root;
    std::filesystem::path theories;
    std::string uri;
    std::vector<Index29> cipher;
    std::vector<nlohmann::json> params_list;
    ExpectedFrequencyTable freqs;

    TheoryParityFixture(std::filesystem::path root_in, std::filesystem::path theories_in,
                        std::string uri_in, std::vector<Index29> cipher_in,
                        std::vector<nlohmann::json> params_in, ExpectedFrequencyTable freqs_in)
        : root(std::move(root_in)), theories(std::move(theories_in)), uri(std::move(uri_in)),
          cipher(std::move(cipher_in)), params_list(std::move(params_in)),
          freqs(std::move(freqs_in)) {}
};

[[nodiscard]] TheoryParityFixture make_fixture() {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const auto root =
        std::filesystem::temp_directory_path() / "parcae_theory_cpu_gpu_topk_parity";
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

    std::vector<Index29> cipher;
    cipher.reserve(96);
    for (std::uint8_t i = 0; i < 96; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 3u) % 29u)});
    }

    std::vector<nlohmann::json> params_list{
        nlohmann::json{{"c2", 1}, {"c1", 0}, {"c0", 0}},
        nlohmann::json{{"c2", 0}, {"c1", 1}, {"c0", 0}},
        nlohmann::json{{"c2", 2}, {"c1", 3}, {"c0", 5}},
        nlohmann::json{{"c2", 0}, {"c1", 0}, {"c0", 14}},
        nlohmann::json{{"c2", 4}, {"c1", 7}, {"c0", 1}},
        nlohmann::json{{"c2", 3}, {"c1", 11}, {"c0", 8}},
        nlohmann::json{{"c2", 5}, {"c1", 2}, {"c0", 9}},
        nlohmann::json{{"c2", 1}, {"c1", 1}, {"c0", 1}},
    };

    return TheoryParityFixture{
        root,
        root / "theories",
        "parcae://theories/quadratic_polynomial_stream@1",
        std::move(cipher),
        std::move(params_list),
        std::move(freqs.value()),
    };
}

void require_same_topk(const GpuCandidateExport::Result& cpu,
                       const GpuCandidateExport::Result& gpu) {
    REQUIRE(cpu.size() == gpu.size());
    REQUIRE(cpu.backend() == Backend::Cpu);
    REQUIRE(gpu.backend() == Backend::Cuda);
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        REQUIRE(cpu.rows()[i].source_index() == gpu.rows()[i].source_index());
        REQUIRE(cpu.rows()[i].candidate().candidate_id() ==
                gpu.rows()[i].candidate().candidate_id());
        REQUIRE(cpu.rows()[i].score() == gpu.rows()[i].score());
        REQUIRE(cpu.rows()[i].candidate().params() == gpu.rows()[i].candidate().params());
        REQUIRE(cpu.rows()[i].candidate().output_indices() ==
                gpu.rows()[i].candidate().output_indices());
        REQUIRE(cpu.rows()[i].rank() == gpu.rows()[i].rank());
        REQUIRE(cpu.rows()[i].candidate().transform_id().str() ==
                gpu.rows()[i].candidate().transform_id().str());
    }
}

} // namespace

TEST_CASE("has_fused_cuda_chi2_export(theory) is true after fused land",
          "[search][export][theory][parity]") {
    REQUIRE(SearchJob::has_fused_cuda_chi2_export("theory"));
    REQUIRE_FALSE(SearchJob::is_cpu_export_only_family("theory"));
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"

TEST_CASE("CPU host-scores vs CUDA theory_explicit_params top-k χ² order parity",
          "[search][export][theory][parity][cuda]") {
    REQUIRE(ParcaeCuda::available());
    TheoryParityFixture fx = make_fixture();
    constexpr std::size_t k = 4;

    // CPU oracle path: generator + host χ² scores → theory_from_host_scores.
    StatusOr<std::vector<TransformCandidate>> generated =
        TheoryExplicitParamsCandidateGenerator::generate(fx.cipher, fx.theories, fx.uri,
                                                         fx.params_list,
                                                         TransformDirection::Decrypt);
    REQUIRE(generated.ok());
    REQUIRE(generated.value().size() == fx.params_list.size());

    std::vector<double> host_scores;
    host_scores.reserve(generated.value().size());
    for (const TransformCandidate& c : generated.value()) {
        StatusOr<double> s = Chi2EnglishGp::score(c.output_indices(), fx.freqs);
        REQUIRE(s.ok());
        host_scores.push_back(s.value());
    }

    StatusOr<GpuCandidateExport::Result> cpu = GpuCandidateExport::theory_from_host_scores(
        fx.cipher, fx.theories, fx.uri, fx.params_list, host_scores, k, TransformDirection::Decrypt,
        Backend::Cpu);
    REQUIRE(cpu.ok());

    StatusOr<GpuCandidateExport::Result> gpu = GpuCandidateExport::theory_explicit_params(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, k);
    REQUIRE(gpu.ok());

    require_same_topk(cpu.value(), gpu.value());

    // Ascending χ² order (ScoreOrder::Asc).
    for (std::size_t i = 1; i < gpu.value().size(); ++i) {
        REQUIRE(gpu.value().rows()[i - 1].score() <= gpu.value().rows()[i].score());
    }

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

TEST_CASE("CpuCandidateExport theory job vs GpuCandidateExport fused top-k ids",
          "[search][export][theory][parity][cuda]") {
    REQUIRE(ParcaeCuda::available());
    TheoryParityFixture fx = make_fixture();

    // Mirror CpuCandidateExport sandbox: data_root with theories/ + profiles/.
    std::error_code ec;
    std::filesystem::create_directories(fx.root / "profiles" / "scores", ec);
    std::filesystem::copy_file(
        std::filesystem::path(PARCAE_TEST_DATA_DIR) / "profiles" / "scores" /
            "english-gp-expected-v0.json",
        fx.root / "profiles" / "scores" / "english-gp-expected-v0.json",
        std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE_FALSE(ec);

    const nlohmann::json param_grid{{"theory_uri", fx.uri},
                                    {"params_list", fx.params_list}};
    StatusOr<SearchJob> job =
        SearchJob::make("_example", "theory", "chi2_english_gp_v0", /*k=*/3, /*seed=*/1,
                        Backend::Cpu, /*max_candidates=*/64, TransformDirection::Decrypt,
                        param_grid, std::nullopt, "v0", false, true);
    REQUIRE(job.ok());

    const Context ctx{fx.root};
    StatusOr<CpuCandidateExport::Result> cpu =
        CpuCandidateExport::from_job(fx.cipher, job.value(), ctx);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().size() == 3);

    StatusOr<GpuCandidateExport::Result> gpu = GpuCandidateExport::theory_explicit_params(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, 3);
    REQUIRE(gpu.ok());
    REQUIRE(gpu.value().size() == 3);

    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE(cpu.value().rows()[i].candidate().candidate_id() ==
                gpu.value().rows()[i].candidate().candidate_id());
        REQUIRE(cpu.value().rows()[i].score() == gpu.value().rows()[i].score());
        REQUIRE(cpu.value().rows()[i].candidate().output_indices() ==
                gpu.value().rows()[i].candidate().output_indices());
    }

    std::filesystem::remove_all(fx.root, ec);
}

#else

TEST_CASE("theory CPU/GPU top-k parity skipped (PARCAE_HAS_CUDA unset)",
          "[search][export][theory][parity][cuda]") {
    SUCCEED("CUDA not built");
}

#endif
