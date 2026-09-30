#include <catch2/catch_test_macros.hpp>

#include <parcae/batch/batch_runner.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/generate/theory_explicit_params_candidate_generator.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <utility>
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

struct TheoryFixture {
    std::filesystem::path root;
    std::filesystem::path theories;
    std::string uri;
    std::vector<Index29> cipher;
    std::vector<nlohmann::json> params_list;
    ExpectedFrequencyTable freqs;

    TheoryFixture(std::filesystem::path root_in, std::filesystem::path theories_in,
                  std::string uri_in, std::vector<Index29> cipher_in,
                  std::vector<nlohmann::json> params_in, ExpectedFrequencyTable freqs_in)
        : root(std::move(root_in)), theories(std::move(theories_in)), uri(std::move(uri_in)),
          cipher(std::move(cipher_in)), params_list(std::move(params_in)),
          freqs(std::move(freqs_in)) {}
};

[[nodiscard]] TheoryFixture make_fixture() {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_gpu_export_theory";
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

    return TheoryFixture{
        root,
        root / "theories",
        "parcae://theories/quadratic_polynomial_stream@1",
        {Index29{3}, Index29{10}, Index29{28}, Index29{0}, Index29{7}, Index29{15}, Index29{22},
         Index29{4}, Index29{11}, Index29{19}},
        {
            nlohmann::json{{"c2", 1}, {"c1", 0}, {"c0", 0}},
            nlohmann::json{{"c2", 0}, {"c1", 1}, {"c0", 0}},
            nlohmann::json{{"c2", 2}, {"c1", 3}, {"c0", 5}},
            nlohmann::json{{"c2", 0}, {"c1", 0}, {"c0", 14}},
            nlohmann::json{{"c2", 4}, {"c1", 7}, {"c0", 1}},
        },
        std::move(freqs.value()),
    };
}

[[nodiscard]] std::vector<double> cpu_scores(const TheoryFixture& fx) {
    StatusOr<std::vector<TransformCandidate>> generated =
        TheoryExplicitParamsCandidateGenerator::generate(fx.cipher, fx.theories, fx.uri,
                                                         fx.params_list,
                                                         TransformDirection::Decrypt);
    REQUIRE(generated.ok());
    std::vector<double> scores;
    scores.reserve(generated.value().size());
    for (const TransformCandidate& c : generated.value()) {
        StatusOr<double> s = Chi2EnglishGp::score(c.output_indices(), fx.freqs);
        REQUIRE(s.ok());
        scores.push_back(s.value());
    }
    return scores;
}

} // namespace

TEST_CASE("GpuCandidateExport theory_from_host_scores top-k matches CPU χ² order",
          "[search][export][theory]") {
    TheoryFixture fx = make_fixture();
    const std::vector<double> scores = cpu_scores(fx);
    constexpr std::size_t k = 3;

    StatusOr<GpuCandidateExport::Result> exported = GpuCandidateExport::theory_from_host_scores(
        fx.cipher, fx.theories, fx.uri, fx.params_list, scores, k, TransformDirection::Decrypt,
        Backend::Cpu);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().size() == k);
    REQUIRE(exported.value().backend() == Backend::Cpu);

    std::vector<std::size_t> order(scores.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; });

    for (std::size_t rank = 0; rank < k; ++rank) {
        const std::size_t src = order[rank];
        REQUIRE(exported.value().rows()[rank].source_index() == src);
        REQUIRE(exported.value().rows()[rank].score() == scores[src]);
        REQUIRE(exported.value().rows()[rank].candidate().candidate_id() ==
                TheoryExplicitParamsCandidateGenerator::make_candidate_id(fx.uri, src));
        REQUIRE(exported.value().rows()[rank].candidate().transform_id().str() == fx.uri);
        REQUIRE(exported.value().rows()[rank].candidate().params() == fx.params_list[src]);

        StatusOr<std::vector<TransformCandidate>> one =
            TheoryExplicitParamsCandidateGenerator::generate(
                fx.cipher, fx.theories, fx.uri,
                std::vector<nlohmann::json>{fx.params_list[src]}, TransformDirection::Decrypt);
        REQUIRE(one.ok());
        REQUIRE(exported.value().rows()[rank].candidate().output_indices() ==
                one.value()[0].output_indices());
    }

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

TEST_CASE("GpuCandidateExport theory rejects encrypt / interrupt / empty list",
          "[search][export][theory]") {
    TheoryFixture fx = make_fixture();
    const std::vector<double> scores = cpu_scores(fx);

    REQUIRE_FALSE(GpuCandidateExport::theory_explicit_params(
                      fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, 2,
                      TransformDirection::Encrypt)
                      .ok());

    StatusOr<InterruptPolicy> irq = InterruptPolicy::from_skip_indices({0});
    REQUIRE(irq.ok());
    REQUIRE_FALSE(GpuCandidateExport::theory_from_host_scores(
                      fx.cipher, fx.theories, fx.uri, fx.params_list, scores, 2,
                      TransformDirection::Decrypt, Backend::Cpu, BatchRunner::Progress{},
                      irq.value())
                      .ok());
    REQUIRE_FALSE(GpuCandidateExport::theory_explicit_params(
                      fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, 2,
                      TransformDirection::Decrypt, BatchRunner::Progress{}, irq.value())
                      .ok());

    std::vector<nlohmann::json> empty;
    REQUIRE_FALSE(GpuCandidateExport::theory_from_host_scores(
                      fx.cipher, fx.theories, fx.uri, empty, std::span<const double>{}, 1)
                      .ok());

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"

TEST_CASE("GpuCandidateExport theory_explicit_params CUDA top-k matches host scores",
          "[search][export][theory][cuda]") {
    REQUIRE(ParcaeCuda::available());
    TheoryFixture fx = make_fixture();
    const std::vector<double> host = cpu_scores(fx);
    constexpr std::size_t k = 3;

    StatusOr<GpuCandidateExport::Result> exported = GpuCandidateExport::theory_explicit_params(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, k);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().backend() == Backend::Cuda);
    REQUIRE(exported.value().size() == k);

    StatusOr<GpuCandidateExport::Result> from_host = GpuCandidateExport::theory_from_host_scores(
        fx.cipher, fx.theories, fx.uri, fx.params_list, host, k, TransformDirection::Decrypt,
        Backend::Cpu);
    REQUIRE(from_host.ok());

    for (std::size_t rank = 0; rank < k; ++rank) {
        REQUIRE(exported.value().rows()[rank].source_index() ==
                from_host.value().rows()[rank].source_index());
        REQUIRE(exported.value().rows()[rank].score() ==
                from_host.value().rows()[rank].score());
        REQUIRE(exported.value().rows()[rank].candidate().params() ==
                from_host.value().rows()[rank].candidate().params());
        REQUIRE(exported.value().rows()[rank].candidate().output_indices() ==
                from_host.value().rows()[rank].candidate().output_indices());
    }

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

#else

TEST_CASE("GpuCandidateExport theory_explicit_params requires CUDA build",
          "[search][export][theory][cuda]") {
    TheoryFixture fx = make_fixture();
    REQUIRE_FALSE(GpuCandidateExport::theory_explicit_params(fx.cipher, fx.freqs, fx.theories,
                                                             fx.uri, fx.params_list, 2)
                      .ok());
    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

#endif
