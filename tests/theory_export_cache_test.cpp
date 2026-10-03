#include <catch2/catch_test_macros.hpp>

#include <parcae/batch/batch_runner.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/theory_export_cache.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
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
    std::vector<nlohmann::json> params_chunk_b;
    ExpectedFrequencyTable freqs;

    TheoryFixture(std::filesystem::path root_in, std::filesystem::path theories_in,
                  std::string uri_in, std::vector<Index29> cipher_in,
                  std::vector<nlohmann::json> params_in, std::vector<nlohmann::json> params_b_in,
                  ExpectedFrequencyTable freqs_in)
        : root(std::move(root_in)), theories(std::move(theories_in)), uri(std::move(uri_in)),
          cipher(std::move(cipher_in)), params_list(std::move(params_in)),
          params_chunk_b(std::move(params_b_in)), freqs(std::move(freqs_in)) {}
};

[[nodiscard]] TheoryFixture make_fixture() {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_theory_export_cache";
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
        },
        {
            nlohmann::json{{"c2", 0}, {"c1", 0}, {"c0", 14}},
            nlohmann::json{{"c2", 4}, {"c1", 7}, {"c0", 1}},
        },
        std::move(freqs.value()),
    };
}

} // namespace

TEST_CASE("TheoryExportCache host bytecode hits after first ensure",
          "[search][export][theory][cache]") {
    TheoryFixture fx = make_fixture();
    TheoryExportCache cache;

    StatusOr<const TheoryExportCache::Entry*> a =
        cache.ensure(fx.theories, fx.uri, TransformDirection::Decrypt);
    REQUIRE(a.ok());
    REQUIRE(cache.host_compile_count() == 1);
    REQUIRE(cache.host_hit_count() == 0);
    REQUIRE_FALSE(a.value()->ops_u8().empty());

    StatusOr<const TheoryExportCache::Entry*> b =
        cache.ensure(fx.theories, fx.uri, TransformDirection::Decrypt);
    REQUIRE(b.ok());
    REQUIRE(b.value() == a.value());
    REQUIRE(cache.host_compile_count() == 1);
    REQUIRE(cache.host_hit_count() == 1);

    StatusOr<const TheoryExportCache::Entry*> enc =
        cache.ensure(fx.theories, fx.uri, TransformDirection::Encrypt);
    REQUIRE(enc.ok());
    REQUIRE(cache.host_compile_count() == 2);

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

TEST_CASE("GpuCandidateExport theory_from_host_scores cache preserves top-k",
          "[search][export][theory][cache]") {
    TheoryFixture fx = make_fixture();
    std::vector<double> scores(fx.params_list.size(), 0.0);
    for (std::size_t i = 0; i < fx.params_list.size(); ++i) {
        scores[i] = static_cast<double>(i) + 1.0;
    }

    TheoryExportCache cache;
    StatusOr<GpuCandidateExport::Result> first = GpuCandidateExport::theory_from_host_scores(
        fx.cipher, fx.theories, fx.uri, fx.params_list, scores, 2, TransformDirection::Decrypt,
        Backend::Cpu, BatchRunner::Progress{}, InterruptPolicy::none(), &cache);
    REQUIRE(first.ok());
    REQUIRE(cache.host_compile_count() == 1);

    StatusOr<GpuCandidateExport::Result> second = GpuCandidateExport::theory_from_host_scores(
        fx.cipher, fx.theories, fx.uri, fx.params_list, scores, 2, TransformDirection::Decrypt,
        Backend::Cpu, BatchRunner::Progress{}, InterruptPolicy::none(), &cache);
    REQUIRE(second.ok());
    REQUIRE(cache.host_compile_count() == 1);
    REQUIRE(cache.host_hit_count() >= 1);
    REQUIRE(second.value().size() == first.value().size());
    for (std::size_t i = 0; i < first.value().size(); ++i) {
        REQUIRE(second.value().rows()[i].source_index() == first.value().rows()[i].source_index());
        REQUIRE(second.value().rows()[i].score() == first.value().rows()[i].score());
        REQUIRE(second.value().rows()[i].candidate().output_indices() ==
                first.value().rows()[i].candidate().output_indices());
    }

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

#if defined(PARCAE_HAS_CUDA)

#include "cuda_stream_pair.hpp"
#include "parcae_cuda.hpp"
#include "theory_device_scratch.hpp"

TEST_CASE("GpuCandidateExport theory cache reuses device program across chunks",
          "[search][export][theory][cache][cuda]") {
    REQUIRE(ParcaeCuda::available());
    TheoryFixture fx = make_fixture();
    TheoryExportCache cache;

    StatusOr<std::vector<double>> scores_a = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, TransformDirection::Decrypt,
        BatchRunner::Progress{}, InterruptPolicy::none(), &cache);
    REQUIRE(scores_a.ok());
    REQUIRE(cache.host_compile_count() == 1);
    REQUIRE(cache.device_upload_count() == 1);

    StatusOr<std::vector<double>> scores_b = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_chunk_b, TransformDirection::Decrypt,
        BatchRunner::Progress{}, InterruptPolicy::none(), &cache);
    REQUIRE(scores_b.ok());
    REQUIRE(cache.host_compile_count() == 1);
    REQUIRE(cache.host_hit_count() >= 1);
    REQUIRE(cache.device_upload_count() == 1);
    REQUIRE(cache.device_hit_count() >= 1);
    REQUIRE(scores_b.value().size() == fx.params_chunk_b.size());

    // Cold path (no shared cache) must match warm scores for chunk A.
    StatusOr<std::vector<double>> cold = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list);
    REQUIRE(cold.ok());
    REQUIRE(cold.value().size() == scores_a.value().size());
    for (std::size_t i = 0; i < cold.value().size(); ++i) {
        REQUIRE(std::abs(cold.value()[i] - scores_a.value()[i]) < 1e-12);
    }

    StatusOr<GpuCandidateExport::Result> exported = GpuCandidateExport::theory_explicit_params(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, 2, TransformDirection::Decrypt,
        BatchRunner::Progress{}, InterruptPolicy::none(), &cache);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().backend() == Backend::Cuda);
    // prepare inside scores + materialize should hit, not recompile.
    REQUIRE(cache.host_compile_count() == 1);

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

TEST_CASE("GpuCandidateExport theory scratch skips cipher/probs across chunks",
          "[search][export][theory][scratch][cuda]") {
    REQUIRE(ParcaeCuda::available());
    TheoryFixture fx = make_fixture();
    TheoryExportCache cache;
    TheoryDeviceScratch scratch;

    StatusOr<std::vector<double>> scores_a = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, TransformDirection::Decrypt,
        BatchRunner::Progress{}, InterruptPolicy::none(), &cache, &scratch);
    REQUIRE(scores_a.ok());
    REQUIRE(scratch.cipher_upload_count() == 1);
    REQUIRE(scratch.probs_upload_count() == 1);
    REQUIRE(scratch.capacity_C() >= fx.params_list.size());
    REQUIRE(scratch.capacity_T() >= fx.cipher.size());

    StatusOr<std::vector<double>> scores_b = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_chunk_b, TransformDirection::Decrypt,
        BatchRunner::Progress{}, InterruptPolicy::none(), &cache, &scratch);
    REQUIRE(scores_b.ok());
    REQUIRE(scratch.cipher_upload_count() == 1);
    REQUIRE(scratch.probs_upload_count() == 1);
    REQUIRE(scores_b.value().size() == fx.params_chunk_b.size());

    StatusOr<std::vector<double>> cold = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list);
    REQUIRE(cold.ok());
    REQUIRE(cold.value().size() == scores_a.value().size());
    for (std::size_t i = 0; i < cold.value().size(); ++i) {
        REQUIRE(std::abs(cold.value()[i] - scores_a.value()[i]) < 1e-12);
    }

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

TEST_CASE("GpuCandidateExport theory stream pair matches cold scores",
          "[search][export][theory][streams][cuda]") {
    REQUIRE(ParcaeCuda::available());
    TheoryFixture fx = make_fixture();
    TheoryExportCache cache;
    TheoryDeviceScratch scratch;
    CudaStreamPair streams = CudaStreamPair::create_or_legacy();
    REQUIRE(streams.uses_dedicated_streams());

    StatusOr<std::vector<double>> streamed = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list, TransformDirection::Decrypt,
        BatchRunner::Progress{}, InterruptPolicy::none(), &cache, &scratch, &streams);
    REQUIRE(streamed.ok());

    StatusOr<std::vector<double>> cold = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_list);
    REQUIRE(cold.ok());
    REQUIRE(cold.value().size() == streamed.value().size());
    for (std::size_t i = 0; i < cold.value().size(); ++i) {
        REQUIRE(std::abs(cold.value()[i] - streamed.value()[i]) < 1e-12);
    }

    // Second chunk on the same streams stays correct.
    StatusOr<std::vector<double>> chunk_b = GpuCandidateExport::theory_scores_only(
        fx.cipher, fx.freqs, fx.theories, fx.uri, fx.params_chunk_b, TransformDirection::Decrypt,
        BatchRunner::Progress{}, InterruptPolicy::none(), &cache, &scratch, &streams);
    REQUIRE(chunk_b.ok());
    REQUIRE(chunk_b.value().size() == fx.params_chunk_b.size());
    REQUIRE(scratch.cipher_upload_count() == 1);

    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

#else

TEST_CASE("GpuCandidateExport theory_scores_only requires CUDA build",
          "[search][export][theory][cache][cuda]") {
    TheoryFixture fx = make_fixture();
    REQUIRE_FALSE(
        GpuCandidateExport::theory_scores_only(fx.cipher, fx.freqs, fx.theories, fx.uri,
                                               fx.params_list)
            .ok());
    std::error_code ec;
    std::filesystem::remove_all(fx.root, ec);
}

#endif
