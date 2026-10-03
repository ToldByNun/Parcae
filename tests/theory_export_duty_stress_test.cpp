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

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
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

/// Multi-chunk theory export stress for nsys duty-cycle captures.
/// Forces repeated prepare/bind/H2D/kernel/D2H with a warm TheoryExportCache
/// so PCIe/host gaps are visible next to hist kernels (NVTX ranges).
class TheoryExportDutyStress {
public:
    static constexpr std::size_t kTokens = 65536;
    static constexpr std::size_t kCandidates = 64;
    static constexpr int kChunks = 8;

    class Fixture {
    public:
        std::filesystem::path root;
        std::filesystem::path theories;
        std::string uri;
        std::vector<Index29> cipher;
        std::vector<nlohmann::json> params;
        ExpectedFrequencyTable freqs;

        Fixture(std::filesystem::path root_in, std::filesystem::path theories_in, std::string uri_in,
                std::vector<Index29> cipher_in, std::vector<nlohmann::json> params_in,
                ExpectedFrequencyTable freqs_in)
            : root(std::move(root_in)), theories(std::move(theories_in)), uri(std::move(uri_in)),
              cipher(std::move(cipher_in)), params(std::move(params_in)),
              freqs(std::move(freqs_in)) {}
    };

    [[nodiscard]] static DslCompile::Options compile_options() {
        DslCompile::Options opt;
        (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
        (void)opt.set_python_path(PARCAE_PYTHON_DIR);
        return opt;
    }

    [[nodiscard]] static Fixture make_fixture() {
        REQUIRE(DslCompile::pipeline_ready(compile_options()));

        const std::filesystem::path root =
            std::filesystem::temp_directory_path() / "parcae_theory_export_duty";
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

        std::vector<Index29> cipher(kTokens);
        for (std::size_t i = 0; i < kTokens; ++i) {
            cipher[i] = Index29{static_cast<std::uint8_t>((i * 3u + 7u) % 29u)};
        }

        std::vector<nlohmann::json> params;
        params.reserve(kCandidates);
        for (std::size_t c = 0; c < kCandidates; ++c) {
            params.push_back(nlohmann::json{{"c2", static_cast<int>(c % 29)},
                                            {"c1", static_cast<int>((c * 2) % 29)},
                                            {"c0", static_cast<int>((c * 5) % 29)}});
        }

        return Fixture{root,
                       root / "theories",
                       "parcae://theories/quadratic_polynomial_stream@1",
                       std::move(cipher),
                       std::move(params),
                       std::move(freqs.value())};
    }

    static void cleanup(const Fixture& fx) {
        std::error_code ec;
        std::filesystem::remove_all(fx.root, ec);
    }

private:
    TheoryExportDutyStress() = delete;
};

TEST_CASE("Theory export multi-chunk duty stress (nsys)",
          "[search][export][theory][duty][cuda]") {
    REQUIRE(ParcaeCuda::available());
    TheoryExportDutyStress::Fixture fx = TheoryExportDutyStress::make_fixture();
    TheoryExportCache cache;
    TheoryDeviceScratch scratch;
    CudaStreamPair streams = CudaStreamPair::create_or_legacy();
    TheoryExportPipeline pipe(cache, scratch, streams);

    for (int chunk = 0; chunk < TheoryExportDutyStress::kChunks; ++chunk) {
        // Rotate a few params so bind_slots work is non-trivial each chunk.
        std::vector<nlohmann::json> chunk_params = fx.params;
        for (std::size_t i = 0; i < chunk_params.size(); ++i) {
            chunk_params[i]["c0"] =
                static_cast<int>((chunk_params[i]["c0"].get<int>() + chunk) % 29);
        }

        StatusOr<std::optional<std::vector<double>>> prior =
            pipe.submit(fx.cipher, fx.freqs, fx.theories, fx.uri, chunk_params);
        REQUIRE(prior.ok());
        if (prior.value().has_value()) {
            REQUIRE(prior.value()->size() == TheoryExportDutyStress::kCandidates);
        }
    }
    StatusOr<std::vector<double>> last = pipe.flush();
    REQUIRE(last.ok());
    REQUIRE(last.value().size() == TheoryExportDutyStress::kCandidates);

    REQUIRE(cache.host_compile_count() == 1);
    REQUIRE(cache.device_upload_count() == 1);
    REQUIRE(cache.host_hit_count() >= static_cast<std::uint64_t>(TheoryExportDutyStress::kChunks - 1));
    REQUIRE(scratch.cipher_upload_count() == 1);
    REQUIRE(scratch.probs_upload_count() == 1);

    TheoryExportDutyStress::cleanup(fx);
}

#endif
