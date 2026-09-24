#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <parcae/core/index29.hpp>
#include <parcae/generate/atbash_candidate_generator.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/search/cpu_candidate_export.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/workspace_cipher.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <string>
#include <string_view>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] Context test_context() {
    return Context{std::string(PARCAE_TEST_DATA_DIR)};
}

[[nodiscard]] std::vector<Index29> a_warning_cipher() {
    StatusOr<WorkspaceCipher> cipher =
        WorkspaceCipher::from_fixture(std::string(PARCAE_TEST_DATA_DIR), "_example", "a-warning");
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value().indices().size() == 184);
    return cipher.value().indices();
}

void require_same_topk(const CpuCandidateExport::Result& cpu,
                       const GpuCandidateExport::Result& gpu) {
    REQUIRE(cpu.size() == gpu.size());
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        REQUIRE(cpu.rows()[i].candidate().candidate_id() ==
                gpu.rows()[i].candidate().candidate_id());
        REQUIRE(cpu.rows()[i].score() == gpu.rows()[i].score());
        REQUIRE(cpu.rows()[i].candidate().output_indices() ==
                gpu.rows()[i].candidate().output_indices());
        REQUIRE(cpu.rows()[i].rank() == gpu.rows()[i].rank());
    }
}

} // namespace

TEST_CASE("CpuCandidateExport a-warning CPU path: atbash / caesar / atbash_caesar",
          "[search][export][parity][a-warning]") {
    const Context ctx = test_context();
    const std::vector<Index29> cipher = a_warning_cipher();

    {
        StatusOr<CpuCandidateExport::Result> atbash =
            CpuCandidateExport::run(cipher, "atbash", "chi2_english_gp_v0", 1, ctx);
        REQUIRE(atbash.ok());
        REQUIRE(atbash.value().size() == 1);
        REQUIRE(atbash.value().backend() == Backend::Cpu);
        REQUIRE(atbash.value().rows()[0].candidate().candidate_id() ==
                AtbashCandidateGenerator::make_candidate_id());
    }

    {
        constexpr std::size_t k = 5;
        StatusOr<CpuCandidateExport::Result> caesar =
            CpuCandidateExport::run(cipher, "caesar", "chi2_english_gp_v0", k, ctx);
        REQUIRE(caesar.ok());
        REQUIRE(caesar.value().size() == k);
        for (std::size_t i = 1; i < k; ++i) {
            REQUIRE(caesar.value().rows()[i - 1].score() <= caesar.value().rows()[i].score());
        }
    }

    {
        constexpr std::size_t k = 5;
        StatusOr<CpuCandidateExport::Result> composed =
            CpuCandidateExport::run(cipher, "atbash_caesar", "chi2_english_gp_v0", k, ctx);
        REQUIRE(composed.ok());
        REQUIRE(composed.value().size() == k);
    }
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"

TEST_CASE("CpuCandidateExport vs GpuCandidateExport top-k parity on a-warning",
          "[search][export][parity][a-warning][cuda]") {
    if (!ParcaeCuda::available()) {
        SKIP("No CUDA device");
    }

    const Context ctx = test_context();
    const std::vector<Index29> cipher = a_warning_cipher();
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    {
        StatusOr<CpuCandidateExport::Result> cpu =
            CpuCandidateExport::run(cipher, "atbash", "chi2_english_gp_v0", 1, ctx);
        REQUIRE(cpu.ok());
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::atbash(cipher, freqs.value(), 1);
        REQUIRE(gpu.ok());
        REQUIRE(gpu.value().backend() == Backend::Cuda);
        require_same_topk(cpu.value(), gpu.value());
    }

    {
        constexpr std::size_t k = 8;
        StatusOr<CpuCandidateExport::Result> cpu =
            CpuCandidateExport::run(cipher, "caesar", "chi2_english_gp_v0", k, ctx);
        REQUIRE(cpu.ok());
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::caesar(cipher, freqs.value(), k);
        REQUIRE(gpu.ok());
        require_same_topk(cpu.value(), gpu.value());
    }

    {
        constexpr std::size_t k = 8;
        StatusOr<CpuCandidateExport::Result> cpu =
            CpuCandidateExport::run(cipher, "atbash_caesar", "chi2_english_gp_v0", k, ctx);
        REQUIRE(cpu.ok());
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::atbash_caesar(cipher, freqs.value(), k);
        REQUIRE(gpu.ok());
        require_same_topk(cpu.value(), gpu.value());
    }

    {
        constexpr std::size_t k = 5;
        StatusOr<CpuCandidateExport::Result> cpu =
            CpuCandidateExport::run(cipher, "affine", "chi2_english_gp_v0", k, ctx);
        REQUIRE(cpu.ok());
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::affine(cipher, freqs.value(), k);
        REQUIRE(gpu.ok());
        require_same_topk(cpu.value(), gpu.value());
    }

    {
        constexpr std::size_t k = 4;
        constexpr std::size_t max_key_length = 8;
        const nlohmann::json grid = {{"max_key_length", static_cast<int>(max_key_length)}};
        StatusOr<CpuCandidateExport::Result> cpu = CpuCandidateExport::run(
            cipher, "vigenere", "chi2_english_gp_v0", k, ctx, TransformDirection::Decrypt, grid);
        REQUIRE(cpu.ok());
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::vigenere_bounded(cipher, freqs.value(), k, max_key_length);
        REQUIRE(gpu.ok());
        require_same_topk(cpu.value(), gpu.value());
    }
}

#else

TEST_CASE("Cpu vs Gpu export parity device path skipped (PARCAE_HAS_CUDA unset)",
          "[search][export][parity][a-warning][cuda]") {
    SUCCEED("PARCAE_HAS_CUDA unset — CPU path covered above; fused CUDA compare not linked");
}

#endif
