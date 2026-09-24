#include <parcae/core/index29.hpp>
#include <parcae/generate/compose_recipe_candidate_generator.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/search/cpu_candidate_export.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/workspace_cipher.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] Context test_context() {
    return Context{std::string(PARCAE_TEST_DATA_DIR)};
}

[[nodiscard]] std::vector<Index29> a_warning_cipher() {
    StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::from_fixture(
        std::string(PARCAE_TEST_DATA_DIR), "_example", "a-warning");
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value().indices().size() == 184);
    return cipher.value().indices();
}

void require_same_topk(
    const CpuCandidateExport::Result& cpu,
    const GpuCandidateExport::Result& gpu) {
    REQUIRE(cpu.size() == gpu.size());
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        REQUIRE(
            cpu.rows()[i].candidate().candidate_id() ==
            gpu.rows()[i].candidate().candidate_id());
        REQUIRE(cpu.rows()[i].score() == gpu.rows()[i].score());
        REQUIRE(
            cpu.rows()[i].candidate().output_indices() ==
            gpu.rows()[i].candidate().output_indices());
        REQUIRE(cpu.rows()[i].rank() == gpu.rows()[i].rank());
        REQUIRE(
            cpu.rows()[i].candidate().transform_id() == TransformId::compose());
        REQUIRE(
            gpu.rows()[i].candidate().transform_id() == TransformId::compose());
    }
}

[[nodiscard]] nlohmann::json two_atbash_caesar_recipes() {
    return nlohmann::json{
        {"recipes",
         nlohmann::json::array(
             {ComposeTransform::atbash_then_caesar_params(3),
              ComposeTransform::atbash_then_caesar_params(7),
              ComposeTransform::atbash_then_caesar_params(11)})}};
}

}  // namespace

TEST_CASE(
    "Compose job CPU smoke: default grid and explicit recipes",
    "[search][export][compose][parity]") {
    const Context ctx = test_context();
    const std::vector<Index29> cipher = a_warning_cipher();

    constexpr std::size_t k = 5;
    StatusOr<SearchJob> default_job = SearchJob::make(
        "_example",
        "compose",
        "chi2_english_gp_v0",
        k,
        1,
        Backend::Cpu,
        64);
    REQUIRE(default_job.ok());
    StatusOr<CpuCandidateExport::Result> def =
        CpuCandidateExport::from_job(cipher, default_job.value(), ctx);
    REQUIRE(def.ok());
    REQUIRE(def.value().size() == k);
    REQUIRE(def.value().backend() == Backend::Cpu);

    StatusOr<CpuCandidateExport::Result> atbash_caesar = CpuCandidateExport::run(
        cipher, "atbash_caesar", "chi2_english_gp_v0", k, ctx);
    REQUIRE(atbash_caesar.ok());
    REQUIRE(
        def.value().rows()[0].candidate().candidate_id() ==
        atbash_caesar.value().rows()[0].candidate().candidate_id());
    REQUIRE(def.value().rows()[0].score() == atbash_caesar.value().rows()[0].score());

    const nlohmann::json grid = two_atbash_caesar_recipes();
    StatusOr<SearchJob> recipe_job = SearchJob::make(
        "_example",
        "compose",
        "chi2_english_gp_v0",
        2,
        1,
        Backend::Cpu,
        64,
        TransformDirection::Decrypt,
        grid);
    REQUIRE(recipe_job.ok());
    StatusOr<CpuCandidateExport::Result> recipes =
        CpuCandidateExport::from_job(cipher, recipe_job.value(), ctx);
    REQUIRE(recipes.ok());
    REQUIRE(recipes.value().size() == 2);
    REQUIRE(
        recipes.value().rows()[0].candidate().candidate_id().find("atbash_caesar:shift=") !=
        std::string::npos);
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"

TEST_CASE(
    "Compose job CPU vs CUDA mirror smoke (AtbashCaesar grid + ComposeDriver recipes)",
    "[search][export][compose][parity][cuda]") {
    if (!ParcaeCuda::available()) {
        SKIP("No CUDA device");
    }

    const Context ctx = test_context();
    const std::vector<Index29> cipher = a_warning_cipher();
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());

    // Default compose grid ↔ fused Atbash∘Caesar export.
    {
        constexpr std::size_t k = 8;
        StatusOr<CpuCandidateExport::Result> cpu = CpuCandidateExport::run(
            cipher, "compose", "chi2_english_gp_v0", k, ctx);
        REQUIRE(cpu.ok());

        StatusOr<GpuCandidateExport::Result> gpu_compose =
            GpuCandidateExport::compose_from_param_grid(
                cipher, freqs.value(), nlohmann::json::object(), k);
        REQUIRE(gpu_compose.ok());
        REQUIRE(gpu_compose.value().backend() == Backend::Cuda);
        require_same_topk(cpu.value(), gpu_compose.value());

        StatusOr<GpuCandidateExport::Result> gpu_atbash =
            GpuCandidateExport::atbash_caesar(cipher, freqs.value(), k);
        REQUIRE(gpu_atbash.ok());
        require_same_topk(cpu.value(), gpu_atbash.value());
    }

    // Explicit short recipe list ↔ ComposeDriver apply + host χ².
    {
        constexpr std::size_t k = 2;
        const nlohmann::json grid = two_atbash_caesar_recipes();
        REQUIRE_FALSE(
            ComposeRecipeCandidateGenerator::is_full_atbash_caesar_grid(
                ComposeRecipeCandidateGenerator::recipes_from_param_grid(grid).value()));

        StatusOr<CpuCandidateExport::Result> cpu = CpuCandidateExport::run(
            cipher,
            "compose",
            "chi2_english_gp_v0",
            k,
            ctx,
            TransformDirection::Decrypt,
            grid);
        REQUIRE(cpu.ok());
        REQUIRE(cpu.value().size() == k);

        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::compose_from_param_grid(
                cipher, freqs.value(), grid, k);
        REQUIRE(gpu.ok());
        REQUIRE(gpu.value().backend() == Backend::Cuda);
        require_same_topk(cpu.value(), gpu.value());
    }

    // SearchJob backend=cuda compose default through from_job is CPU-export only;
    // scheduler uses GpuCandidateExport — mirror via compose_from_param_grid above.
    {
        StatusOr<SearchJob> job = SearchJob::make(
            "_example",
            "compose",
            "chi2_english_gp_v0",
            3,
            1,
            Backend::Cuda,
            64);
        REQUIRE(job.ok());
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::compose_from_param_grid(
                cipher, freqs.value(), job.value().param_grid(), job.value().k());
        REQUIRE(gpu.ok());
        StatusOr<CpuCandidateExport::Result> cpu =
            CpuCandidateExport::from_job(cipher, job.value(), ctx);
        // from_job always uses CPU generate+rank; still must match GPU top-k.
        REQUIRE(cpu.ok());
        require_same_topk(cpu.value(), gpu.value());
    }
}

#else

TEST_CASE(
    "Compose job CUDA mirror skipped (PARCAE_HAS_CUDA unset)",
    "[search][export][compose][parity][cuda]") {
    SUCCEED(
        "PARCAE_HAS_CUDA unset — CPU compose smoke covered above; CUDA mirror not linked");
}

#endif
