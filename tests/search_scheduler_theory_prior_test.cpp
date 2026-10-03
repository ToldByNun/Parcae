#include <catch2/catch_test_macros.hpp>

#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_prior.hpp>
#include <parcae/search/search_scheduler.hpp>
#include <parcae/search/theory_export_cache.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

#if defined(PARCAE_HAS_CUDA)
#include "cuda_stream_pair.hpp"
#include "parcae_cuda.hpp"
#include "theory_device_scratch.hpp"
#endif

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

[[nodiscard]] std::filesystem::path repo_data() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

[[nodiscard]] std::filesystem::path make_sandbox(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    std::filesystem::create_directories(root / "profiles" / "scores", ec);
    std::filesystem::create_directories(root / "profiles" / "gematria", ec);
    std::filesystem::create_directories(root / "profiles" / "separators", ec);
    std::filesystem::create_directories(root / "fixtures" / "solved" / "a-warning", ec);
    std::filesystem::create_directories(root / "theories", ec);

    std::filesystem::copy_file(repo_data() / "profiles" / "scores" / "english-gp-expected-v0.json",
                               root / "profiles" / "scores" / "english-gp-expected-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(repo_data() / "profiles" / "scores" / "english-gp-bigram-v0.json",
                               root / "profiles" / "scores" / "english-gp-bigram-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(repo_data() / "profiles" / "gematria" / "gematria-primus-v0.json",
                               root / "profiles" / "gematria" / "gematria-primus-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(repo_data() / "profiles" / "separators" /
                                   "rtkd-separator-grammar-v0.json",
                               root / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);

    const auto src_fix = repo_data() / "fixtures" / "solved" / "a-warning";
    const auto dst_fix = root / "fixtures" / "solved" / "a-warning";
    std::filesystem::copy_file(src_fix / "ciphertext.txt", dst_fix / "ciphertext.txt",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(src_fix / "manifest.json", dst_fix / "manifest.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    return root;
}

[[nodiscard]] StatusOr<WorkspaceManifest> make_fixture_workspace(std::string_view workspace_id,
                                                                 std::string_view utc) {
    nlohmann::json root{
        {"schema", "parcae.workspace.v0"},
        {"id", std::string(workspace_id)},
        {"created_utc", std::string(utc)},
        {"updated_utc", std::string(utc)},
        {"title", "g23"},
        {"notes", ""},
        {"input", {{"kind", "fixture_ciphertext"}, {"fixture_id", "a-warning"}, {"path", nullptr}}},
        {"default_score_id", "chi2_english_gp_v0"},
        {"default_score_version", "v0"},
    };
    return WorkspaceManifest::from_json(root);
}

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

} // namespace

#if defined(PARCAE_HAS_CUDA)

TEST_CASE("SearchScheduler catalog CUDA with prior skips GPU (no silent discard)",
          "[search][scheduler][prior][cuda]") {
    REQUIRE(ParcaeCuda::available());

    const auto root = make_sandbox("parcae_scheduler_catalog_prior_skip");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("catalog-prior-ws", "2026-10-03T07:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    const nlohmann::json excluded = {{"shift", 7}};
    StatusOr<SearchPrior> prior = SearchPrior::make(
        "catalog-prior-ws", {},
        {SearchPrior::Exclusion{SearchPrior::param_hash_of(excluded), "h-excl-shift7", "test"}},
        "2026-10-03T07:00:01Z");
    REQUIRE(prior.ok());

    StatusOr<SearchJob> job =
        SearchJob::make("catalog-prior-ws", "caesar", "chi2_english_gp_v0", /*k=*/3, /*seed=*/1,
                        Backend::Cuda, /*max_candidates=*/64, TransformDirection::Decrypt,
                        nlohmann::json::object(), prior.value().to_json());
    REQUIRE(job.ok());

    SearchScheduler::Options opts;
    opts.created_utc = "2026-10-03T07:00:02Z";
    opts.batch_id = "b-catalog-prior-0001";
    opts.omit_timing = true;

    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_once(root, ctx, job.value(), opts);
    if (!cycle.ok()) {
        FAIL(cycle.status().message());
    }
    REQUIRE(cycle.value().export_backend() == Backend::Cpu);

    StatusOr<BatchArtifact> loaded =
        BatchArtifact::load(root, "catalog-prior-ws", "b-catalog-prior-0001");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().backend() == Backend::Cpu);
    REQUIRE(loaded.value().candidate_count() == 3);
    for (const auto& line : loaded.value().candidates()) {
        REQUIRE(line.at("envelope").at("params") != excluded);
    }

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("SearchScheduler theory CUDA filters exclusions then scores_only (keeps GPU)",
          "[search][scheduler][theory][prior][cuda]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const auto root = make_sandbox("parcae_scheduler_theory_prior_cuda");
    StatusOr<DslCompile::Result> compiled = DslCompile::compile_file(
        std::filesystem::path(PARCAE_EXAMPLES_DIR) / "new_math_example.py", root / "theories",
        compile_options());
    REQUIRE(compiled.ok());

    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("theory-prior-ws", "2026-10-03T07:10:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    const std::string uri = "parcae://theories/quadratic_polynomial_stream@1";
    const nlohmann::json excluded_params = {{"c2", 1}, {"c1", 0}, {"c0", 0}};
    const nlohmann::json kept_a = {{"c2", 0}, {"c1", 1}, {"c0", 0}};
    const nlohmann::json kept_b = {{"c2", 2}, {"c1", 3}, {"c0", 5}};
    const nlohmann::json kept_c = {{"c2", 0}, {"c1", 0}, {"c0", 14}};

    StatusOr<SearchPrior> prior = SearchPrior::make(
        "theory-prior-ws", {},
        {SearchPrior::Exclusion{SearchPrior::param_hash_of(excluded_params), "h-excl-c2", "test"}},
        "2026-10-03T07:10:01Z");
    REQUIRE(prior.ok());

    const nlohmann::json param_grid{
        {"theory_uri", uri},
        {"params_list", nlohmann::json::array({excluded_params, kept_a, kept_b, kept_c})}};

    StatusOr<SearchJob> job =
        SearchJob::make("theory-prior-ws", "theory", "chi2_english_gp_v0", /*k=*/2, /*seed=*/1,
                        Backend::Cuda, /*max_candidates=*/64, TransformDirection::Decrypt,
                        param_grid, prior.value().to_json(), "v0", /*allow_extended=*/false,
                        /*allow_theory_uri=*/true);
    REQUIRE(job.ok());

    TheoryExportCache cache;
    TheoryDeviceScratch scratch;
    CudaStreamPair streams = CudaStreamPair::create_or_legacy();
    TheoryExportPipeline pipe(cache, scratch, streams);

    SearchScheduler::Options opts;
    opts.created_utc = "2026-10-03T07:10:02Z";
    opts.batch_id = "b-theory-prior-0001";
    opts.omit_timing = true;
    opts.theory_cache = &cache;
    opts.theory_scratch = &scratch;
    opts.theory_streams = &streams;
    opts.theory_pipeline = &pipe;

    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_once(root, ctx, job.value(), opts);
    if (!cycle.ok()) {
        FAIL(cycle.status().message());
    }
    REQUIRE(cycle.value().export_backend() == Backend::Cuda);

    StatusOr<BatchArtifact> loaded =
        BatchArtifact::load(root, "theory-prior-ws", "b-theory-prior-0001");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().backend() == Backend::Cuda);
    REQUIRE(loaded.value().candidate_count() == 2);
    for (const auto& line : loaded.value().candidates()) {
        REQUIRE(line.at("envelope").at("params") != excluded_params);
    }

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

#else

TEST_CASE("SearchScheduler theory prior CUDA path skipped without CUDA build",
          "[search][scheduler][theory][prior]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
