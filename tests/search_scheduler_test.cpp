#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/hypothesis_bridge.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_scheduler.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
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
    std::filesystem::create_directories(
        root / "fixtures" / "solved" / "a-warning", ec);

    std::filesystem::copy_file(
        repo_data() / "profiles" / "scores" / "english-gp-expected-v0.json",
        root / "profiles" / "scores" / "english-gp-expected-v0.json",
        std::filesystem::copy_options::overwrite_existing,
        ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(
        repo_data() / "profiles" / "gematria" / "gematria-primus-v0.json",
        root / "profiles" / "gematria" / "gematria-primus-v0.json",
        std::filesystem::copy_options::overwrite_existing,
        ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(
        repo_data() / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
        root / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
        std::filesystem::copy_options::overwrite_existing,
        ec);
    REQUIRE(!ec);

    const auto src_fix = repo_data() / "fixtures" / "solved" / "a-warning";
    const auto dst_fix = root / "fixtures" / "solved" / "a-warning";
    std::filesystem::copy_file(
        src_fix / "ciphertext.txt",
        dst_fix / "ciphertext.txt",
        std::filesystem::copy_options::overwrite_existing,
        ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(
        src_fix / "manifest.json",
        dst_fix / "manifest.json",
        std::filesystem::copy_options::overwrite_existing,
        ec);
    REQUIRE(!ec);
    return root;
}

[[nodiscard]] StatusOr<WorkspaceManifest> make_fixture_workspace(
    std::string_view workspace_id,
    std::string_view utc) {
    nlohmann::json root{
        {"schema", "parcae.workspace.v0"},
        {"id", std::string(workspace_id)},
        {"created_utc", std::string(utc)},
        {"updated_utc", std::string(utc)},
        {"title", "g23"},
        {"notes", ""},
        {"input",
         {{"kind", "fixture_ciphertext"},
          {"fixture_id", "a-warning"},
          {"path", nullptr}}},
        {"default_score_id", "chi2_english_gp_v0"},
        {"default_score_version", "v0"},
    };
    return WorkspaceManifest::from_json(root);
}

}  // namespace

TEST_CASE(
    "SearchScheduler::run_once CPU cycle writes batch and proposes hypotheses",
    "[search][scheduler]") {
    const auto root = make_sandbox("parcae_search_scheduler_g23");
    const parcae::tool::Context ctx{root};

    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g23-ws", "2026-09-22T15:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make(
        "g23-ws",
        "atbash",
        "chi2_english_gp_v0",
        /*k=*/1,
        /*seed=*/1,
        parcae::tool::Backend::Cpu,
        /*max_candidates=*/8);
    REQUIRE(job.ok());

    SearchScheduler::Options opts;
    opts.created_utc = "2026-09-22T15:00:01Z";
    opts.batch_id = "b-g23-atbash-0001";
    opts.omit_timing = true;

    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_once(root, ctx, job.value(), opts);
    if (!cycle.ok()) {
        FAIL(cycle.status().message());
    }
    REQUIRE(cycle.value().workspace_id() == "g23-ws");
    REQUIRE(cycle.value().iterations() == 1);
    REQUIRE(cycle.value().stop_reason() == "completed_iterations");
    REQUIRE(cycle.value().omit_timing());
    REQUIRE(cycle.value().batches().size() == 1);
    REQUIRE(cycle.value().batches()[0].batch_id() == "b-g23-atbash-0001");
    REQUIRE(cycle.value().batches()[0].candidate_count() == 1);
    REQUIRE(cycle.value().hypotheses_written() == 1);

    StatusOr<BatchArtifact> loaded =
        BatchArtifact::load(root, "g23-ws", "b-g23-atbash-0001");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().family() == "atbash");
    REQUIRE(loaded.value().candidate_count() == 1);

    const std::string hid = HypothesisBridge::hypothesis_id_for(
        "g23-ws",
        "b-g23-atbash-0001",
        loaded.value().candidates()[0].at("candidate_id").get<std::string>());
    StatusOr<HypothesisRecord> hyp = HypothesisRecord::load(root, "g23-ws", hid);
    REQUIRE(hyp.ok());
    REQUIRE(hyp.value().status() == HypothesisStatus::Proposed);
    REQUIRE(hyp.value().source().at("batch_id").get<std::string>() == "b-g23-atbash-0001");

    const nlohmann::json payload = cycle.value().to_json();
    REQUIRE(payload.at("schema").get<std::string>() == "parcae.search_cycle_result.v0");
    REQUIRE(payload.at("hypotheses_written").get<std::size_t>() == 1);

    // Idempotent re-run with same batch_id updates rather than duplicating.
    StatusOr<SearchScheduler::CycleResult> again =
        SearchScheduler::run_once(root, ctx, job.value(), opts);
    REQUIRE(again.ok());
    REQUIRE(again.value().hypotheses_written() == 1);
    StatusOr<std::vector<std::string>> ids = HypothesisRecord::list_ids(root, "g23-ws");
    REQUIRE(ids.ok());
    REQUIRE(ids.value().size() == 1);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "SearchScheduler::run_once auto batch_id is deterministic",
    "[search][scheduler]") {
    const std::string a = SearchScheduler::make_batch_id(
        "ws", "caesar", "aaa", "bbb", "2026-09-22T15:00:00Z");
    const std::string b = SearchScheduler::make_batch_id(
        "ws", "caesar", "aaa", "bbb", "2026-09-22T15:00:00Z");
    REQUIRE(a == b);
    REQUIRE(a.size() == 33);
    REQUIRE(a[0] == 'b');
    REQUIRE(WorkspacePaths::validate_id(a).ok());
    REQUIRE(
        SearchScheduler::make_batch_id("ws", "caesar", "aaa", "bbb", "2026-09-22T15:00:01Z") !=
        a);
}

TEST_CASE(
    "SearchScheduler::run_once rejects cuda when not built",
    "[search][scheduler]") {
#if defined(PARCAE_HAS_CUDA)
    SUCCEED("CUDA build — NotBuilt path covered by BackendUtil elsewhere");
#else
    const auto root = make_sandbox("parcae_search_scheduler_g23_cuda");
    const parcae::tool::Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g23-cuda-ws", "2026-09-22T15:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make(
        "g23-cuda-ws",
        "caesar",
        "chi2_english_gp_v0",
        3,
        1,
        parcae::tool::Backend::Cuda,
        64);
    REQUIRE(job.ok());

    SearchScheduler::Options opts;
    opts.created_utc = "2026-09-22T15:00:01Z";
    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_once(root, ctx, job.value(), opts);
    REQUIRE_FALSE(cycle.ok());
    REQUIRE(cycle.status().message().find("CUDA") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
#endif
}
