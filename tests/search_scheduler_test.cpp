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

TEST_CASE(
    "SearchScheduler::run_loop hits completed_iterations across two batches",
    "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_loop");
    const parcae::tool::Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g24-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make(
        "g24-ws",
        "caesar",
        "chi2_english_gp_v0",
        /*k=*/2,
        /*seed=*/1,
        parcae::tool::Backend::Cpu,
        /*max_candidates=*/64);
    REQUIRE(job.ok());

    SearchScheduler::LoopOptions loop;
    loop.created_utc = "2026-09-22T16:00:00Z";
    loop.max_iterations = 2;
    loop.omit_timing = true;
    loop.batch_ids = {"b-g24-caesar-0001", "b-g24-caesar-0002"};

    StatusOr<SearchScheduler::CycleResult> result =
        SearchScheduler::run_loop(root, ctx, job.value(), loop);
    if (!result.ok()) {
        FAIL(result.status().message());
    }
    REQUIRE(result.value().iterations() == 2);
    REQUIRE(result.value().stop_reason() == SearchScheduler::stop_completed_iterations);
    REQUIRE(result.value().batches().size() == 2);
    REQUIRE(result.value().batches()[0].batch_id() == "b-g24-caesar-0001");
    REQUIRE(result.value().batches()[1].batch_id() == "b-g24-caesar-0002");
    REQUIRE(result.value().hypotheses_written() == 4);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "SearchScheduler::run_loop stops on wall_budget before any cycle",
    "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_wall");
    const parcae::tool::Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g24-wall-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make(
        "g24-wall-ws",
        "atbash",
        "chi2_english_gp_v0",
        1,
        1,
        parcae::tool::Backend::Cpu,
        8);
    REQUIRE(job.ok());

    SearchScheduler::LoopOptions loop;
    loop.created_utc = "2026-09-22T16:00:00Z";
    loop.max_iterations = 3;
    loop.max_wall_seconds = 0.0;

    StatusOr<SearchScheduler::CycleResult> result =
        SearchScheduler::run_loop(root, ctx, job.value(), loop);
    REQUIRE(result.ok());
    REQUIRE(result.value().iterations() == 0);
    REQUIRE(result.value().stop_reason() == SearchScheduler::stop_wall_budget);
    REQUIRE(result.value().batches().empty());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "SearchScheduler::run_loop stops on success_promoted",
    "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_promoted");
    const parcae::tool::Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g24-prom-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    // Seed a promoted hypothesis so the first cycle can trip stop_on_promoted.
    const nlohmann::json method = {
        {"transform_id", "atbash"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    };
    StatusOr<HypothesisRecord> seed = HypothesisRecord::make_draft(
        "g24-prom-ws",
        "h-g24-promoted-seed",
        "2026-09-22T15:59:00Z",
        "seed",
        method);
    REQUIRE(seed.ok());
    REQUIRE(seed.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(seed.value().set_status(HypothesisStatus::Promoted).ok());
    REQUIRE(seed.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make(
        "g24-prom-ws",
        "caesar",
        "chi2_english_gp_v0",
        1,
        1,
        parcae::tool::Backend::Cpu,
        64);
    REQUIRE(job.ok());

    SearchScheduler::LoopOptions loop;
    loop.created_utc = "2026-09-22T16:00:00Z";
    loop.max_iterations = 5;
    loop.stop_on_promoted = true;
    loop.batch_ids = {
        "b-g24-prom-0001",
        "b-g24-prom-0002",
        "b-g24-prom-0003",
        "b-g24-prom-0004",
        "b-g24-prom-0005",
    };

    StatusOr<SearchScheduler::CycleResult> result =
        SearchScheduler::run_loop(root, ctx, job.value(), loop);
    if (!result.ok()) {
        FAIL(result.status().message());
    }
    REQUIRE(result.value().iterations() == 1);
    REQUIRE(result.value().stop_reason() == SearchScheduler::stop_success_promoted);
    REQUIRE(result.value().batches().size() == 1);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "SearchScheduler::run_loop stops on success_validate",
    "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_validate");
    const parcae::tool::Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g24-val-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make(
        "g24-val-ws",
        "atbash",
        "chi2_english_gp_v0",
        1,
        1,
        parcae::tool::Backend::Cpu,
        8);
    REQUIRE(job.ok());

    const bool validate_ok = true;
    SearchScheduler::LoopOptions loop;
    loop.created_utc = "2026-09-22T16:00:00Z";
    loop.max_iterations = 4;
    loop.stop_on_validate = true;
    loop.validate_ok = &validate_ok;
    loop.batch_ids = {
        "b-g24-val-0001",
        "b-g24-val-0002",
        "b-g24-val-0003",
        "b-g24-val-0004",
    };

    StatusOr<SearchScheduler::CycleResult> result =
        SearchScheduler::run_loop(root, ctx, job.value(), loop);
    REQUIRE(result.ok());
    REQUIRE(result.value().iterations() == 1);
    REQUIRE(result.value().stop_reason() == SearchScheduler::stop_success_validate);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "SearchScheduler::run_loop stops on no_new_candidates after prior exclusion",
    "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_empty");
    const parcae::tool::Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g24-empty-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    // Rejected atbash (empty params) → prior excludes the only atbash lane.
    const nlohmann::json method = {
        {"transform_id", "atbash"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    };
    StatusOr<HypothesisRecord> rejected = HypothesisRecord::make_draft(
        "g24-empty-ws",
        "h-g24-reject-atbash",
        "2026-09-22T15:59:00Z",
        "reject",
        method);
    REQUIRE(rejected.ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Rejected).ok());
    REQUIRE(rejected.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make(
        "g24-empty-ws",
        "atbash",
        "chi2_english_gp_v0",
        1,
        1,
        parcae::tool::Backend::Cpu,
        8);
    REQUIRE(job.ok());

    SearchScheduler::LoopOptions loop;
    loop.created_utc = "2026-09-22T16:00:00Z";
    loop.max_iterations = 3;

    StatusOr<SearchScheduler::CycleResult> result =
        SearchScheduler::run_loop(root, ctx, job.value(), loop);
    if (!result.ok()) {
        FAIL(result.status().message());
    }
    REQUIRE(result.value().iterations() == 1);
    REQUIRE(result.value().stop_reason() == SearchScheduler::stop_no_new_candidates);
    REQUIRE(result.value().batches().empty());
    REQUIRE(result.value().hypotheses_written() == 0);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "SearchScheduler::advance_utc_seconds is deterministic",
    "[search][scheduler][loop]") {
    StatusOr<std::string> next =
        SearchScheduler::advance_utc_seconds("2026-09-22T16:00:00Z", 1);
    REQUIRE(next.ok());
    REQUIRE(next.value() == "2026-09-22T16:00:01Z");
    StatusOr<std::string> day =
        SearchScheduler::advance_utc_seconds("2026-09-22T23:59:59Z", 1);
    REQUIRE(day.ok());
    REQUIRE(day.value() == "2026-09-23T00:00:00Z");
}
