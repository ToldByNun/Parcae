#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <parcae/cli/console_progress_sink.hpp>
#include <parcae/core/sha256.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/hypothesis_bridge.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_prior.hpp>
#include <parcae/search/search_scheduler.hpp>
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

    std::filesystem::copy_file(repo_data() / "profiles" / "scores" / "english-gp-expected-v0.json",
                               root / "profiles" / "scores" / "english-gp-expected-v0.json",
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

[[nodiscard]] std::vector<std::string> candidate_ids(const BatchArtifact& batch) {
    std::vector<std::string> ids;
    ids.reserve(batch.candidates().size());
    for (const nlohmann::json& row : batch.candidates()) {
        ids.push_back(row.at("candidate_id").get<std::string>());
    }
    return ids;
}

[[nodiscard]] std::vector<double> candidate_scores(const BatchArtifact& batch) {
    std::vector<double> scores;
    scores.reserve(batch.candidates().size());
    for (const nlohmann::json& row : batch.candidates()) {
        scores.push_back(row.at("score").at("value").get<double>());
    }
    return scores;
}

[[nodiscard]] std::string fixture_ciphertext_sha256(const std::filesystem::path& data_root) {
    const auto path = data_root / "fixtures" / "solved" / "a-warning" / "ciphertext.txt";
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return Sha256::hex_digest(bytes);
}

} // namespace

TEST_CASE("SearchScheduler::run_once CPU cycle writes batch and proposes hypotheses",
          "[search][scheduler]") {
    const auto root = make_sandbox("parcae_search_scheduler_g23");
    const Context ctx{root};

    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g23-ws", "2026-09-22T15:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make("g23-ws", "atbash", "chi2_english_gp_v0",
                                              /*k=*/1,
                                              /*seed=*/1, Backend::Cpu,
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

    StatusOr<BatchArtifact> loaded = BatchArtifact::load(root, "g23-ws", "b-g23-atbash-0001");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().family() == "atbash");
    REQUIRE(loaded.value().candidate_count() == 1);

    const std::string hid = HypothesisBridge::hypothesis_id_for(
        "g23-ws", "b-g23-atbash-0001",
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

TEST_CASE("SearchScheduler::run_once auto batch_id is deterministic", "[search][scheduler]") {
    const std::string a =
        SearchScheduler::make_batch_id("ws", "caesar", "aaa", "bbb", "2026-09-22T15:00:00Z");
    const std::string b =
        SearchScheduler::make_batch_id("ws", "caesar", "aaa", "bbb", "2026-09-22T15:00:00Z");
    REQUIRE(a == b);
    REQUIRE(a.size() == 33);
    REQUIRE(a[0] == 'b');
    REQUIRE(WorkspacePaths::validate_id(a).ok());
    REQUIRE(SearchScheduler::make_batch_id("ws", "caesar", "aaa", "bbb", "2026-09-22T15:00:01Z") !=
            a);
}

TEST_CASE("SearchScheduler::run_once rejects cuda when not built", "[search][scheduler]") {
#if defined(PARCAE_HAS_CUDA)
    SUCCEED("CUDA build — NotBuilt path covered by BackendUtil elsewhere");
#else
    const auto root = make_sandbox("parcae_search_scheduler_g23_cuda");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g23-cuda-ws", "2026-09-22T15:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job =
        SearchJob::make("g23-cuda-ws", "caesar", "chi2_english_gp_v0", 3, 1, Backend::Cuda, 64);
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

TEST_CASE("SearchScheduler::run_loop hits completed_iterations across two batches",
          "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_loop");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g24-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make("g24-ws", "caesar", "chi2_english_gp_v0",
                                              /*k=*/2,
                                              /*seed=*/1, Backend::Cpu,
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

TEST_CASE("SearchScheduler::run_loop stops on wall_budget before any cycle",
          "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_wall");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g24-wall-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job =
        SearchJob::make("g24-wall-ws", "atbash", "chi2_english_gp_v0", 1, 1, Backend::Cpu, 8);
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

TEST_CASE("SearchScheduler::run_loop stops on success_promoted", "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_promoted");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g24-prom-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    // Seed a promoted hypothesis so the first cycle can trip stop_on_promoted.
    const nlohmann::json method = {
        {"transform_id", "atbash"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    };
    StatusOr<HypothesisRecord> seed = HypothesisRecord::make_draft(
        "g24-prom-ws", "h-g24-promoted-seed", "2026-09-22T15:59:00Z", "seed", method);
    REQUIRE(seed.ok());
    REQUIRE(seed.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(seed.value().set_status(HypothesisStatus::Promoted).ok());
    REQUIRE(seed.value().store(root).ok());

    StatusOr<SearchJob> job =
        SearchJob::make("g24-prom-ws", "caesar", "chi2_english_gp_v0", 1, 1, Backend::Cpu, 64);
    REQUIRE(job.ok());

    SearchScheduler::LoopOptions loop;
    loop.created_utc = "2026-09-22T16:00:00Z";
    loop.max_iterations = 5;
    loop.stop_on_promoted = true;
    loop.batch_ids = {
        "b-g24-prom-0001", "b-g24-prom-0002", "b-g24-prom-0003",
        "b-g24-prom-0004", "b-g24-prom-0005",
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

TEST_CASE("SearchScheduler::run_loop stops on success_validate", "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_validate");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g24-val-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job =
        SearchJob::make("g24-val-ws", "atbash", "chi2_english_gp_v0", 1, 1, Backend::Cpu, 8);
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

TEST_CASE("SearchScheduler::run_loop stops on no_new_candidates after prior exclusion",
          "[search][scheduler][loop]") {
    const auto root = make_sandbox("parcae_search_scheduler_g24_empty");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g24-empty-ws", "2026-09-22T16:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    // Rejected atbash (empty params) → prior excludes the only atbash lane.
    const nlohmann::json method = {
        {"transform_id", "atbash"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    };
    StatusOr<HypothesisRecord> rejected = HypothesisRecord::make_draft(
        "g24-empty-ws", "h-g24-reject-atbash", "2026-09-22T15:59:00Z", "reject", method);
    REQUIRE(rejected.ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Rejected).ok());
    REQUIRE(rejected.value().store(root).ok());

    StatusOr<SearchJob> job =
        SearchJob::make("g24-empty-ws", "atbash", "chi2_english_gp_v0", 1, 1, Backend::Cpu, 8);
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

TEST_CASE("SearchScheduler::advance_utc_seconds is deterministic", "[search][scheduler][loop]") {
    StatusOr<std::string> next = SearchScheduler::advance_utc_seconds("2026-09-22T16:00:00Z", 1);
    REQUIRE(next.ok());
    REQUIRE(next.value() == "2026-09-22T16:00:01Z");
    StatusOr<std::string> day = SearchScheduler::advance_utc_seconds("2026-09-22T23:59:59Z", 1);
    REQUIRE(day.ok());
    REQUIRE(day.value() == "2026-09-23T00:00:00Z");
}

TEST_CASE("SearchScheduler next cycle excludes rejected params from prior",
          "[search][scheduler][prior]") {
    const auto root = make_sandbox("parcae_search_scheduler_g25_excl");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g25-excl-ws", "2026-09-22T17:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make("g25-excl-ws", "caesar", "chi2_english_gp_v0",
                                              /*k=*/5,
                                              /*seed=*/1, Backend::Cpu,
                                              /*max_candidates=*/64);
    REQUIRE(job.ok());

    SearchScheduler::Options first;
    first.created_utc = "2026-09-22T17:00:01Z";
    first.batch_id = "b-g25-excl-0001";
    StatusOr<SearchScheduler::CycleResult> cycle1 =
        SearchScheduler::run_once(root, ctx, job.value(), first);
    if (!cycle1.ok()) {
        FAIL(cycle1.status().message());
    }
    REQUIRE(cycle1.value().batches().size() == 1);

    StatusOr<BatchArtifact> batch1 = BatchArtifact::load(root, "g25-excl-ws", "b-g25-excl-0001");
    REQUIRE(batch1.ok());
    REQUIRE_FALSE(batch1.value().candidates().empty());
    const nlohmann::json& top = batch1.value().candidates()[0];
    const std::string candidate_id = top.at("candidate_id").get<std::string>();
    const int rejected_shift = top.at("envelope").at("params").at("shift").get<int>();

    const std::string hid =
        HypothesisBridge::hypothesis_id_for("g25-excl-ws", "b-g25-excl-0001", candidate_id);
    StatusOr<HypothesisRecord> hyp = HypothesisRecord::load(root, "g25-excl-ws", hid);
    REQUIRE(hyp.ok());
    REQUIRE(hyp.value().set_status(HypothesisStatus::Rejected).ok());
    REQUIRE(hyp.value().store(root).ok());

    StatusOr<SearchPrior> prior =
        SearchPrior::from_workspace(root, "g25-excl-ws", "2026-09-22T17:00:02Z");
    REQUIRE(prior.ok());
    REQUIRE(prior.value().exclusions().size() == 1);
    REQUIRE(prior.value().excludes_params(nlohmann::json{{"shift", rejected_shift}}));

    SearchScheduler::Options second;
    second.created_utc = "2026-09-22T17:00:02Z";
    second.batch_id = "b-g25-excl-0002";
    StatusOr<SearchScheduler::CycleResult> cycle2 =
        SearchScheduler::run_once(root, ctx, job.value(), second);
    if (!cycle2.ok()) {
        FAIL(cycle2.status().message());
    }

    StatusOr<BatchArtifact> batch2 = BatchArtifact::load(root, "g25-excl-ws", "b-g25-excl-0002");
    REQUIRE(batch2.ok());
    REQUIRE(batch2.value().prior_digest_sha256() != batch1.value().prior_digest_sha256());
    for (const nlohmann::json& row : batch2.value().candidates()) {
        REQUIRE(row.at("envelope").at("params").at("shift").get<int>() != rejected_shift);
        REQUIRE(row.at("candidate_id").get<std::string>() != candidate_id);
    }

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("SearchScheduler next cycle forces promoted seed into export",
          "[search][scheduler][prior]") {
    const auto root = make_sandbox("parcae_search_scheduler_g25_seed");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g25-seed-ws", "2026-09-22T17:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    const nlohmann::json shift7 = {
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", 7}}},
    };

    // Exclude grid lane shift=7, then promote the same params as a seed so the
    // next job must materialize `prior-seed:…` (search-loop.md seed rule).
    StatusOr<HypothesisRecord> rejected = HypothesisRecord::make_draft(
        "g25-seed-ws", "h-g25-reject-shift7", "2026-09-22T16:59:00Z", "reject shift 7", shift7);
    REQUIRE(rejected.ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Rejected).ok());
    REQUIRE(rejected.value().store(root).ok());

    StatusOr<HypothesisRecord> promoted = HypothesisRecord::make_draft(
        "g25-seed-ws", "h-g25-promote-shift7", "2026-09-22T16:59:01Z", "promote shift 7", shift7);
    REQUIRE(promoted.ok());
    REQUIRE(promoted.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(promoted.value().set_status(HypothesisStatus::Promoted).ok());
    REQUIRE(promoted.value().store(root).ok());

    StatusOr<SearchJob> job = SearchJob::make("g25-seed-ws", "caesar", "chi2_english_gp_v0",
                                              /*k=*/29,
                                              /*seed=*/1, Backend::Cpu,
                                              /*max_candidates=*/64);
    REQUIRE(job.ok());

    SearchScheduler::Options opts;
    opts.created_utc = "2026-09-22T17:00:01Z";
    opts.batch_id = "b-g25-seed-0001";
    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_once(root, ctx, job.value(), opts);
    if (!cycle.ok()) {
        FAIL(cycle.status().message());
    }

    StatusOr<BatchArtifact> batch = BatchArtifact::load(root, "g25-seed-ws", "b-g25-seed-0001");
    REQUIRE(batch.ok());

    bool saw_seed = false;
    for (const nlohmann::json& row : batch.value().candidates()) {
        const std::string cid = row.at("candidate_id").get<std::string>();
        if (cid.rfind("prior-seed:", 0) == 0) {
            saw_seed = true;
            REQUIRE(cid == "prior-seed:h-g25-promote-shift7");
            REQUIRE(row.at("envelope").at("params").at("shift").get<int>() == 7);
        }
        REQUIRE(cid != "caesar:shift=7");
    }
    REQUIRE(saw_seed);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("SearchScheduler prefers inline SearchJob.prior over workspace rebuild",
          "[search][scheduler][prior]") {
    const auto root = make_sandbox("parcae_search_scheduler_g25_inline");
    const Context ctx{root};
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("g25-inline-ws", "2026-09-22T17:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    // Workspace has a rejected shift=3, but inline prior is empty → shift 3 may appear.
    const nlohmann::json shift3 = {
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", 3}}},
    };
    StatusOr<HypothesisRecord> rejected = HypothesisRecord::make_draft(
        "g25-inline-ws", "h-g25-inline-reject", "2026-09-22T16:59:00Z", "reject", shift3);
    REQUIRE(rejected.ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Rejected).ok());
    REQUIRE(rejected.value().store(root).ok());

    StatusOr<SearchPrior> empty_prior =
        SearchPrior::make("g25-inline-ws", {}, {}, "2026-09-22T17:00:01Z");
    REQUIRE(empty_prior.ok());

    StatusOr<SearchJob> job =
        SearchJob::make("g25-inline-ws", "caesar", "chi2_english_gp_v0",
                        /*k=*/29,
                        /*seed=*/1, Backend::Cpu,
                        /*max_candidates=*/64, TransformDirection::Decrypt,
                        nlohmann::json::object(), empty_prior.value().to_json());
    REQUIRE(job.ok());

    SearchScheduler::Options opts;
    opts.created_utc = "2026-09-22T17:00:01Z";
    opts.batch_id = "b-g25-inline-0001";
    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_once(root, ctx, job.value(), opts);
    if (!cycle.ok()) {
        FAIL(cycle.status().message());
    }

    StatusOr<BatchArtifact> batch = BatchArtifact::load(root, "g25-inline-ws", "b-g25-inline-0001");
    REQUIRE(batch.ok());
    REQUIRE(batch.value().candidate_count() == 29);
    bool saw_shift3 = false;
    for (const nlohmann::json& row : batch.value().candidates()) {
        if (row.at("envelope").at("params").at("shift").get<int>() == 3) {
            saw_shift3 = true;
        }
    }
    REQUIRE(saw_shift3);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("SearchScheduler two-iteration CPU loop is deterministic (stable prior)",
          "[search][scheduler][loop][determinism]") {
    // Conformance: two iterations with fixed seed are byte-stable for digests and
    // candidate_id order when priors do not unexpectedly change mid-run
    // (proposed-only hypotheses leave SearchPrior empty).
    auto run_twice = [](std::string_view sandbox_name) {
        const auto root = make_sandbox(sandbox_name);
        const std::string fixture_sha_before = fixture_ciphertext_sha256(root);
        const Context ctx{root};

        StatusOr<WorkspaceManifest> ws = make_fixture_workspace("g26-ws", "2026-09-22T18:00:00Z");
        REQUIRE(ws.ok());
        REQUIRE(ws.value().store(root).ok());

        StatusOr<SearchJob> job = SearchJob::make("g26-ws", "caesar", "chi2_english_gp_v0",
                                                  /*k=*/5,
                                                  /*seed=*/1, Backend::Cpu,
                                                  /*max_candidates=*/64);
        REQUIRE(job.ok());
        const std::string job_digest = job.value().job_digest_sha256();

        SearchScheduler::LoopOptions loop;
        // Same created_utc → identical empty prior digests; distinct batch_ids for
        // artifact paths (search-loop.md: priors do not unexpectedly change).
        loop.created_utcs = {"2026-09-22T18:00:01Z", "2026-09-22T18:00:01Z"};
        loop.max_iterations = 2;
        loop.omit_timing = true;
        loop.batch_ids = {"b-g26-caesar-0001", "b-g26-caesar-0002"};

        StatusOr<SearchScheduler::CycleResult> result =
            SearchScheduler::run_loop(root, ctx, job.value(), loop);
        if (!result.ok()) {
            FAIL(result.status().message());
        }
        REQUIRE(result.value().iterations() == 2);
        REQUIRE(result.value().stop_reason() == SearchScheduler::stop_completed_iterations);
        REQUIRE(result.value().batches().size() == 2);
        REQUIRE(result.value().hypotheses_written() == 10);

        StatusOr<BatchArtifact> b1 = BatchArtifact::load(root, "g26-ws", "b-g26-caesar-0001");
        StatusOr<BatchArtifact> b2 = BatchArtifact::load(root, "g26-ws", "b-g26-caesar-0002");
        REQUIRE(b1.ok());
        REQUIRE(b2.ok());

        // Proposed ingest does not seed/exclude → prior digest stable across iterations.
        REQUIRE(b1.value().prior_digest_sha256() == b2.value().prior_digest_sha256());
        REQUIRE(b1.value().job_digest_sha256() == job_digest);
        REQUIRE(b2.value().job_digest_sha256() == job_digest);
        REQUIRE(candidate_ids(b1.value()) == candidate_ids(b2.value()));
        REQUIRE(candidate_scores(b1.value()) == candidate_scores(b2.value()));

        // Scheduler/bridge MUST NOT mutate fixture ciphertext (search-loop.md).
        REQUIRE(fixture_ciphertext_sha256(root) == fixture_sha_before);

        nlohmann::json snapshot{
            {"cycle", result.value().to_json()},
            {"job_digest", job_digest},
            {"prior_digest", b1.value().prior_digest_sha256()},
            {"ids1", candidate_ids(b1.value())},
            {"ids2", candidate_ids(b2.value())},
            {"scores1", candidate_scores(b1.value())},
            {"scores2", candidate_scores(b2.value())},
            {"manifest1", b1.value().manifest_to_json()},
            {"manifest2", b2.value().manifest_to_json()},
        };

        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        return snapshot;
    };

    const nlohmann::json a = run_twice("parcae_search_scheduler_g26_a");
    const nlohmann::json b = run_twice("parcae_search_scheduler_g26_b");
    REQUIRE(a == b);
}

TEST_CASE("SearchScheduler two-iteration CPU loop with reject feedback is deterministic",
          "[search][scheduler][loop][determinism]") {
    // Exit-criteria path: job → artifact → hypotheses → prior → second iteration.
    auto run_feedback = [](std::string_view sandbox_name) {
        const auto root = make_sandbox(sandbox_name);
        const Context ctx{root};

        StatusOr<WorkspaceManifest> ws =
            make_fixture_workspace("g26-fb-ws", "2026-09-22T18:00:00Z");
        REQUIRE(ws.ok());
        REQUIRE(ws.value().store(root).ok());

        StatusOr<SearchJob> job = SearchJob::make("g26-fb-ws", "caesar", "chi2_english_gp_v0",
                                                  /*k=*/5,
                                                  /*seed=*/1, Backend::Cpu,
                                                  /*max_candidates=*/64);
        REQUIRE(job.ok());

        SearchScheduler::Options first;
        first.created_utc = "2026-09-22T18:00:01Z";
        first.batch_id = "b-g26-fb-0001";
        StatusOr<SearchScheduler::CycleResult> cycle1 =
            SearchScheduler::run_once(root, ctx, job.value(), first);
        if (!cycle1.ok()) {
            FAIL(cycle1.status().message());
        }

        StatusOr<BatchArtifact> b1 = BatchArtifact::load(root, "g26-fb-ws", "b-g26-fb-0001");
        REQUIRE(b1.ok());
        const std::string top_id = b1.value().candidates()[0].at("candidate_id").get<std::string>();
        const int rejected_shift =
            b1.value().candidates()[0].at("envelope").at("params").at("shift").get<int>();
        const std::string hid =
            HypothesisBridge::hypothesis_id_for("g26-fb-ws", "b-g26-fb-0001", top_id);
        StatusOr<HypothesisRecord> hyp = HypothesisRecord::load(root, "g26-fb-ws", hid);
        REQUIRE(hyp.ok());
        REQUIRE(hyp.value().set_status(HypothesisStatus::Rejected).ok());
        REQUIRE(hyp.value().store(root).ok());

        SearchScheduler::Options second;
        second.created_utc = "2026-09-22T18:00:02Z";
        second.batch_id = "b-g26-fb-0002";
        StatusOr<SearchScheduler::CycleResult> cycle2 =
            SearchScheduler::run_once(root, ctx, job.value(), second);
        if (!cycle2.ok()) {
            FAIL(cycle2.status().message());
        }

        StatusOr<BatchArtifact> b2 = BatchArtifact::load(root, "g26-fb-ws", "b-g26-fb-0002");
        REQUIRE(b2.ok());
        REQUIRE(b2.value().prior_digest_sha256() != b1.value().prior_digest_sha256());
        for (const nlohmann::json& row : b2.value().candidates()) {
            REQUIRE(row.at("envelope").at("params").at("shift").get<int>() != rejected_shift);
        }

        nlohmann::json snapshot{
            {"job_digest", job.value().job_digest_sha256()},
            {"prior1", b1.value().prior_digest_sha256()},
            {"prior2", b2.value().prior_digest_sha256()},
            {"ids1", candidate_ids(b1.value())},
            {"ids2", candidate_ids(b2.value())},
            {"scores1", candidate_scores(b1.value())},
            {"scores2", candidate_scores(b2.value())},
            {"rejected_shift", rejected_shift},
        };

        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        return snapshot;
    };

    const nlohmann::json a = run_feedback("parcae_search_scheduler_g26_fb_a");
    const nlohmann::json b = run_feedback("parcae_search_scheduler_g26_fb_b");
    REQUIRE(a == b);
    REQUIRE(a.at("ids1") != a.at("ids2"));
}

namespace {

class SchedulerProgressRecordingSink : public ConsoleProgressSink {
public:
    void on_progress(const ConsoleProgressSnapshot& /*snapshot*/) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++progress_count;
    }

    void on_stage(std::string_view stage, const ConsoleProgressSnapshot& snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stages.emplace_back(stage);
        if (stage == "iteration" && snapshot.candidates_done() > 0) {
            iteration_dones.push_back(snapshot.candidates_done());
        }
    }

    std::mutex mutex_;
    std::size_t progress_count = 0;
    std::vector<std::string> stages;
    std::vector<std::size_t> iteration_dones;
};

} // namespace

TEST_CASE("SearchScheduler::run_loop emits iteration stages / digests match silent run",
          "[search][scheduler][loop][progress]") {
    const auto root_silent = make_sandbox("parcae_search_scheduler_g28_silent");
    const auto root_live = make_sandbox("parcae_search_scheduler_g28_live");
    const Context ctx_silent{root_silent};
    const Context ctx_live{root_live};

    StatusOr<WorkspaceManifest> ws_silent =
        make_fixture_workspace("g28-ws", "2026-09-22T17:00:00Z");
    REQUIRE(ws_silent.ok());
    REQUIRE(ws_silent.value().store(root_silent).ok());

    StatusOr<WorkspaceManifest> ws_live = make_fixture_workspace("g28-ws", "2026-09-22T17:00:00Z");
    REQUIRE(ws_live.ok());
    REQUIRE(ws_live.value().store(root_live).ok());

    StatusOr<SearchJob> job = SearchJob::make("g28-ws", "caesar", "chi2_english_gp_v0",
                                              /*k=*/2,
                                              /*seed=*/1, Backend::Cpu,
                                              /*max_candidates=*/64);
    REQUIRE(job.ok());

    SearchScheduler::LoopOptions loop_silent;
    loop_silent.created_utc = "2026-09-22T17:00:00Z";
    loop_silent.max_iterations = 2;
    loop_silent.omit_timing = true;
    loop_silent.batch_ids = {"b-g28-caesar-0001", "b-g28-caesar-0002"};

    SearchScheduler::LoopOptions loop_live = loop_silent;
    SchedulerProgressRecordingSink sink;
    loop_live.progress = &sink;

    StatusOr<SearchScheduler::CycleResult> silent =
        SearchScheduler::run_loop(root_silent, ctx_silent, job.value(), loop_silent);
    REQUIRE(silent.ok());

    StatusOr<SearchScheduler::CycleResult> live =
        SearchScheduler::run_loop(root_live, ctx_live, job.value(), loop_live);
    REQUIRE(live.ok());

    REQUIRE(silent.value().to_json() == live.value().to_json());
    REQUIRE(silent.value().batches()[0].job_digest_sha256() ==
            live.value().batches()[0].job_digest_sha256());
    REQUIRE(silent.value().batches()[1].job_digest_sha256() ==
            live.value().batches()[1].job_digest_sha256());

    REQUIRE(sink.iteration_dones.size() == 2);
    REQUIRE(sink.iteration_dones[0] == 1);
    REQUIRE(sink.iteration_dones[1] == 2);

    auto count_stage = [&](std::string_view name) {
        return static_cast<std::size_t>(
            std::count(sink.stages.begin(), sink.stages.end(), std::string(name)));
    };
    REQUIRE(count_stage("iteration") == 2);
    REQUIRE(count_stage("load") == 2);
    REQUIRE(count_stage("prior") == 2);
    REQUIRE(count_stage("write") == 2);
    REQUIRE(count_stage("bridge") == 2);
    REQUIRE(count_stage("expand") >= 2);
    REQUIRE(count_stage("score") >= 2);

    std::error_code ec;
    std::filesystem::remove_all(root_silent, ec);
    std::filesystem::remove_all(root_live, ec);
}
