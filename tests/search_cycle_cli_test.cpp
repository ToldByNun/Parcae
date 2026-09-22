#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/search_scheduler.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

#if defined(PARCAE_HAS_CLI_GOLDENS)
#include "parcae_cli_paths.h"
#if !defined(PARCAE_CLI_SEARCH_CYCLE)
#error "PARCAE_CLI_SEARCH_CYCLE required"
#endif
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

    auto copy = [&](const std::filesystem::path& rel) {
        std::filesystem::copy_file(
            repo_data() / rel,
            root / rel,
            std::filesystem::copy_options::overwrite_existing,
            ec);
        REQUIRE(!ec);
    };
    copy("profiles/scores/english-gp-expected-v0.json");
    copy("profiles/gematria/gematria-primus-v0.json");
    copy("profiles/separators/rtkd-separator-grammar-v0.json");
    copy("fixtures/solved/a-warning/ciphertext.txt");
    copy("fixtures/solved/a-warning/manifest.json");
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
        {"title", "h28"},
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

#if defined(PARCAE_HAS_CLI_GOLDENS)
[[nodiscard]] std::string quote_arg(const std::string& arg) {
    return std::string("\"") + arg + '"';
}

[[nodiscard]] std::pair<int, std::string> run_cli(
    const std::filesystem::path& exe,
    const std::vector<std::string>& args) {
    const auto tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path out_path = tmp / "parcae_search_cycle_h28_out.json";
    const std::filesystem::path err_path = tmp / "parcae_search_cycle_h28_err.txt";
    const std::filesystem::path script_path = tmp / "parcae_search_cycle_h28_run.cmd";

    {
        std::ofstream script(script_path, std::ios::binary);
        REQUIRE(script);
        script << "@echo off\r\n";
        script << quote_arg(exe.string());
        for (const std::string& arg : args) {
            script << ' ' << quote_arg(arg);
        }
        script << " >" << quote_arg(out_path.string()) << " 2>"
               << quote_arg(err_path.string()) << "\r\n";
        script << "exit /B %ERRORLEVEL%\r\n";
    }

    const int exit_code =
        std::system((std::string("cmd /C ") + quote_arg(script_path.string())).c_str());

    std::string stdout_text;
    if (std::filesystem::exists(out_path)) {
        std::ifstream in(out_path, std::ios::binary);
        std::ostringstream buf;
        buf << in.rdbuf();
        stdout_text = buf.str();
    }

    std::error_code ec;
    std::filesystem::remove(out_path, ec);
    std::filesystem::remove(err_path, ec);
    std::filesystem::remove(script_path, ec);
    return {exit_code, stdout_text};
}
#endif

}  // namespace

#if defined(PARCAE_HAS_CLI_GOLDENS)

TEST_CASE(
    "parcae-search-cycle --status --json reports run_ready",
    "[tool][search_cycle][status]") {
    const auto [code, out] = run_cli(
        PARCAE_CLI_SEARCH_CYCLE,
        {"--status", "--json", "--data-dir", std::string(PARCAE_TEST_DATA_DIR)});
    REQUIRE(code == 0);
    const nlohmann::json envelope = nlohmann::json::parse(out);
    REQUIRE(envelope.at("ok").get<bool>());
    REQUIRE(envelope.at("tool").get<std::string>() == "search_cycle");
    REQUIRE(envelope.at("result").at("run_ready").get<bool>());
    REQUIRE(envelope.at("result").at("scheduler_ready").get<bool>());
    REQUIRE(
        envelope.at("result").at("search_cycle_result_schema").get<std::string>() ==
        SearchScheduler::result_schema_id);
}

TEST_CASE(
    "parcae-search-cycle --workspace --family --backend cpu --json runs one cycle",
    "[tool][search_cycle]") {
    const auto root = make_sandbox("parcae_search_cycle_h28_run");
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("h28-ws", "2026-09-22T19:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    const auto [code, out] = run_cli(
        PARCAE_CLI_SEARCH_CYCLE,
        {"--workspace",
         "h28-ws",
         "--family",
         "atbash",
         "--k",
         "1",
         "--iterations",
         "1",
         "--backend",
         "cpu",
         "--json",
         "--data-dir",
         root.string()});
    REQUIRE(code == 0);
    const nlohmann::json envelope = nlohmann::json::parse(out);
    REQUIRE(envelope.at("ok").get<bool>());
    REQUIRE(envelope.at("tool").get<std::string>() == "search_cycle");
    REQUIRE(envelope.at("backend").get<std::string>() == "cpu");
    const nlohmann::json& result = envelope.at("result");
    REQUIRE(result.at("schema").get<std::string>() == "parcae.search_cycle_result.v0");
    REQUIRE(result.at("workspace_id").get<std::string>() == "h28-ws");
    REQUIRE(result.at("iterations").get<std::size_t>() == 1);
    REQUIRE(result.at("stop_reason").get<std::string>() == "completed_iterations");
    REQUIRE(result.at("hypotheses_written").get<std::size_t>() == 1);
    REQUIRE(result.at("batches").is_array());
    REQUIRE(result.at("batches").size() == 1);

    const std::string batch_id = result.at("batches")[0].at("batch_id").get<std::string>();
    StatusOr<BatchArtifact> loaded = BatchArtifact::load(root, "h28-ws", batch_id);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().family() == "atbash");

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "parcae-search-cycle --backend cuda without --allow-cuda is denied",
    "[tool][search_cycle][policy]") {
    const auto root = make_sandbox("parcae_search_cycle_h28_policy");
    StatusOr<WorkspaceManifest> ws =
        make_fixture_workspace("h28-pol-ws", "2026-09-22T19:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    const auto [code, out] = run_cli(
        PARCAE_CLI_SEARCH_CYCLE,
        {"--workspace",
         "h28-pol-ws",
         "--family",
         "caesar",
         "--k",
         "1",
         "--backend",
         "cuda",
         "--json",
         "--data-dir",
         root.string()});
    REQUIRE(code != 0);
    const nlohmann::json envelope = nlohmann::json::parse(out);
    REQUIRE_FALSE(envelope.at("ok").get<bool>());
    REQUIRE(envelope.at("error").at("code").get<std::string>() == "policy");

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "parcae-search-cycle --omit-timing requires --json",
    "[tool][search_cycle][omit-timing]") {
    const auto [code, out] = run_cli(
        PARCAE_CLI_SEARCH_CYCLE,
        {"--workspace",
         "_example",
         "--family",
         "atbash",
         "--omit-timing",
         "--data-dir",
         std::string(PARCAE_TEST_DATA_DIR)});
    REQUIRE(code != 0);
    (void)out;
}

TEST_CASE(
    "parcae-search-cycle --omit-timing --created-utc yields replayable digests",
    "[tool][search_cycle][omit-timing][determinism]") {
    auto run_once = [](std::string_view sandbox, std::string_view workspace_id) {
        const auto root = make_sandbox(sandbox);
        StatusOr<WorkspaceManifest> ws =
            make_fixture_workspace(workspace_id, "2026-09-22T20:00:00Z");
        REQUIRE(ws.ok());
        REQUIRE(ws.value().store(root).ok());

        const auto [code, out] = run_cli(
            PARCAE_CLI_SEARCH_CYCLE,
            {"--workspace",
             std::string(workspace_id),
             "--family",
             "caesar",
             "--k",
             "3",
             "--seed",
             "1",
             "--iterations",
             "1",
             "--backend",
             "cpu",
             "--created-utc",
             "2026-09-22T20:00:01Z",
             "--json",
             "--omit-timing",
             "--data-dir",
             root.string()});
        REQUIRE(code == 0);
        const nlohmann::json envelope = nlohmann::json::parse(out);
        REQUIRE(envelope.at("ok").get<bool>());
        const nlohmann::json& result = envelope.at("result");
        REQUIRE(result.at("omit_timing").get<bool>());
        REQUIRE_FALSE(result.contains("tok_per_sec"));
        REQUIRE_FALSE(result.contains("wall_ms"));

        const std::string batch_id = result.at("batches")[0].at("batch_id").get<std::string>();
        StatusOr<BatchArtifact> batch =
            BatchArtifact::load(root, workspace_id, batch_id);
        REQUIRE(batch.ok());
        REQUIRE_FALSE(batch.value().report_relpath().has_value());
        REQUIRE_FALSE(batch.value().report().has_value());

        nlohmann::json snapshot{
            {"result", result},
            {"job_digest", batch.value().job_digest_sha256()},
            {"prior_digest", batch.value().prior_digest_sha256()},
            {"batch_id", batch_id},
            {"candidate_ids", nlohmann::json::array()},
        };
        for (const nlohmann::json& row : batch.value().candidates()) {
            snapshot["candidate_ids"].push_back(row.at("candidate_id"));
        }

        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        return snapshot;
    };

    const nlohmann::json a = run_once("parcae_search_cycle_h29_a", "h29-ws");
    const nlohmann::json b = run_once("parcae_search_cycle_h29_b", "h29-ws");
    // Same workspace_id + created-utc + seed ⇒ identical digests and batch_id across sandboxes.
    REQUIRE(a.at("job_digest") == b.at("job_digest"));
    REQUIRE(a.at("prior_digest") == b.at("prior_digest"));
    REQUIRE(a.at("batch_id") == b.at("batch_id"));
    REQUIRE(a.at("candidate_ids") == b.at("candidate_ids"));
    REQUIRE(a.at("result").at("stop_reason") == b.at("result").at("stop_reason"));
    REQUIRE(a.at("result").at("hypotheses_written") == b.at("result").at("hypotheses_written"));
}

#endif  // PARCAE_HAS_CLI_GOLDENS
