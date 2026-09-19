#include <parcae/core/index29.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/tool/tool_response.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

#if defined(PARCAE_HAS_CLI_GOLDENS)
#include "parcae_cli_paths.h"
#if !defined(PARCAE_CLI_HYPOTHESIS)
#error "PARCAE_CLI_HYPOTHESIS required for sandbox CLI tests"
#endif
#include <cstdlib>
#include <sstream>
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] std::filesystem::path make_sandbox_root(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    // Mirror a fixtures tree so deny_fixtures_write has something to protect.
    std::filesystem::create_directories(root / "fixtures" / "solved" / "a-warning", ec);
    {
        std::ofstream out(root / "fixtures" / "solved" / "a-warning" / "ciphertext.txt");
        out << "placeholder\n";
    }
    return root;
}

#if defined(PARCAE_HAS_CLI_GOLDENS)
[[nodiscard]] std::string quote_arg(const std::string& arg) {
    return std::string("\"") + arg + '"';
}

[[nodiscard]] std::pair<int, std::string> run_hypothesis_cli(const std::vector<std::string>& args) {
    const auto tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path out_path = tmp / "parcae_hyp_sandbox_out.json";
    const std::filesystem::path err_path = tmp / "parcae_hyp_sandbox_err.txt";
    const std::filesystem::path script_path = tmp / "parcae_hyp_sandbox_run.cmd";

    {
        std::ofstream script(script_path, std::ios::binary);
        REQUIRE(script);
        script << "@echo off\r\n";
        script << quote_arg(std::string(PARCAE_CLI_HYPOTHESIS));
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

TEST_CASE("sandbox: id validation blocks traversal tokens", "[hypothesis][sandbox][paths]") {
    REQUIRE_FALSE(WorkspacePaths::validate_id("..").ok());
    REQUIRE_FALSE(WorkspacePaths::validate_id("../evil").ok());
    REQUIRE_FALSE(WorkspacePaths::validate_id("a/b").ok());
    REQUIRE_FALSE(WorkspacePaths::validate_id("a\\b").ok());
    REQUIRE_FALSE(WorkspacePaths::validate_id(".hidden").ok());
    REQUIRE_FALSE(WorkspacePaths::hypothesis_file(data_root(), "..", "h-x").ok());
    REQUIRE_FALSE(WorkspacePaths::hypothesis_file(data_root(), "_example", "../h").ok());
}

TEST_CASE("sandbox: resolve_under rejects escape and absolute paths", "[hypothesis][sandbox][paths]") {
    StatusOr<std::filesystem::path> ws =
        WorkspacePaths::workspace_root(data_root(), "_example");
    REQUIRE(ws.ok());

    REQUIRE_FALSE(WorkspacePaths::resolve_under(ws.value(), "").ok());
    REQUIRE_FALSE(WorkspacePaths::resolve_under(ws.value(), "..").ok());
    REQUIRE_FALSE(WorkspacePaths::resolve_under(ws.value(), "../workspaces").ok());
    REQUIRE_FALSE(WorkspacePaths::resolve_under(ws.value(), "hypotheses/../../../fixtures").ok());
    REQUIRE_FALSE(
        WorkspacePaths::resolve_under(ws.value(), std::filesystem::path("/etc/passwd")).ok());
#if defined(_WIN32)
    REQUIRE_FALSE(
        WorkspacePaths::resolve_under(ws.value(), std::filesystem::path("C:/Windows/System32")).ok());
#endif

    StatusOr<std::filesystem::path> ok =
        WorkspacePaths::resolve_under(ws.value(), "hypotheses/h-atbash-example.json");
    REQUIRE(ok.ok());
    REQUIRE(WorkspacePaths::require_under(ws.value(), ok.value()).ok());
}

TEST_CASE("sandbox: fixture writes denied; workspace writes allowed", "[hypothesis][sandbox][paths]") {
    const auto root = make_sandbox_root("parcae_hypothesis_sandbox_paths");
    const auto fixtures_file =
        root / "fixtures" / "solved" / "a-warning" / "manifest.json";
    REQUIRE_FALSE(WorkspacePaths::deny_fixtures_write(root, fixtures_file).ok());
    REQUIRE_FALSE(
        WorkspacePaths::deny_fixtures_write(root, root / "fixtures").ok());

    StatusOr<HypothesisRecord> draft = HypothesisRecord::make_draft(
        "safe-ws",
        "h-ok",
        "2026-09-19T22:00:00Z",
        "sandbox",
        nlohmann::json{
            {"transform_id", "atbash"},
            {"direction", "decrypt"},
            {"params", nlohmann::json::object()},
        });
    REQUIRE(draft.ok());
    draft.value().recompute_method_digest();
    REQUIRE(draft.value().store(root).ok());
    REQUIRE(std::filesystem::exists(
        root / "workspaces" / "safe-ws" / "hypotheses" / "h-ok.json"));
    REQUIRE_FALSE(std::filesystem::exists(fixtures_file));

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("sandbox: digest verify catches method and indices tampering", "[hypothesis][sandbox][digest]") {
    StatusOr<HypothesisRecord> record = HypothesisRecord::make_draft(
        "digest-ws",
        "h-digest",
        "2026-09-19T22:00:00Z",
        "",
        nlohmann::json{
            {"transform_id", "caesar"},
            {"direction", "decrypt"},
            {"params", {{"shift", 3}}},
        });
    REQUIRE(record.ok());
    record.value().recompute_method_digest();
    REQUIRE(record.value().verify_method_digest().ok());

    // Nested key order must not affect the digest.
    const nlohmann::json shuffled{
        {"params", {{"shift", 3}}},
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
    };
    REQUIRE(
        HypothesisRecord::method_sha256(shuffled) ==
        record.value().digests().at("method_sha256").get<std::string>());

    // Tamper method without refreshing digest.
    REQUIRE(record.value()
                .set_method(nlohmann::json{
                    {"transform_id", "caesar"},
                    {"direction", "decrypt"},
                    {"params", {{"shift", 7}}},
                })
                .ok());
    REQUIRE_FALSE(record.value().verify_method_digest().ok());

    record.value().recompute_method_digest();
    REQUIRE(record.value().verify_method_digest().ok());

    const std::vector<Index29> plain = {I(0), I(1), I(2)};
    const std::vector<Index29> other = {I(0), I(1), I(3)};
    record.value().set_output_indices_digest(plain);
    REQUIRE(record.value().verify_output_indices_digest(plain).ok());
    REQUIRE_FALSE(record.value().verify_output_indices_digest(other).ok());
}

TEST_CASE(
    "sandbox: store+reload preserves digests and rejects mismatched workspace",
    "[hypothesis][sandbox][digest]") {
    const auto root = make_sandbox_root("parcae_hypothesis_sandbox_digest_io");

    StatusOr<WorkspaceManifest> ws = WorkspaceManifest::make(
        "digest-io-ws", "2026-09-19T22:00:00Z", "sandbox");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    StatusOr<HypothesisRecord> draft = HypothesisRecord::make_draft(
        "digest-io-ws",
        "h-io",
        "2026-09-19T22:00:00Z",
        "io",
        nlohmann::json{
            {"transform_id", "atbash"},
            {"direction", "decrypt"},
            {"params", nlohmann::json::object()},
        });
    REQUIRE(draft.ok());
    draft.value().recompute_method_digest();
    const std::vector<Index29> indices = {I(10), I(11), I(12)};
    draft.value().set_output_indices_digest(indices);
    REQUIRE(draft.value().store(root).ok());

    StatusOr<HypothesisRecord> loaded =
        HypothesisRecord::load(root, "digest-io-ws", "h-io");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().verify_method_digest().ok());
    REQUIRE(loaded.value().verify_output_indices_digest(indices).ok());
    REQUIRE_FALSE(
        loaded.value().verify_output_indices_digest(std::vector<Index29>{I(0)}).ok());

    // Corrupt on-disk workspace_id by rewriting file under another stem — load_file rejects.
    const auto path = root / "workspaces" / "digest-io-ws" / "hypotheses" / "h-io.json";
    REQUIRE_FALSE(HypothesisRecord::load_file(path, "digest-io-ws", "other-id").ok());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

#if defined(PARCAE_HAS_CLI_GOLDENS)
TEST_CASE("sandbox CLI: rejects traversal workspace/id", "[hypothesis][sandbox][cli]") {
    {
        const auto [exit_code, stdout_text] = run_hypothesis_cli(
            {"--data-dir",
             std::string(PARCAE_TEST_DATA_DIR),
             "init",
             "--workspace",
             "../evil",
             "--id",
             "h-x",
             "--json"});
        REQUIRE(exit_code != 0);
        StatusOr<nlohmann::json> envelope = ToolResponse::parse(stdout_text);
        REQUIRE(envelope.ok());
        REQUIRE_FALSE(envelope.value().at("ok").get<bool>());
        REQUIRE(envelope.value().at("tool").get<std::string>() == "hypothesis_init");
    }
    {
        const auto [exit_code, stdout_text] = run_hypothesis_cli(
            {"--data-dir",
             std::string(PARCAE_TEST_DATA_DIR),
             "init",
             "--workspace",
             "_example",
             "--id",
             "h/../x",
             "--json"});
        REQUIRE(exit_code != 0);
        StatusOr<nlohmann::json> envelope = ToolResponse::parse(stdout_text);
        REQUIRE(envelope.ok());
        REQUIRE_FALSE(envelope.value().at("ok").get<bool>());
    }
}

TEST_CASE("sandbox CLI: show _example and list stay read-only", "[hypothesis][sandbox][cli]") {
    const auto [list_exit, list_out] = run_hypothesis_cli(
        {"--data-dir",
         std::string(PARCAE_TEST_DATA_DIR),
         "list",
         "--workspace",
         "_example",
         "--json"});
    REQUIRE(list_exit == 0);
    StatusOr<nlohmann::json> listed = ToolResponse::parse(list_out);
    REQUIRE(listed.ok());
    REQUIRE(listed.value().at("ok").get<bool>());
    REQUIRE(listed.value().at("result").at("count").get<std::size_t>() >= 1);

    // Fixtures must remain untouched by read-only commands.
    REQUIRE(std::filesystem::exists(
        data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt"));
}
#endif
