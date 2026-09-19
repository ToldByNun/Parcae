#include <parcae/tool/agent_policy.hpp>
#include <parcae/tool/tool_response.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

#if defined(PARCAE_HAS_CLI_GOLDENS)
#include "parcae_cli_paths.h"
#if !defined(PARCAE_CLI_HYPOTHESIS) || !defined(PARCAE_CLI_SCORE)
#error "PARCAE_CLI_HYPOTHESIS and PARCAE_CLI_SCORE required"
#endif
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

#if defined(PARCAE_HAS_CLI_GOLDENS)
[[nodiscard]] std::string quote_arg(const std::string& arg) {
    return std::string("\"") + arg + '"';
}

[[nodiscard]] std::pair<int, std::string> run_cli(
    const std::filesystem::path& exe,
    const std::vector<std::string>& args) {
    const auto tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path out_path = tmp / "parcae_policy_e22_out.json";
    const std::filesystem::path err_path = tmp / "parcae_policy_e22_err.txt";
    const std::filesystem::path script_path = tmp / "parcae_policy_e22_run.cmd";

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

void expect_policy_denial(
    const std::filesystem::path& exe,
    const std::vector<std::string>& args,
    std::string_view expected_tool) {
    const auto [exit_code, stdout_text] = run_cli(exe, args);
    INFO(stdout_text);
    REQUIRE(exit_code == 2);

    StatusOr<nlohmann::json> envelope = ToolResponse::parse(stdout_text);
    REQUIRE(envelope.ok());
    REQUIRE_FALSE(envelope.value().at("ok").get<bool>());
    REQUIRE(envelope.value().at("tool").get<std::string>() == expected_tool);
    REQUIRE(envelope.value().at("result").is_null());
    REQUIRE(envelope.value().at("error").at("code").get<std::string>() == "policy");
    REQUIRE_FALSE(envelope.value().at("error").at("message").get<std::string>().empty());
}
#endif

}  // namespace

TEST_CASE(
    "AgentPolicy refuses writes when data_root sits under fixtures/",
    "[tool][policy][fixtures]") {
    const auto fixture_as_root =
        data_root() / "fixtures" / "solved" / "a-warning";
    REQUIRE(std::filesystem::is_directory(fixture_as_root));

    const AgentPolicy policy(fixture_as_root);
    Status denied = policy.allow_workspace_write("evil-ws", "hypotheses/h-evil.json");
    REQUIRE_FALSE(denied.ok());
    REQUIRE(AgentPolicy::error_code_for(denied) == ToolErrorCode::Policy);
    REQUIRE(denied.message().find("fixtures") != std::string::npos);

    // Direct write under the mis-pointed root is also denied.
    REQUIRE_FALSE(policy.allow_write(fixture_as_root / "workspaces" / "evil-ws" / "x.json").ok());
}

TEST_CASE(
    "AgentPolicy still allows normal workspace writes under real data_root",
    "[tool][policy][fixtures]") {
    const AgentPolicy policy(data_root());
    REQUIRE(policy.allow_workspace_write("policy-ok-ws", "hypotheses/h-ok.json").ok());
    REQUIRE_FALSE(
        policy.allow_write(data_root() / "fixtures" / "solved" / "a-warning" / "evil.json").ok());
}

#if defined(PARCAE_HAS_CLI_GOLDENS)
TEST_CASE(
    "CLI hypothesis init with --data-dir inside fixtures yields policy envelope exit 2",
    "[tool][policy][cli][fixtures]") {
    const auto fixture_dir = data_root() / "fixtures" / "solved" / "a-warning";
    const auto cipher_before = fixture_dir / "ciphertext.txt";
    REQUIRE(std::filesystem::exists(cipher_before));
    const auto size_before = std::filesystem::file_size(cipher_before);

    expect_policy_denial(
        PARCAE_CLI_HYPOTHESIS,
        {"--data-dir",
         fixture_dir.string(),
         "init",
         "--workspace",
         "evil-ws",
         "--id",
         "h-evil",
         "--utc",
         "2026-09-20T00:00:00Z",
         "--json"},
        "hypothesis_init");

    // Fixture corpus untouched — no workspaces/ tree created under a-warning.
    REQUIRE(std::filesystem::exists(cipher_before));
    REQUIRE(std::filesystem::file_size(cipher_before) == size_before);
    REQUIRE_FALSE(std::filesystem::exists(fixture_dir / "workspaces"));
}

TEST_CASE(
    "CLI score --backend cuda without --allow-cuda yields policy envelope exit 2",
    "[tool][policy][cli]") {
    expect_policy_denial(
        PARCAE_CLI_SCORE,
        {"--data-dir",
         std::string(PARCAE_TEST_DATA_DIR),
         "--score-id",
         "ic_mod29",
         "--indices",
         "--backend",
         "cuda",
         "--input",
         (data_root() / "fixtures" / "cli" / "score-indices.txt").string(),
         "--json"},
        "score");
}
#endif
