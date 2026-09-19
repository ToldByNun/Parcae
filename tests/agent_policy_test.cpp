#include <parcae/tool/agent_policy.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/tool/tool_response.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

[[nodiscard]] std::filesystem::path make_policy_root(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "fixtures" / "solved" / "a-warning", ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    {
        std::ofstream out(root / "fixtures" / "solved" / "a-warning" / "ciphertext.txt");
        out << "x\n";
    }
    return root;
}

}  // namespace

TEST_CASE("AgentPolicy allow-list and deny-list", "[tool][policy]") {
    const AgentPolicy policy(data_root());

    REQUIRE(policy.allow_tool("tokenize").ok());
    REQUIRE(policy.allow_tool("hypothesis_score").ok());
    REQUIRE_FALSE(policy.allow_tool("blind_crack").ok());
    REQUIRE_FALSE(policy.allow_tool("shell").ok());

    REQUIRE(AgentPolicy::is_default_allowed_tool("rank"));
    REQUIRE(AgentPolicy::is_default_denied_binary("parcae-blind-crack"));
    REQUIRE(AgentPolicy::is_default_denied_binary("search-run"));
    REQUIRE(policy.allow_binary("parcae-tokenize").ok());
    REQUIRE_FALSE(policy.allow_binary("parcae-parity").ok());
    REQUIRE_FALSE(policy.allow_binary("throughput-tiers").ok());
}

TEST_CASE("AgentPolicy CUDA requires allow_cuda opt-in", "[tool][policy][backend]") {
    AgentPolicy denied(data_root(), /*allow_cuda=*/false);
    REQUIRE(denied.allow_backend(parcae::tool::Backend::Cpu).ok());
    REQUIRE_FALSE(denied.allow_backend(parcae::tool::Backend::Cuda).ok());
    REQUIRE(
        AgentPolicy::error_code_for(denied.allow_backend(parcae::tool::Backend::Cuda)) ==
        ToolErrorCode::Policy);

    StatusOr<parcae::tool::Backend> checked = denied.check_backend_string("cuda");
    REQUIRE_FALSE(checked.ok());
    REQUIRE(AgentPolicy::error_code_for(checked.status()) == ToolErrorCode::Policy);

    AgentPolicy allowed(data_root(), /*allow_cuda=*/true);
    REQUIRE(allowed.allow_backend(parcae::tool::Backend::Cuda).ok());
    // Build availability is orthogonal — ensure_usable may still fail without CUDA.
    Status usable = parcae::tool::BackendUtil::ensure_usable(parcae::tool::Backend::Cuda);
    if (!parcae::tool::BackendUtil::cuda_built()) {
        REQUIRE_FALSE(usable.ok());
    }
}

TEST_CASE("AgentPolicy denies fixture writes and path escape", "[tool][policy][paths]") {
    const auto root = make_policy_root("parcae_agent_policy_paths");
    const AgentPolicy policy(root);

    const auto fixture_write =
        root / "fixtures" / "solved" / "a-warning" / "evil.json";
    REQUIRE_FALSE(policy.allow_write(fixture_write).ok());
    REQUIRE(
        AgentPolicy::error_code_for(policy.allow_write(fixture_write)) == ToolErrorCode::Policy);

    REQUIRE(policy.allow_read(root / "fixtures" / "solved" / "a-warning" / "ciphertext.txt").ok());

    REQUIRE_FALSE(policy.allow_write(root / ".." / "outside.json").ok());
#if defined(_WIN32)
    REQUIRE_FALSE(policy.allow_read(std::filesystem::path("C:/Windows/System32")).ok());
#endif

    REQUIRE(policy.allow_workspace_write("safe-ws", "hypotheses/h-1.json").ok());
    REQUIRE_FALSE(policy.allow_workspace_write("safe-ws", "../fixtures/x.json").ok());
    REQUIRE_FALSE(policy.allow_workspace_write("..", "hypotheses/h-1.json").ok());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("AgentPolicy workspace write under data_root succeeds", "[tool][policy][paths]") {
    const auto root = make_policy_root("parcae_agent_policy_ws");
    AgentPolicy policy(root);

    Status allowed = policy.allow_workspace_write("ws-a", "hypotheses/h-ok.json");
    REQUIRE(allowed.ok());

    const auto abs =
        root / "workspaces" / "ws-a" / "hypotheses" / "h-ok.json";
    REQUIRE(policy.allow_write(abs).ok());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}
