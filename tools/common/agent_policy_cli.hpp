#ifndef AGENT_POLICY_CLI_HPP
#define AGENT_POLICY_CLI_HPP

#include "cli_io.hpp"
#include "parcae/tool/agent_policy.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "tool_cli_json.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Shared AgentPolicy wiring for allow-listed tool CLIs.
class AgentPolicyCli {
public:
    [[nodiscard]] static bool allow_cuda_flag(const std::vector<std::string>& args) {
        return CliIo::has_flag(args, "--allow-cuda");
    }

    [[nodiscard]] static AgentPolicy make(
        const Context& ctx,
        const std::vector<std::string>& args) {
        return AgentPolicy{ctx.data_root(), allow_cuda_flag(args)};
    }

    /// Parse `--backend`, enforce AgentPolicy, then `BackendUtil::ensure_usable`.
    /// On policy denial → `ToolErrorCode::Policy`; on missing build → `NotBuilt`.
    [[nodiscard]] static StatusOr<Backend> resolve_backend(
        const AgentPolicy& policy,
        const std::vector<std::string>& args,
        std::string_view default_backend = "cpu") {
        StatusOr<Backend> backend = policy.check_backend_string(
            CliIo::optional_option(args, "--backend", std::string(default_backend)));
        if (!backend.ok()) {
            return backend.status();
        }
        Status usable = BackendUtil::ensure_usable(backend.value());
        if (!usable.ok()) {
            return usable;
        }
        return backend;
    }

    [[nodiscard]] static ToolErrorCode backend_error_code(const Status& status) {
        if (status.ok()) {
            return ToolErrorCode::Internal;
        }
        if (AgentPolicy::error_code_for(status) == ToolErrorCode::Policy) {
            return ToolErrorCode::Policy;
        }
        const std::string& msg = status.message();
        if (msg.find("without CUDA") != std::string::npos ||
            msg.find("not available") != std::string::npos ||
            msg.find("PARCAE_BUILD_CUDA") != std::string::npos) {
            return ToolErrorCode::NotBuilt;
        }
        if (msg.find("Unknown backend") != std::string::npos) {
            return ToolErrorCode::Usage;
        }
        return ToolErrorCode::Internal;
    }

private:
    AgentPolicyCli() = delete;
};

#endif  // AGENT_POLICY_CLI_HPP
