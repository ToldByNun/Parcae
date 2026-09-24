#ifndef AGENT_POLICY_HPP
#define AGENT_POLICY_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/tool/tool_response.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Path / backend / tool-name guards for agent-facing CLIs.
///
/// Denials map to `ToolErrorCode::Policy` (exit 2). Missing CUDA in the build
/// remains `NotBuilt` via `BackendUtil::ensure_usable` — call that after
/// `allow_backend` succeeds.
class AgentPolicy {
public:
    explicit AgentPolicy(std::filesystem::path data_root, bool allow_cuda = false)
        : data_root_(std::move(data_root)), allow_cuda_(allow_cuda) {}

    [[nodiscard]] const std::filesystem::path& data_root() const noexcept {
        return data_root_;
    }

    [[nodiscard]] bool allow_cuda() const noexcept {
        return allow_cuda_;
    }

    void set_allow_cuda(bool enabled) noexcept {
        allow_cuda_ = enabled;
    }

    /// Default agent tool names from `docs/spec/agent-tools.md`.
    [[nodiscard]] static const std::vector<std::string_view>& default_allow_list() {
        static const std::vector<std::string_view> kAllow = {
            "tokenize",
            "decode",
            "score",
            "validate",
            "catalog",
            "generate",
            "rank",
            "hypothesis_init",
            "hypothesis_propose",
            "hypothesis_show",
            "hypothesis_list",
            "hypothesis_score",
            "hypothesis_set_status",
            "search_cycle",
        };
        return kAllow;
    }

    /// Default deny-listed CLI basenames (no `parcae-` prefix required).
    [[nodiscard]] static const std::vector<std::string_view>& default_deny_binaries() {
        static const std::vector<std::string_view> kDeny = {
            "blind-crack",
            "parcae-blind-crack",
            "bench",
            "parcae-bench",
            "throughput-tiers",
            "parcae-throughput-tiers",
            "parity",
            "parcae-parity",
            "parity-gen",
            "parcae-parity-gen",
            "search-run",
            "parcae-search-run",
        };
        return kDeny;
    }

    [[nodiscard]] static bool is_default_allowed_tool(std::string_view tool) {
        const auto& allow = default_allow_list();
        return std::find(allow.begin(), allow.end(), tool) != allow.end();
    }

    [[nodiscard]] static bool is_default_denied_binary(std::string_view name) {
        const auto& deny = default_deny_binaries();
        return std::find(deny.begin(), deny.end(), name) != deny.end();
    }

    /// Allow-list tool name (agent function id), not argv0.
    [[nodiscard]] Status allow_tool(std::string_view tool) const {
        if (is_default_allowed_tool(tool)) {
            return Status::success();
        }
        return Status::error(
            "AgentPolicy: tool not on default allow-list: " + std::string(tool));
    }

    /// Deny-list check for a CLI basename (`parcae-blind-crack` / `blind-crack`).
    [[nodiscard]] Status allow_binary(std::string_view basename) const {
        if (is_default_denied_binary(basename)) {
            return Status::error(
                "AgentPolicy: binary is deny-listed by default: " + std::string(basename));
        }
        return Status::success();
    }

    /// `cpu` always OK. `cuda` requires `allow_cuda` opt-in (else policy).
    [[nodiscard]] Status allow_backend(parcae::tool::Backend backend) const {
        if (backend == parcae::tool::Backend::Cpu) {
            return Status::success();
        }
        if (!allow_cuda_) {
            return Status::error(
                "AgentPolicy: CUDA backend requires explicit allow_cuda / --allow-cuda");
        }
        return Status::success();
    }

    /// Resolve `backend` string then apply `allow_backend`.
    [[nodiscard]] StatusOr<parcae::tool::Backend> check_backend_string(
        std::string_view backend_text) const {
        StatusOr<parcae::tool::Backend> backend =
            parcae::tool::BackendUtil::from_string(backend_text);
        if (!backend.ok()) {
            return backend.status();
        }
        Status allowed = allow_backend(backend.value());
        if (!allowed.ok()) {
            return allowed;
        }
        return backend;
    }

    /// Writes MUST stay under `data_root` and MUST NOT land under `fixtures/`.
    /// Also rejects a mis-pointed `--data-dir` that itself sits under `fixtures/`.
    [[nodiscard]] Status allow_write(const std::filesystem::path& write_path) const {
        Status root_ok = deny_data_root_inside_fixtures();
        if (!root_ok.ok()) {
            return root_ok;
        }
        Status under = require_under_data_root(write_path);
        if (!under.ok()) {
            return under;
        }
        Status fixtures = WorkspacePaths::deny_fixtures_write(data_root_, write_path);
        if (!fixtures.ok()) {
            return Status::error(std::string("AgentPolicy: ") + fixtures.message());
        }
        return Status::success();
    }

    /// Reads SHOULD stay under `data_root` (fixtures are readable).
    [[nodiscard]] Status allow_read(const std::filesystem::path& read_path) const {
        return require_under_data_root(read_path);
    }

    /// Workspace-relative write: `data_root/workspaces/<id>/<relative>`.
    [[nodiscard]] Status allow_workspace_write(
        std::string_view workspace_id,
        const std::filesystem::path& relative) const {
        Status root_ok = deny_data_root_inside_fixtures();
        if (!root_ok.ok()) {
            return root_ok;
        }
        StatusOr<std::filesystem::path> ws =
            WorkspacePaths::workspace_root(data_root_, workspace_id);
        if (!ws.ok()) {
            return ws.status();
        }
        StatusOr<std::filesystem::path> resolved =
            WorkspacePaths::resolve_under(ws.value(), relative);
        if (!resolved.ok()) {
            return Status::error(std::string("AgentPolicy: ") + resolved.status().message());
        }
        return allow_write(resolved.value());
    }

    /// Map a Status denial to the agent envelope code (policy vs usage/io).
    [[nodiscard]] static ToolErrorCode error_code_for(const Status& status) {
        if (status.ok()) {
            return ToolErrorCode::Internal;
        }
        const std::string& msg = status.message();
        if (msg.find("AgentPolicy:") == 0 || msg.find("writes under data/fixtures/") != std::string::npos ||
            msg.find("path traversal") != std::string::npos ||
            msg.find("escapes") != std::string::npos ||
            msg.find("absolute paths are not allowed") != std::string::npos ||
            msg.find("deny-listed") != std::string::npos ||
            msg.find("allow-list") != std::string::npos ||
            msg.find("allow_cuda") != std::string::npos ||
            msg.find("under fixtures/") != std::string::npos) {
            return ToolErrorCode::Policy;
        }
        return ToolErrorCode::Internal;
    }

private:
    /// `--data-dir` must not point at (or inside) the corpus `fixtures/` tree.
    [[nodiscard]] Status deny_data_root_inside_fixtures() const {
        std::error_code ec;
        std::filesystem::path cur = std::filesystem::weakly_canonical(data_root_, ec);
        if (ec) {
            cur = data_root_.lexically_normal();
        }
        while (true) {
            if (cur.filename() == "fixtures") {
                return Status::error(
                    "AgentPolicy: data_root must not be under fixtures/ (refusing writes)");
            }
            const std::filesystem::path parent = cur.parent_path();
            if (parent.empty() || parent == cur) {
                break;
            }
            cur = parent;
        }
        return Status::success();
    }

    [[nodiscard]] Status require_under_data_root(const std::filesystem::path& candidate) const {
        std::error_code ec;
        const std::filesystem::path root_canon =
            std::filesystem::weakly_canonical(data_root_, ec);
        if (ec) {
            return Status::error(
                "AgentPolicy: failed to canonicalize data_root: " + ec.message());
        }

        std::filesystem::path target = candidate;
        if (!target.is_absolute()) {
            target = root_canon / target;
        }
        const std::filesystem::path resolved = std::filesystem::weakly_canonical(target, ec);
        if (ec) {
            const std::filesystem::path normalized = target.lexically_normal();
            Status under = WorkspacePaths::require_under(root_canon, normalized);
            if (!under.ok()) {
                return Status::error(
                    "AgentPolicy: path escapes data_root: " + normalized.string());
            }
            return Status::success();
        }

        Status under = WorkspacePaths::require_under(root_canon, resolved);
        if (!under.ok()) {
            return Status::error("AgentPolicy: path escapes data_root: " + resolved.string());
        }
        return Status::success();
    }

    std::filesystem::path data_root_;
    bool allow_cuda_ = false;
};

#endif  // AGENT_POLICY_HPP
