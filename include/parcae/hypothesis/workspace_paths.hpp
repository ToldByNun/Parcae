#ifndef WORKSPACE_PATHS_HPP
#define WORKSPACE_PATHS_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <filesystem>
#include <string>
#include <string_view>

/// Path-safe helpers for `data/workspaces/<id>/` (HypothesisRecord I/O).
class WorkspacePaths {
public:
    /// `[a-z_][a-z0-9_-]{0,63}` — shared by workspace_id and hypothesis_id.
    /// Leading `_` is allowed for reserved trees such as `_example`.
    [[nodiscard]] static StatusOr<std::string> validate_id(std::string_view id) {
        if (id.empty() || id.size() > 64) {
            return Status::error("workspace/hypothesis id length must be 1..64");
        }
        const unsigned char first = static_cast<unsigned char>(id[0]);
        if (!((first >= 'a' && first <= 'z') || first == '_')) {
            return Status::error(
                "workspace/hypothesis id must start with a lowercase letter or '_': " +
                std::string(id));
        }
        for (char ch : id) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
                continue;
            }
            return Status::error(
                "workspace/hypothesis id has illegal character: " + std::string(id));
        }
        return std::string(id);
    }

    [[nodiscard]] static std::filesystem::path workspaces_root(
        const std::filesystem::path& data_root) {
        return data_root / "workspaces";
    }

    [[nodiscard]] static StatusOr<std::filesystem::path> workspace_root(
        const std::filesystem::path& data_root,
        std::string_view workspace_id) {
        StatusOr<std::string> id = validate_id(workspace_id);
        if (!id.ok()) {
            return id.status();
        }
        return workspaces_root(data_root) / id.value();
    }

    [[nodiscard]] static StatusOr<std::filesystem::path> hypotheses_dir(
        const std::filesystem::path& data_root,
        std::string_view workspace_id) {
        StatusOr<std::filesystem::path> root = workspace_root(data_root, workspace_id);
        if (!root.ok()) {
            return root.status();
        }
        return root.value() / "hypotheses";
    }

    [[nodiscard]] static StatusOr<std::filesystem::path> hypothesis_file(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view hypothesis_id) {
        StatusOr<std::string> hid = validate_id(hypothesis_id);
        if (!hid.ok()) {
            return hid.status();
        }
        StatusOr<std::filesystem::path> dir = hypotheses_dir(data_root, workspace_id);
        if (!dir.ok()) {
            return dir.status();
        }
        return dir.value() / (hid.value() + ".json");
    }

    [[nodiscard]] static StatusOr<std::filesystem::path> batches_dir(
        const std::filesystem::path& data_root,
        std::string_view workspace_id) {
        StatusOr<std::filesystem::path> root = workspace_root(data_root, workspace_id);
        if (!root.ok()) {
            return root.status();
        }
        return root.value() / "batches";
    }

    [[nodiscard]] static StatusOr<std::filesystem::path> batch_dir(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view batch_id) {
        StatusOr<std::string> bid = validate_id(batch_id);
        if (!bid.ok()) {
            return bid.status();
        }
        StatusOr<std::filesystem::path> dir = batches_dir(data_root, workspace_id);
        if (!dir.ok()) {
            return dir.status();
        }
        return dir.value() / bid.value();
    }

    /// Reject absolute paths and any `..` segment; require result under `root`.
    [[nodiscard]] static StatusOr<std::filesystem::path> resolve_under(
        const std::filesystem::path& root,
        const std::filesystem::path& relative) {
        if (relative.empty()) {
            return Status::error("relative path must be non-empty");
        }
        if (relative.is_absolute()) {
            return Status::error("absolute paths are not allowed inside a workspace");
        }
        for (const std::filesystem::path& part : relative) {
            if (part == "..") {
                return Status::error("path traversal ('..') is not allowed");
            }
        }

        std::error_code ec;
        const std::filesystem::path root_canon =
            std::filesystem::weakly_canonical(root, ec);
        if (ec) {
            return Status::error("Failed to canonicalize workspace root: " + ec.message());
        }
        const std::filesystem::path joined = root_canon / relative;
        const std::filesystem::path resolved = std::filesystem::weakly_canonical(joined, ec);
        if (ec) {
            // Parent may not exist yet for writes — fall back to lexically_normal.
            const std::filesystem::path normalized = joined.lexically_normal();
            Status under = require_under(root_canon, normalized);
            if (!under.ok()) {
                return under;
            }
            return normalized;
        }

        Status under = require_under(root_canon, resolved);
        if (!under.ok()) {
            return under;
        }
        return resolved;
    }

    [[nodiscard]] static Status require_under(
        const std::filesystem::path& root_canon,
        const std::filesystem::path& candidate) {
        const std::filesystem::path rel = candidate.lexically_relative(root_canon);
        if (rel.empty() || rel == ".") {
            return Status::success();
        }
        if (*rel.begin() == "..") {
            return Status::error("resolved path escapes workspace root");
        }
        return Status::success();
    }

    /// Writes MUST NOT land under `data_root/fixtures/`.
    [[nodiscard]] static Status deny_fixtures_write(
        const std::filesystem::path& data_root,
        const std::filesystem::path& write_path) {
        std::error_code ec;
        const std::filesystem::path fixtures =
            std::filesystem::weakly_canonical(data_root / "fixtures", ec);
        if (ec) {
            // No fixtures dir — nothing to deny.
            return Status::success();
        }
        const std::filesystem::path target = write_path.lexically_normal();
        const std::filesystem::path rel = target.lexically_relative(fixtures);
        if (!rel.empty() && *rel.begin() != "..") {
            return Status::error("writes under data/fixtures/ are denied");
        }
        // Also catch when target is fixtures itself.
        if (target == fixtures) {
            return Status::error("writes under data/fixtures/ are denied");
        }
        return Status::success();
    }

private:
    WorkspacePaths() = delete;
};

#endif  // WORKSPACE_PATHS_HPP
