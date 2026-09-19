#ifndef WORKSPACE_MANIFEST_HPP
#define WORKSPACE_MANIFEST_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

/// Workspace root manifest (`parcae.workspace.v0`).
class WorkspaceManifest {
public:
    static constexpr std::string_view schema_id = "parcae.workspace.v0";

    WorkspaceManifest() = default;

    [[nodiscard]] const std::string& id() const noexcept {
        return id_;
    }

    [[nodiscard]] const std::string& default_score_id() const noexcept {
        return default_score_id_;
    }

    [[nodiscard]] const std::string& default_score_version() const noexcept {
        return default_score_version_;
    }

    [[nodiscard]] nlohmann::json to_json() const {
        return nlohmann::json{
            {"schema", std::string(schema_id)},
            {"id", id_},
            {"created_utc", created_utc_},
            {"updated_utc", updated_utc_},
            {"title", title_},
            {"notes", notes_},
            {"input", input_},
            {"default_score_id",
             default_score_id_.empty() ? nlohmann::json(nullptr)
                                       : nlohmann::json(default_score_id_)},
            {"default_score_version", default_score_version_},
        };
    }

    [[nodiscard]] static StatusOr<WorkspaceManifest> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("WorkspaceManifest must be a JSON object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error("WorkspaceManifest.schema must be parcae.workspace.v0");
        }
        if (!root.contains("id") || !root.at("id").is_string()) {
            return Status::error("WorkspaceManifest.id must be a string");
        }
        StatusOr<std::string> id = WorkspacePaths::validate_id(root.at("id").get<std::string>());
        if (!id.ok()) {
            return id.status();
        }

        WorkspaceManifest m;
        m.id_ = std::move(id.value());
        if (!root.contains("created_utc") || !root.at("created_utc").is_string()) {
            return Status::error("WorkspaceManifest.created_utc must be a string");
        }
        if (!root.contains("updated_utc") || !root.at("updated_utc").is_string()) {
            return Status::error("WorkspaceManifest.updated_utc must be a string");
        }
        m.created_utc_ = root.at("created_utc").get<std::string>();
        m.updated_utc_ = root.at("updated_utc").get<std::string>();
        if (root.contains("title") && root.at("title").is_string()) {
            m.title_ = root.at("title").get<std::string>();
        }
        if (root.contains("notes") && root.at("notes").is_string()) {
            m.notes_ = root.at("notes").get<std::string>();
        }
        if (!root.contains("input") || !root.at("input").is_object()) {
            return Status::error("WorkspaceManifest.input must be an object");
        }
        Status input_ok = validate_input(root.at("input"));
        if (!input_ok.ok()) {
            return input_ok;
        }
        m.input_ = root.at("input");
        if (root.contains("default_score_id") && root.at("default_score_id").is_string()) {
            m.default_score_id_ = root.at("default_score_id").get<std::string>();
        }
        if (root.contains("default_score_version") && root.at("default_score_version").is_string()) {
            m.default_score_version_ = root.at("default_score_version").get<std::string>();
        } else {
            m.default_score_version_ = "v0";
        }
        return m;
    }

    [[nodiscard]] static StatusOr<WorkspaceManifest> load(
        const std::filesystem::path& data_root,
        std::string_view workspace_id) {
        StatusOr<std::filesystem::path> root =
            WorkspacePaths::workspace_root(data_root, workspace_id);
        if (!root.ok()) {
            return root.status();
        }
        const std::filesystem::path path = root.value() / "workspace.json";
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("Failed to open workspace.json: " + path.string());
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        nlohmann::json json;
        try {
            json = nlohmann::json::parse(buf.str());
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid workspace.json: ") + ex.what());
        }
        StatusOr<WorkspaceManifest> m = from_json(json);
        if (!m.ok()) {
            return m.status();
        }
        if (m.value().id() != workspace_id) {
            return Status::error("workspace.json id does not match directory name");
        }
        return m;
    }

    [[nodiscard]] Status store(const std::filesystem::path& data_root) const {
        StatusOr<std::filesystem::path> root =
            WorkspacePaths::workspace_root(data_root, id_);
        if (!root.ok()) {
            return root.status();
        }
        Status fixtures = WorkspacePaths::deny_fixtures_write(data_root, root.value());
        if (!fixtures.ok()) {
            return fixtures;
        }
        std::error_code ec;
        std::filesystem::create_directories(root.value(), ec);
        if (ec) {
            return Status::error("Failed to create workspace directory: " + ec.message());
        }
        const std::filesystem::path path = root.value() / "workspace.json";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::error("Failed to write workspace.json: " + path.string());
        }
        out << to_json().dump(2) << '\n';
        if (!out) {
            return Status::error("Failed while writing workspace.json");
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<WorkspaceManifest> make(
        std::string_view workspace_id,
        std::string_view created_utc,
        std::string_view title = "",
        std::string_view default_score_id = "chi2_english_gp_v0") {
        StatusOr<std::string> id = WorkspacePaths::validate_id(workspace_id);
        if (!id.ok()) {
            return id.status();
        }
        WorkspaceManifest m;
        m.id_ = std::move(id.value());
        m.created_utc_ = std::string(created_utc);
        m.updated_utc_ = std::string(created_utc);
        m.title_ = std::string(title);
        m.input_ = nlohmann::json{
            {"kind", "inline_pending"},
            {"fixture_id", nullptr},
            {"path", nullptr},
        };
        m.default_score_id_ = std::string(default_score_id);
        m.default_score_version_ = "v0";
        return m;
    }

    /// Load if present; otherwise create a minimal manifest and store it.
    [[nodiscard]] static StatusOr<WorkspaceManifest> ensure(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view utc) {
        StatusOr<WorkspaceManifest> existing = load(data_root, workspace_id);
        if (existing.ok()) {
            return existing;
        }
        StatusOr<WorkspaceManifest> created = make(workspace_id, utc);
        if (!created.ok()) {
            return created.status();
        }
        Status stored = created.value().store(data_root);
        if (!stored.ok()) {
            return stored;
        }
        return created;
    }

private:
    [[nodiscard]] static Status validate_input(const nlohmann::json& input) {
        if (!input.contains("kind") || !input.at("kind").is_string()) {
            return Status::error("input.kind must be a string");
        }
        const std::string kind = input.at("kind").get<std::string>();
        if (kind == "fixture_ciphertext") {
            if (!input.contains("fixture_id") || !input.at("fixture_id").is_string()) {
                return Status::error("fixture_ciphertext requires input.fixture_id");
            }
            return Status::success();
        }
        if (kind == "workspace_file") {
            if (!input.contains("path") || !input.at("path").is_string()) {
                return Status::error("workspace_file requires input.path");
            }
            return Status::success();
        }
        if (kind == "inline_pending") {
            return Status::success();
        }
        return Status::error("Unknown input.kind: " + kind);
    }

    std::string id_;
    std::string created_utc_;
    std::string updated_utc_;
    std::string title_;
    std::string notes_;
    nlohmann::json input_ = nlohmann::json::object();
    std::string default_score_id_;
    std::string default_score_version_{"v0"};
};

#endif  // WORKSPACE_MANIFEST_HPP
