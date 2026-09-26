#ifndef WORKSPACE_SCAFFOLD_HPP
#define WORKSPACE_SCAFFOLD_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/workspace_manifest.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Creates the canonical research workspace tree under `data/workspaces/<id>/`.
///
/// Layout (dirs always ensured; text templates only when missing):
/// ```text
/// README.md  SOURCE.txt  workspace.json
/// pages/  inputs/  hypotheses/  batches/  research/  transcripts/
/// inputs/INDEX.md  research/REPRODUCE.md
/// ```
/// Existing files are never overwritten. Manifest uses `workspace_file` →
/// `inputs/ciphertext.txt` when newly created (LP2 research recipe).
class WorkspaceScaffold {
public:
    class Options {
    public:
        Options() = default;

        /// Required when a new `workspace.json` must be created (RFC 3339 UTC).
        std::string created_utc;
        std::string title;
        std::string notes;
        std::string default_score_id{"chi2_english_gp_v0"};
    };

    class Result {
    public:
        [[nodiscard]] const std::string& workspace_id() const noexcept { return workspace_id_; }

        [[nodiscard]] bool created_manifest() const noexcept { return created_manifest_; }

        [[nodiscard]] bool created_readme() const noexcept { return created_readme_; }

        [[nodiscard]] bool created_source() const noexcept { return created_source_; }

        [[nodiscard]] bool created_inputs_index() const noexcept { return created_inputs_index_; }

        [[nodiscard]] bool created_reproduce() const noexcept { return created_reproduce_; }

        /// Relative paths under the workspace that were created (dirs + files).
        [[nodiscard]] const std::vector<std::string>& created() const noexcept { return created_; }

        [[nodiscard]] bool any_created() const noexcept { return !created_.empty(); }

    private:
        friend class WorkspaceScaffold;
        std::string workspace_id_;
        bool created_manifest_ = false;
        bool created_readme_ = false;
        bool created_source_ = false;
        bool created_inputs_index_ = false;
        bool created_reproduce_ = false;
        std::vector<std::string> created_;
    };

    /// Ensure the research tree exists. Idempotent: second call with the same
    /// options creates nothing new when the tree is already complete.
    [[nodiscard]] static StatusOr<Result> ensure_research(const std::filesystem::path& data_root,
                                                          std::string_view workspace_id,
                                                          const Options& options) {
        StatusOr<std::string> id = WorkspacePaths::validate_id(workspace_id);
        if (!id.ok()) {
            return id.status();
        }
        if (options.created_utc.empty()) {
            // Still allow ensure when manifest already exists; require utc for create.
            StatusOr<WorkspaceManifest> existing = WorkspaceManifest::load(data_root, id.value());
            if (!existing.ok()) {
                return Status::error(
                    "WorkspaceScaffold::ensure_research requires Options.created_utc when "
                    "workspace.json is missing");
            }
        }

        StatusOr<std::filesystem::path> root = WorkspacePaths::workspace_root(data_root, id.value());
        if (!root.ok()) {
            return root.status();
        }
        Status fixtures = WorkspacePaths::deny_fixtures_write(data_root, root.value());
        if (!fixtures.ok()) {
            return fixtures;
        }

        Result out;
        out.workspace_id_ = id.value();

        Status dirs = ensure_directories(data_root, id.value(), out);
        if (!dirs.ok()) {
            return dirs;
        }

        StatusOr<bool> manifest = ensure_manifest(data_root, id.value(), options, out);
        if (!manifest.ok()) {
            return manifest.status();
        }

        StatusOr<std::filesystem::path> readme = WorkspacePaths::readme_path(data_root, id.value());
        if (!readme.ok()) {
            return readme.status();
        }
        StatusOr<bool> wrote_readme =
            write_if_missing(readme.value(), readme_template(id.value(), options), out, "README.md");
        if (!wrote_readme.ok()) {
            return wrote_readme.status();
        }
        out.created_readme_ = wrote_readme.value();

        StatusOr<std::filesystem::path> source = WorkspacePaths::source_path(data_root, id.value());
        if (!source.ok()) {
            return source.status();
        }
        StatusOr<bool> wrote_source =
            write_if_missing(source.value(), source_template(), out, "SOURCE.txt");
        if (!wrote_source.ok()) {
            return wrote_source.status();
        }
        out.created_source_ = wrote_source.value();

        StatusOr<std::filesystem::path> inputs = WorkspacePaths::inputs_dir(data_root, id.value());
        if (!inputs.ok()) {
            return inputs.status();
        }
        StatusOr<bool> wrote_index =
            write_if_missing(inputs.value() / "INDEX.md", inputs_index_template(), out,
                             "inputs/INDEX.md");
        if (!wrote_index.ok()) {
            return wrote_index.status();
        }
        out.created_inputs_index_ = wrote_index.value();

        StatusOr<std::filesystem::path> research =
            WorkspacePaths::research_dir(data_root, id.value());
        if (!research.ok()) {
            return research.status();
        }
        StatusOr<bool> wrote_repro =
            write_if_missing(research.value() / "REPRODUCE.md", reproduce_template(id.value()), out,
                             "research/REPRODUCE.md");
        if (!wrote_repro.ok()) {
            return wrote_repro.status();
        }
        out.created_reproduce_ = wrote_repro.value();

        return out;
    }

private:
    WorkspaceScaffold() = delete;

    [[nodiscard]] static Status ensure_directories(const std::filesystem::path& data_root,
                                                   std::string_view workspace_id, Result& out) {
        StatusOr<std::filesystem::path> root =
            WorkspacePaths::workspace_root(data_root, workspace_id);
        if (!root.ok()) {
            return root.status();
        }
        Status root_ok = create_dir_if_missing(root.value(), out, ".");
        if (!root_ok.ok()) {
            return root_ok;
        }

        struct NamedDir {
            const char* rel;
            StatusOr<std::filesystem::path> (*fn)(const std::filesystem::path&, std::string_view);
        };
        const NamedDir named[] = {
            {"pages", &WorkspacePaths::pages_dir},
            {"inputs", &WorkspacePaths::inputs_dir},
            {"hypotheses", &WorkspacePaths::hypotheses_dir},
            {"batches", &WorkspacePaths::batches_dir},
            {"research", &WorkspacePaths::research_dir},
            {"transcripts", &WorkspacePaths::transcripts_dir},
        };
        for (const NamedDir& entry : named) {
            StatusOr<std::filesystem::path> path = entry.fn(data_root, workspace_id);
            if (!path.ok()) {
                return path.status();
            }
            Status ok = create_dir_if_missing(path.value(), out, entry.rel);
            if (!ok.ok()) {
                return ok;
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status create_dir_if_missing(const std::filesystem::path& path,
                                                      Result& out, std::string_view rel) {
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec)) {
            return Status::success();
        }
        if (ec && ec != std::errc::no_such_file_or_directory) {
            return Status::error("WorkspaceScaffold: failed to stat " + path.string() + ": " +
                                 ec.message());
        }
        std::filesystem::create_directories(path, ec);
        if (ec) {
            return Status::error("WorkspaceScaffold: failed to create " + path.string() + ": " +
                                 ec.message());
        }
        out.created_.emplace_back(rel);
        return Status::success();
    }

    [[nodiscard]] static StatusOr<bool> ensure_manifest(const std::filesystem::path& data_root,
                                                        std::string_view workspace_id,
                                                        const Options& options, Result& out) {
        StatusOr<WorkspaceManifest> existing = WorkspaceManifest::load(data_root, workspace_id);
        if (existing.ok()) {
            return false;
        }

        StatusOr<std::string> id = WorkspacePaths::validate_id(workspace_id);
        if (!id.ok()) {
            return id.status();
        }
        StatusOr<WorkspaceManifest> created =
            WorkspaceManifest::make(id.value(), options.created_utc, options.title,
                                    options.default_score_id);
        if (!created.ok()) {
            return created.status();
        }
        nlohmann::json json = created.value().to_json();
        if (!options.notes.empty()) {
            json["notes"] = options.notes;
        }
        json["input"] = nlohmann::json{
            {"kind", "workspace_file"},
            {"fixture_id", nullptr},
            {"path", "inputs/ciphertext.txt"},
        };
        StatusOr<WorkspaceManifest> research_manifest = WorkspaceManifest::from_json(json);
        if (!research_manifest.ok()) {
            return research_manifest.status();
        }
        Status stored = research_manifest.value().store(data_root);
        if (!stored.ok()) {
            return stored;
        }
        out.created_manifest_ = true;
        out.created_.emplace_back("workspace.json");
        return true;
    }

    [[nodiscard]] static StatusOr<bool> write_if_missing(const std::filesystem::path& path,
                                                         std::string_view contents, Result& out,
                                                         std::string_view rel) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(path, ec)) {
            return false;
        }
        if (ec && ec != std::errc::no_such_file_or_directory) {
            return Status::error("WorkspaceScaffold: failed to stat " + path.string() + ": " +
                                 ec.message());
        }
        {
            const std::filesystem::path parent = path.parent_path();
            if (!parent.empty() && !std::filesystem::is_directory(parent, ec)) {
                std::filesystem::create_directories(parent, ec);
                if (ec) {
                    return Status::error("WorkspaceScaffold: failed to create parent for " +
                                         path.string() + ": " + ec.message());
                }
            }
        }
        std::ofstream out_file(path, std::ios::binary | std::ios::trunc);
        if (!out_file) {
            return Status::error("WorkspaceScaffold: failed to write " + path.string());
        }
        out_file << contents;
        if (!out_file) {
            return Status::error("WorkspaceScaffold: failed while writing " + path.string());
        }
        out.created_.emplace_back(rel);
        return true;
    }

    [[nodiscard]] static std::string readme_template(std::string_view workspace_id,
                                                     const Options& options) {
        std::string title =
            options.title.empty() ? std::string(workspace_id) : options.title;
        std::string body;
        body.reserve(256);
        body += "# ";
        body += title;
        body += "\n\n";
        body += "Parcae research workspace (`";
        body += workspace_id;
        body += "`).\n\n";
        body += "- Ciphertext: `inputs/ciphertext.txt` (or `pages/NN.txt`)\n";
        body += "- Hypotheses: `hypotheses/`\n";
        body += "- Batches: `batches/`\n";
        body += "- Research notes: `research/`\n";
        if (!options.created_utc.empty()) {
            body += "\nScaffold created_utc: ";
            body += options.created_utc;
            body += "\n";
        }
        return body;
    }

    [[nodiscard]] static std::string source_template() {
        return "Describe the Liber Primus / community ciphertext source for this workspace.\n"
               "Do not put host-absolute paths here; keep digests and relative refs only.\n";
    }

    [[nodiscard]] static std::string inputs_index_template() {
        return "# Inputs\n\n"
               "Place ciphertext at `ciphertext.txt` (workspace `input.path`).\n"
               "Optional page splits live under `../pages/` as `00.txt`, `01.txt`, …\n";
    }

    [[nodiscard]] static std::string reproduce_template(std::string_view workspace_id) {
        std::string body;
        body += "# Reproduce\n\n";
        body += "Workspace: `";
        body += workspace_id;
        body += "`\n\n";
        body += "Record fixed UTC timestamps, job digests, and CLI invocations here.\n"
                "Avoid host-absolute paths in `run.log` / digests.\n";
        return body;
    }
};

#endif // WORKSPACE_SCAFFOLD_HPP
