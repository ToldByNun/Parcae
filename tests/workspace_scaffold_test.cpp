#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/hypothesis/workspace_scaffold.hpp>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] std::filesystem::path make_sandbox(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    std::filesystem::create_directories(root / "fixtures" / "solved" / "a-warning", ec);
    {
        std::ofstream out(root / "fixtures" / "solved" / "a-warning" / "ciphertext.txt");
        out << "placeholder\n";
    }
    return root;
}

[[nodiscard]] bool is_dir(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::is_directory(p, ec);
}

[[nodiscard]] bool is_file(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::is_regular_file(p, ec);
}

[[nodiscard]] std::string read_text(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    REQUIRE(in);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("WorkspaceScaffold::ensure_research builds canonical tree",
          "[hypothesis][scaffold][workspace]") {
    const auto data = make_sandbox("parcae_workspace_scaffold_research");

    WorkspaceScaffold::Options opts;
    opts.created_utc = "2026-09-27T00:00:00Z";
    opts.title = "LP2 explore";
    opts.notes = "scaffold test";

    StatusOr<WorkspaceScaffold::Result> first =
        WorkspaceScaffold::ensure_research(data, "lp2-scaffold", opts);
    if (!first.ok()) {
        FAIL(first.status().message());
    }
    REQUIRE(first.value().workspace_id() == "lp2-scaffold");
    REQUIRE(first.value().any_created());
    REQUIRE(first.value().created_manifest());
    REQUIRE(first.value().created_readme());
    REQUIRE(first.value().created_source());
    REQUIRE(first.value().created_inputs_index());
    REQUIRE(first.value().created_reproduce());

    const auto ws = data / "workspaces" / "lp2-scaffold";
    REQUIRE(is_dir(ws));
    REQUIRE(is_dir(ws / "pages"));
    REQUIRE(is_dir(ws / "inputs"));
    REQUIRE(is_dir(ws / "hypotheses"));
    REQUIRE(is_dir(ws / "batches"));
    REQUIRE(is_dir(ws / "research"));
    REQUIRE(is_dir(ws / "transcripts"));
    REQUIRE(is_file(ws / "workspace.json"));
    REQUIRE(is_file(ws / "README.md"));
    REQUIRE(is_file(ws / "SOURCE.txt"));
    REQUIRE(is_file(ws / "inputs" / "INDEX.md"));
    REQUIRE(is_file(ws / "research" / "REPRODUCE.md"));

    StatusOr<WorkspaceManifest> manifest = WorkspaceManifest::load(data, "lp2-scaffold");
    REQUIRE(manifest.ok());
    REQUIRE(manifest.value().id() == "lp2-scaffold");
    REQUIRE(manifest.value().input().at("kind").get<std::string>() == "workspace_file");
    REQUIRE(manifest.value().input().at("path").get<std::string>() == "inputs/ciphertext.txt");

    const std::string readme = read_text(ws / "README.md");
    REQUIRE(readme.find("LP2 explore") != std::string::npos);
    REQUIRE(readme.find("2026-09-27T00:00:00Z") != std::string::npos);
    REQUIRE(readme.find(data.string()) == std::string::npos);

    // Idempotent: second ensure creates nothing and does not clobber templates.
    {
        std::ofstream mark(ws / "README.md", std::ios::binary | std::ios::trunc);
        mark << "custom readme keep me\n";
    }
    StatusOr<WorkspaceScaffold::Result> second =
        WorkspaceScaffold::ensure_research(data, "lp2-scaffold", opts);
    REQUIRE(second.ok());
    REQUIRE_FALSE(second.value().any_created());
    REQUIRE_FALSE(second.value().created_manifest());
    REQUIRE_FALSE(second.value().created_readme());
    REQUIRE(read_text(ws / "README.md") == "custom readme keep me\n");

    std::error_code ec;
    std::filesystem::remove_all(data, ec);
}

TEST_CASE("WorkspaceScaffold::ensure_research rejects bad id and missing utc",
          "[hypothesis][scaffold][workspace]") {
    const auto data = make_sandbox("parcae_workspace_scaffold_errors");

    WorkspaceScaffold::Options opts;
    opts.created_utc = "2026-09-27T00:00:00Z";
    REQUIRE_FALSE(WorkspaceScaffold::ensure_research(data, "BadId", opts).ok());
    REQUIRE_FALSE(WorkspaceScaffold::ensure_research(data, "../evil", opts).ok());

    WorkspaceScaffold::Options no_utc;
    REQUIRE_FALSE(WorkspaceScaffold::ensure_research(data, "needs-utc", no_utc).ok());

    // After a successful create, missing utc is OK (manifest already present).
    REQUIRE(WorkspaceScaffold::ensure_research(data, "needs-utc", opts).ok());
    StatusOr<WorkspaceScaffold::Result> again =
        WorkspaceScaffold::ensure_research(data, "needs-utc", no_utc);
    REQUIRE(again.ok());
    REQUIRE_FALSE(again.value().any_created());

    std::error_code ec;
    std::filesystem::remove_all(data, ec);
}

TEST_CASE("WorkspaceScaffold path helpers match tree", "[hypothesis][scaffold][paths]") {
    const auto data = make_sandbox("parcae_workspace_scaffold_paths");
    WorkspaceScaffold::Options opts;
    opts.created_utc = "2026-09-27T01:00:00Z";
    REQUIRE(WorkspaceScaffold::ensure_research(data, "path-check", opts).ok());

    REQUIRE(WorkspacePaths::pages_dir(data, "path-check").value() ==
            data / "workspaces" / "path-check" / "pages");
    REQUIRE(WorkspacePaths::research_dir(data, "path-check").value() ==
            data / "workspaces" / "path-check" / "research");
    REQUIRE(WorkspacePaths::readme_path(data, "path-check").value() ==
            data / "workspaces" / "path-check" / "README.md");
    REQUIRE(WorkspacePaths::source_path(data, "path-check").value() ==
            data / "workspaces" / "path-check" / "SOURCE.txt");

    std::error_code ec;
    std::filesystem::remove_all(data, ec);
}
