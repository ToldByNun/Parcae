#include <parcae/core/index29.hpp>
#include <parcae/core/sha256.hpp>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/workspace_cipher.hpp>
#include <parcae/tool/api.hpp>
#include <parcae/tool/context.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

}  // namespace

TEST_CASE("WorkspaceCipher loads _example fixture_ciphertext", "[search][cipher]") {
    StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::load(data_root(), "_example");
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value().workspace_id() == "_example");
    REQUIRE(
        cipher.value().source_kind() == WorkspaceCipher::SourceKind::FixtureCiphertext);
    REQUIRE(cipher.value().fixture_id() == "a-warning");
    REQUIRE(cipher.value().size() > 0);
    REQUIRE(cipher.value().ciphertext_sha256().size() == 64);

    // Matches direct tokenize of fixture ciphertext (no plaintext touched).
    const auto cipher_path =
        data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt";
    std::ifstream in(cipher_path, std::ios::binary);
    REQUIRE(in);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const parcae::tool::Context ctx(data_root());
    StatusOr<TokenStream> stream = parcae::tool::tokenize(ctx, text);
    REQUIRE(stream.ok());
    REQUIRE(cipher.value().indices() == stream.value().consumable_indices());
    REQUIRE(
        cipher.value().ciphertext_sha256() ==
        Sha256::hex_digest(
            [&]() {
                // Same LF normalization WorkspaceCipher applies.
                std::string out;
                out.reserve(text.size());
                for (std::size_t i = 0; i < text.size(); ++i) {
                    if (text[i] == '\r') {
                        if (i + 1 < text.size() && text[i + 1] == '\n') {
                            ++i;
                        }
                        out.push_back('\n');
                    } else {
                        out.push_back(text[i]);
                    }
                }
                return out;
            }()));
}

TEST_CASE("WorkspaceCipher from_indices for inline override", "[search][cipher]") {
    StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::from_indices(
        "inline-ws", {Index29{1}, Index29{2}, Index29{3}});
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value().source_kind() == WorkspaceCipher::SourceKind::Inline);
    REQUIRE(cipher.value().size() == 3);
    REQUIRE(cipher.value().indices()[1].value() == 2);
    REQUIRE_FALSE(WorkspaceCipher::from_indices("inline-ws", {}).ok());
}

TEST_CASE("WorkspaceCipher rejects inline_pending without indices", "[search][cipher]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_workspace_cipher_b9_inline";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    StatusOr<WorkspaceManifest> manifest =
        WorkspaceManifest::make("pending-ws", "2026-09-21T20:00:00Z");
    REQUIRE(manifest.ok());
    REQUIRE(manifest.value().store(tmp).ok());
    REQUIRE_FALSE(WorkspaceCipher::load(tmp, "pending-ws").ok());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("WorkspaceCipher loads workspace_file under workspace root", "[search][cipher]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_workspace_cipher_b9_file";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    // Copy profiles needed for tokenize.
    std::filesystem::create_directories(tmp / "profiles" / "gematria", ec);
    std::filesystem::create_directories(tmp / "profiles" / "separators", ec);
    std::filesystem::copy_file(
        data_root() / "profiles" / "gematria" / "gematria-primus-v0.json",
        tmp / "profiles" / "gematria" / "gematria-primus-v0.json",
        ec);
    std::filesystem::copy_file(
        data_root() / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
        tmp / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
        ec);

    StatusOr<WorkspaceManifest> manifest =
        WorkspaceManifest::make("file-ws", "2026-09-21T20:00:00Z");
    REQUIRE(manifest.ok());
    // Rewrite input to workspace_file pointing at copied a-warning ciphertext.
    nlohmann::json j = manifest.value().to_json();
    j["input"] = nlohmann::json{
        {"kind", "workspace_file"},
        {"fixture_id", nullptr},
        {"path", "inputs/ciphertext.txt"},
    };
    StatusOr<WorkspaceManifest> with_file = WorkspaceManifest::from_json(j);
    REQUIRE(with_file.ok());
    REQUIRE(with_file.value().store(tmp).ok());

    StatusOr<std::filesystem::path> root = WorkspacePaths::workspace_root(tmp, "file-ws");
    REQUIRE(root.ok());
    std::filesystem::create_directories(root.value() / "inputs", ec);
    std::filesystem::copy_file(
        data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt",
        root.value() / "inputs" / "ciphertext.txt",
        ec);
    REQUIRE(!ec);

    StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::load(tmp, "file-ws");
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value().source_kind() == WorkspaceCipher::SourceKind::WorkspaceFile);
    REQUIRE(cipher.value().source_path() == "inputs/ciphertext.txt");
    REQUIRE(cipher.value().size() > 0);

    StatusOr<WorkspaceCipher> via_fixture =
        WorkspaceCipher::from_fixture(data_root(), "_example", "a-warning");
    REQUIRE(via_fixture.ok());
    REQUIRE(cipher.value().indices() == via_fixture.value().indices());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("WorkspaceCipher rejects path escape and bad fixture ids", "[search][cipher]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_workspace_cipher_b9_escape";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    StatusOr<WorkspaceManifest> manifest =
        WorkspaceManifest::make("esc-ws", "2026-09-21T20:00:00Z");
    REQUIRE(manifest.ok());
    nlohmann::json j = manifest.value().to_json();
    j["input"] = nlohmann::json{
        {"kind", "workspace_file"},
        {"fixture_id", nullptr},
        {"path", "../fixtures/solved/a-warning/ciphertext.txt"},
    };
    StatusOr<WorkspaceManifest> escaped = WorkspaceManifest::from_json(j);
    REQUIRE(escaped.ok());
    REQUIRE(escaped.value().store(tmp).ok());
    REQUIRE_FALSE(WorkspaceCipher::load(tmp, "esc-ws").ok());

    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "../a-warning").ok());
    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "a-warning/../welcome").ok());
    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "no-such-fixture-zzz").ok());

    std::filesystem::remove_all(tmp, ec);
}
