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
#include <string_view>

#include <nlohmann/json.hpp>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

void copy_tokenize_profiles(const std::filesystem::path& tmp) {
    std::error_code ec;
    std::filesystem::create_directories(tmp / "profiles" / "gematria", ec);
    std::filesystem::create_directories(tmp / "profiles" / "separators", ec);
    std::filesystem::copy_file(
        data_root() / "profiles" / "gematria" / "gematria-primus-v0.json",
        tmp / "profiles" / "gematria" / "gematria-primus-v0.json",
        std::filesystem::copy_options::overwrite_existing,
        ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(
        data_root() / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
        tmp / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
        std::filesystem::copy_options::overwrite_existing,
        ec);
    REQUIRE(!ec);
}

[[nodiscard]] StatusOr<WorkspaceManifest> store_workspace_with_input(
    const std::filesystem::path& data_root_path,
    std::string_view workspace_id,
    const nlohmann::json& input) {
    StatusOr<WorkspaceManifest> base =
        WorkspaceManifest::make(workspace_id, "2026-09-21T22:00:00Z");
    if (!base.ok()) {
        return base.status();
    }
    nlohmann::json j = base.value().to_json();
    j["input"] = input;
    StatusOr<WorkspaceManifest> m = WorkspaceManifest::from_json(j);
    if (!m.ok()) {
        return m.status();
    }
    Status stored = m.value().store(data_root_path);
    if (!stored.ok()) {
        return stored;
    }
    return m;
}

}  // namespace

TEST_CASE(
    "C10 fixture resolve: a-warning and welcome match direct tokenize",
    "[search][cipher][resolve]") {
    for (const char* fixture_id : {"a-warning", "welcome"}) {
        StatusOr<WorkspaceCipher> cipher =
            WorkspaceCipher::from_fixture(data_root(), "_example", fixture_id);
        REQUIRE(cipher.ok());
        REQUIRE(cipher.value().fixture_id() == fixture_id);
        REQUIRE(
            cipher.value().source_kind() == WorkspaceCipher::SourceKind::FixtureCiphertext);
        REQUIRE(cipher.value().size() > 0);

        const auto path =
            data_root() / "fixtures" / "solved" / fixture_id / "ciphertext.txt";
        std::ifstream in(path, std::ios::binary);
        REQUIRE(in);
        const std::string text(
            (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const parcae::tool::Context ctx(data_root());
        StatusOr<TokenStream> stream = parcae::tool::tokenize(ctx, text);
        REQUIRE(stream.ok());
        REQUIRE(cipher.value().indices() == stream.value().consumable_indices());
    }
}

TEST_CASE(
    "C10 fixture resolve: ciphertext-only fixture (no plaintext file) succeeds",
    "[search][cipher][resolve]") {
    // Proves WorkspaceCipher does not go through FixtureLoader (which requires plaintext).
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_cipher_c10_cipher_only";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    copy_tokenize_profiles(tmp);

    const auto fixture_dir = tmp / "fixtures" / "solved" / "cipher-only-fx";
    std::filesystem::create_directories(fixture_dir, ec);
    {
        std::ifstream src(
            data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt",
            std::ios::binary);
        REQUIRE(src);
        std::ofstream dst(fixture_dir / "ciphertext.txt", std::ios::binary);
        REQUIRE(dst);
        dst << src.rdbuf();
    }
    {
        nlohmann::json manifest{
            {"schema", "parcae.fixture_manifest.v0"},
            {"id", "cipher-only-fx"},
            {"files", {{"ciphertext", "ciphertext.txt"}, {"plaintext", "missing-plaintext.txt"}}},
            {"method", {{"transform_id", "identity"}, {"direction", "decrypt"}, {"params", nlohmann::json::object()}}},
            {"non_rune_literal_regions", nlohmann::json::array()},
            {"hashes",
             {{"ciphertext_sha256", nullptr},
              {"plaintext_sha256", nullptr},
              {"normalized_plaintext_sha256", nullptr}}},
            {"verification", {{"status", "draft"}, {"recomputed_ok", false}}},
        };
        std::ofstream out(fixture_dir / "manifest.json");
        REQUIRE(out);
        out << manifest.dump(2);
    }
    // Deliberately do NOT create missing-plaintext.txt.

    StatusOr<WorkspaceCipher> cipher =
        WorkspaceCipher::from_fixture(tmp, "ws-cipher-only", "cipher-only-fx");
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value().size() > 0);
    REQUIRE(cipher.value().fixture_id() == "cipher-only-fx");

    StatusOr<WorkspaceCipher> real =
        WorkspaceCipher::from_fixture(data_root(), "_example", "a-warning");
    REQUIRE(real.ok());
    REQUIRE(cipher.value().indices() == real.value().indices());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE(
    "C10 workspace_file resolve: nested path + from_workspace_file API",
    "[search][cipher][resolve]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_cipher_c10_ws_file";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);
    copy_tokenize_profiles(tmp);

    REQUIRE(store_workspace_with_input(
                tmp,
                "resolve-ws",
                nlohmann::json{
                    {"kind", "workspace_file"},
                    {"fixture_id", nullptr},
                    {"path", "inputs/nested/page.txt"},
                })
                .ok());

    StatusOr<std::filesystem::path> root = WorkspacePaths::workspace_root(tmp, "resolve-ws");
    REQUIRE(root.ok());
    std::filesystem::create_directories(root.value() / "inputs" / "nested", ec);
    std::filesystem::copy_file(
        data_root() / "fixtures" / "solved" / "welcome" / "ciphertext.txt",
        root.value() / "inputs" / "nested" / "page.txt",
        ec);
    REQUIRE(!ec);

    StatusOr<WorkspaceCipher> via_load = WorkspaceCipher::load(tmp, "resolve-ws");
    REQUIRE(via_load.ok());
    REQUIRE(via_load.value().source_path() == "inputs/nested/page.txt");

    StatusOr<WorkspaceCipher> via_api =
        WorkspaceCipher::from_workspace_file(tmp, "resolve-ws", "inputs/nested/page.txt");
    REQUIRE(via_api.ok());
    REQUIRE(via_api.value().indices() == via_load.value().indices());
    REQUIRE(via_api.value().ciphertext_sha256() == via_load.value().ciphertext_sha256());

    StatusOr<WorkspaceCipher> welcome =
        WorkspaceCipher::from_fixture(data_root(), "_example", "welcome");
    REQUIRE(welcome.ok());
    REQUIRE(via_load.value().indices() == welcome.value().indices());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE(
    "C10 workspace_file reject: missing file and empty ciphertext",
    "[search][cipher][resolve]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_cipher_c10_missing";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);
    copy_tokenize_profiles(tmp);

    REQUIRE(store_workspace_with_input(
                tmp,
                "missing-ws",
                nlohmann::json{
                    {"kind", "workspace_file"},
                    {"fixture_id", nullptr},
                    {"path", "inputs/nope.txt"},
                })
                .ok());
    REQUIRE_FALSE(WorkspaceCipher::load(tmp, "missing-ws").ok());

    StatusOr<std::filesystem::path> root = WorkspacePaths::workspace_root(tmp, "missing-ws");
    REQUIRE(root.ok());
    std::filesystem::create_directories(root.value() / "inputs", ec);
    {
        std::ofstream empty(root.value() / "inputs" / "empty.txt");
        REQUIRE(empty);
    }
    REQUIRE_FALSE(
        WorkspaceCipher::from_workspace_file(tmp, "missing-ws", "inputs/empty.txt").ok());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE(
    "C10 path escape reject: traversal, absolute, and unsafe fixture ids",
    "[search][cipher][resolve]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_cipher_c10_escape";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);
    copy_tokenize_profiles(tmp);

    const char* bad_paths[] = {
        "../fixtures/solved/a-warning/ciphertext.txt",
        "../../etc/passwd",
        "inputs/../../workspaces/other/secret.txt",
        "inputs/../../../fixtures/solved/a-warning/ciphertext.txt",
#ifdef _WIN32
        "C:/Windows/System32/drivers/etc/hosts",
#else
        "/etc/passwd",
#endif
    };

    for (const char* bad : bad_paths) {
        REQUIRE(store_workspace_with_input(
                    tmp,
                    "esc-ws",
                    nlohmann::json{
                        {"kind", "workspace_file"},
                        {"fixture_id", nullptr},
                        {"path", bad},
                    })
                    .ok());
        REQUIRE_FALSE(WorkspaceCipher::load(tmp, "esc-ws").ok());
        REQUIRE_FALSE(WorkspaceCipher::from_workspace_file(tmp, "esc-ws", bad).ok());
    }

    REQUIRE_FALSE(WorkspaceCipher::from_fixture(data_root(), "_example", "").ok());
    REQUIRE_FALSE(WorkspaceCipher::from_fixture(data_root(), "_example", "../a-warning").ok());
    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "a-warning/../welcome").ok());
    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "solved/a-warning").ok());
    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "a-warning\\x").ok());
    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "NoCaps").ok());
    REQUIRE_FALSE(
        WorkspaceCipher::from_fixture(data_root(), "_example", "missing-fixture-zzz").ok());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE(
    "C10 fixture manifest rejects unsafe relative ciphertext path",
    "[search][cipher][resolve]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_cipher_c10_unsafe_rel";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    copy_tokenize_profiles(tmp);

    const auto fixture_dir = tmp / "fixtures" / "solved" / "unsafe-rel";
    std::filesystem::create_directories(fixture_dir, ec);
    {
        nlohmann::json manifest{
            {"schema", "parcae.fixture_manifest.v0"},
            {"id", "unsafe-rel"},
            {"files",
             {{"ciphertext", "../a-warning/ciphertext.txt"},
              {"plaintext", "plaintext.txt"}}},
            {"method", {{"transform_id", "identity"}, {"direction", "decrypt"}}},
            {"non_rune_literal_regions", nlohmann::json::array()},
            {"hashes",
             {{"ciphertext_sha256", nullptr},
              {"plaintext_sha256", nullptr},
              {"normalized_plaintext_sha256", nullptr}}},
            {"verification", {{"status", "draft"}, {"recomputed_ok", false}}},
        };
        std::ofstream out(fixture_dir / "manifest.json");
        REQUIRE(out);
        out << manifest.dump(2);
    }

    REQUIRE_FALSE(WorkspaceCipher::from_fixture(tmp, "ws-unsafe", "unsafe-rel").ok());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE(
    "C10 from_manifest routes kinds; unknown kind fails",
    "[search][cipher][resolve]") {
    StatusOr<WorkspaceManifest> example = WorkspaceManifest::load(data_root(), "_example");
    REQUIRE(example.ok());
    StatusOr<WorkspaceCipher> via_manifest =
        WorkspaceCipher::from_manifest(data_root(), example.value());
    REQUIRE(via_manifest.ok());
    REQUIRE(via_manifest.value().fixture_id() == "a-warning");

    nlohmann::json bad = example.value().to_json();
    bad["input"] = nlohmann::json{
        {"kind", "not_a_real_kind"},
        {"fixture_id", nullptr},
        {"path", nullptr},
    };
    // from_json rejects unknown kind at manifest level — craft via load path by
    // writing a raw workspace.json that skips WorkspaceManifest::validate_input.
    // Prefer testing from_manifest with a mutated object through make + rewrite.
    // WorkspaceManifest::from_json rejects unknown kinds, so exercise the cipher
    // branch by calling from_fixture / from_workspace_file / from_indices only.
    // Explicit unknown-kind coverage: load a hand-written workspace.json.
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_cipher_c10_kind";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    const auto ws = tmp / "workspaces" / "kind-ws";
    std::filesystem::create_directories(ws, ec);
    {
        nlohmann::json root{
            {"schema", "parcae.workspace.v0"},
            {"id", "kind-ws"},
            {"created_utc", "2026-09-21T22:00:00Z"},
            {"updated_utc", "2026-09-21T22:00:00Z"},
            {"title", ""},
            {"notes", ""},
            {"input",
             {{"kind", "inline_pending"}, {"fixture_id", nullptr}, {"path", nullptr}}},
            {"default_score_id", "chi2_english_gp_v0"},
            {"default_score_version", "v0"},
        };
        std::ofstream out(ws / "workspace.json");
        REQUIRE(out);
        out << root.dump(2);
    }
    REQUIRE_FALSE(WorkspaceCipher::load(tmp, "kind-ws").ok());

    // Swap to a kind WorkspaceManifest would reject if loaded via from_json —
    // write raw JSON with kind the cipher switch must still reject if somehow present.
    {
        nlohmann::json root{
            {"schema", "parcae.workspace.v0"},
            {"id", "kind-ws"},
            {"created_utc", "2026-09-21T22:00:00Z"},
            {"updated_utc", "2026-09-21T22:00:00Z"},
            {"title", ""},
            {"notes", ""},
            {"input",
             {{"kind", "network_url"}, {"fixture_id", nullptr}, {"path", nullptr}}},
            {"default_score_id", "chi2_english_gp_v0"},
            {"default_score_version", "v0"},
        };
        std::ofstream out(ws / "workspace.json");
        REQUIRE(out);
        out << root.dump(2);
    }
    // WorkspaceManifest::load validates input.kind and fails before cipher.
    REQUIRE_FALSE(WorkspaceCipher::load(tmp, "kind-ws").ok());

    std::filesystem::remove_all(tmp, ec);
}
