#ifndef WORKSPACE_CIPHER_HPP
#define WORKSPACE_CIPHER_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/token_stream.hpp"
#include "parcae/hypothesis/workspace_manifest.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/context.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Resolve `parcae.workspace.v0` input → consumable `Index29` ciphertext stream.
///
/// MUST NOT read fixture plaintext (search-loop.md). Fixtures under
/// `data/fixtures/` are read-only.
class WorkspaceCipher {
public:
    enum class SourceKind {
        FixtureCiphertext,
        WorkspaceFile,
        Inline,
    };

    WorkspaceCipher() = default;

    [[nodiscard]] const std::string& workspace_id() const noexcept { return workspace_id_; }

    [[nodiscard]] SourceKind source_kind() const noexcept { return source_kind_; }

    [[nodiscard]] const std::vector<Index29>& indices() const noexcept { return indices_; }

    [[nodiscard]] std::size_t size() const noexcept { return indices_.size(); }

    /// Fixture id when `source_kind == FixtureCiphertext`; otherwise empty.
    [[nodiscard]] const std::string& fixture_id() const noexcept { return fixture_id_; }

    /// Workspace-relative path when `source_kind == WorkspaceFile`; otherwise empty.
    [[nodiscard]] const std::string& source_path() const noexcept { return source_path_; }

    /// SHA-256 hex of newline-normalized UTF-8 ciphertext bytes before tokenize.
    [[nodiscard]] const std::string& ciphertext_sha256() const noexcept {
        return ciphertext_sha256_;
    }

    [[nodiscard]] static constexpr std::string_view to_string(SourceKind kind) noexcept {
        switch (kind) {
        case SourceKind::FixtureCiphertext:
            return "fixture_ciphertext";
        case SourceKind::WorkspaceFile:
            return "workspace_file";
        case SourceKind::Inline:
            return "inline";
        }
        return "unknown";
    }

    /// Load `workspace.json` and resolve ciphertext to consumable Index29 runes.
    [[nodiscard]] static StatusOr<WorkspaceCipher> load(const std::filesystem::path& data_root,
                                                        std::string_view workspace_id) {
        StatusOr<WorkspaceManifest> manifest = WorkspaceManifest::load(data_root, workspace_id);
        if (!manifest.ok()) {
            return manifest.status();
        }
        return from_manifest(data_root, manifest.value());
    }

    [[nodiscard]] static StatusOr<WorkspaceCipher>
    from_manifest(const std::filesystem::path& data_root, const WorkspaceManifest& manifest) {
        const nlohmann::json& input = manifest.input();
        if (!input.contains("kind") || !input.at("kind").is_string()) {
            return Status::error("WorkspaceCipher: input.kind must be a string");
        }
        const std::string kind = input.at("kind").get<std::string>();
        if (kind == "fixture_ciphertext") {
            if (!input.contains("fixture_id") || !input.at("fixture_id").is_string()) {
                return Status::error(
                    "WorkspaceCipher: fixture_ciphertext requires input.fixture_id");
            }
            return from_fixture(data_root, manifest.id(),
                                input.at("fixture_id").get<std::string>());
        }
        if (kind == "workspace_file") {
            if (!input.contains("path") || !input.at("path").is_string()) {
                return Status::error("WorkspaceCipher: workspace_file requires input.path");
            }
            return from_workspace_file(data_root, manifest.id(),
                                       input.at("path").get<std::string>());
        }
        if (kind == "inline_pending") {
            return Status::error("WorkspaceCipher: inline_pending requires explicit indices "
                                 "(use WorkspaceCipher::from_indices)");
        }
        return Status::error("WorkspaceCipher: unknown input.kind: " + kind);
    }

    /// CLI / agent override when the workspace is `inline_pending` (or tests).
    [[nodiscard]] static StatusOr<WorkspaceCipher> from_indices(std::string_view workspace_id,
                                                                std::vector<Index29> indices) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        if (indices.empty()) {
            return Status::error("WorkspaceCipher: indices must be non-empty");
        }
        WorkspaceCipher out;
        out.workspace_id_ = std::move(wid.value());
        out.source_kind_ = SourceKind::Inline;
        out.indices_ = std::move(indices);
        out.ciphertext_sha256_ = Sha256::hex_digest(indices_digest_bytes(out.indices_));
        return out;
    }

    [[nodiscard]] static StatusOr<WorkspaceCipher>
    from_fixture(const std::filesystem::path& data_root, std::string_view workspace_id,
                 std::string_view fixture_id) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        StatusOr<std::string> fid = validate_fixture_id(fixture_id);
        if (!fid.ok()) {
            return fid.status();
        }
        StatusOr<std::filesystem::path> fixture_dir = resolve_fixture_dir(data_root, fid.value());
        if (!fixture_dir.ok()) {
            return fixture_dir.status();
        }
        StatusOr<std::string> text = read_fixture_ciphertext_only(fixture_dir.value());
        if (!text.ok()) {
            return text.status();
        }
        StatusOr<WorkspaceCipher> cipher =
            from_utf8_runes(data_root, wid.value(), text.value(), SourceKind::FixtureCiphertext);
        if (!cipher.ok()) {
            return cipher.status();
        }
        cipher.value().fixture_id_ = std::move(fid.value());
        return cipher;
    }

    [[nodiscard]] static StatusOr<WorkspaceCipher>
    from_workspace_file(const std::filesystem::path& data_root, std::string_view workspace_id,
                        std::string_view relative_path) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        StatusOr<std::filesystem::path> root =
            WorkspacePaths::workspace_root(data_root, wid.value());
        if (!root.ok()) {
            return root.status();
        }
        StatusOr<std::filesystem::path> resolved =
            WorkspacePaths::resolve_under(root.value(), std::filesystem::path(relative_path));
        if (!resolved.ok()) {
            return resolved.status();
        }
        if (!std::filesystem::is_regular_file(resolved.value())) {
            return Status::error("WorkspaceCipher: workspace_file missing: " +
                                 resolved.value().string());
        }
        StatusOr<std::string> text = read_text_file(resolved.value());
        if (!text.ok()) {
            return text.status();
        }
        StatusOr<WorkspaceCipher> cipher =
            from_utf8_runes(data_root, wid.value(), text.value(), SourceKind::WorkspaceFile);
        if (!cipher.ok()) {
            return cipher.status();
        }
        cipher.value().source_path_ = std::string(relative_path);
        return cipher;
    }

private:
    [[nodiscard]] static StatusOr<WorkspaceCipher>
    from_utf8_runes(const std::filesystem::path& data_root, std::string workspace_id,
                    std::string utf8_text, SourceKind kind) {
        const std::string normalized = normalize_newlines(std::move(utf8_text));
        if (normalized.empty()) {
            return Status::error("WorkspaceCipher: ciphertext text is empty");
        }
        const Context ctx(data_root);
        StatusOr<TokenStream> stream = ToolApi::tokenize(ctx, normalized);
        if (!stream.ok()) {
            return stream.status();
        }
        if (stream.value().consumable_count() == 0) {
            return Status::error("WorkspaceCipher: no consumable runes in ciphertext");
        }
        WorkspaceCipher out;
        out.workspace_id_ = std::move(workspace_id);
        out.source_kind_ = kind;
        out.ciphertext_sha256_ = Sha256::hex_digest(normalized);
        out.indices_ = stream.value().consumable_indices();
        return out;
    }

    [[nodiscard]] static StatusOr<std::string> validate_fixture_id(std::string_view fixture_id) {
        if (fixture_id.empty() || fixture_id.size() > 128) {
            return Status::error("WorkspaceCipher: fixture_id length must be 1..128");
        }
        if (fixture_id.find('/') != std::string_view::npos ||
            fixture_id.find('\\') != std::string_view::npos ||
            fixture_id.find("..") != std::string_view::npos) {
            return Status::error("WorkspaceCipher: fixture_id must not contain path separators");
        }
        const unsigned char first = static_cast<unsigned char>(fixture_id[0]);
        if (!((first >= 'a' && first <= 'z') || (first >= '0' && first <= '9'))) {
            return Status::error(
                "WorkspaceCipher: fixture_id must start with a lowercase letter or digit");
        }
        for (char ch : fixture_id) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
                continue;
            }
            return Status::error("WorkspaceCipher: fixture_id has illegal character");
        }
        return std::string(fixture_id);
    }

    [[nodiscard]] static StatusOr<std::filesystem::path>
    resolve_fixture_dir(const std::filesystem::path& data_root, std::string_view fixture_id) {
        const std::filesystem::path solved =
            data_root / "fixtures" / "solved" / std::string(fixture_id);
        if (std::filesystem::is_directory(solved)) {
            return solved;
        }
        const std::filesystem::path draft =
            data_root / "fixtures" / "draft" / std::string(fixture_id);
        if (std::filesystem::is_directory(draft)) {
            return draft;
        }
        return Status::error("WorkspaceCipher: fixture not found under fixtures/solved|draft: " +
                             std::string(fixture_id));
    }

    /// Read only `files.ciphertext` from the fixture manifest — never plaintext.
    [[nodiscard]] static StatusOr<std::string>
    read_fixture_ciphertext_only(const std::filesystem::path& fixture_dir) {
        const std::filesystem::path manifest_path = fixture_dir / "manifest.json";
        std::ifstream in(manifest_path, std::ios::binary);
        if (!in) {
            return Status::error("WorkspaceCipher: failed to open fixture manifest: " +
                                 manifest_path.string());
        }
        nlohmann::json root;
        try {
            in >> root;
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("WorkspaceCipher: invalid fixture manifest: ") +
                                 ex.what());
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != "parcae.fixture_manifest.v0") {
            return Status::error(
                "WorkspaceCipher: fixture schema must be parcae.fixture_manifest.v0");
        }
        if (!root.contains("files") || !root.at("files").is_object() ||
            !root.at("files").contains("ciphertext") ||
            !root.at("files").at("ciphertext").is_string()) {
            return Status::error("WorkspaceCipher: fixture files.ciphertext is required");
        }
        const std::string rel = root.at("files").at("ciphertext").get<std::string>();
        if (rel.empty() || rel.find("..") != std::string::npos ||
            std::filesystem::path(rel).is_absolute()) {
            return Status::error("WorkspaceCipher: fixture ciphertext path is unsafe");
        }
        StatusOr<std::filesystem::path> resolved =
            WorkspacePaths::resolve_under(fixture_dir, std::filesystem::path(rel));
        if (!resolved.ok()) {
            return resolved.status();
        }
        return read_text_file(resolved.value());
    }

    [[nodiscard]] static StatusOr<std::string> read_text_file(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("WorkspaceCipher: failed to open file: " + path.string());
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        return buf.str();
    }

    [[nodiscard]] static std::string normalize_newlines(std::string text) {
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
    }

    [[nodiscard]] static std::string indices_digest_bytes(const std::vector<Index29>& indices) {
        std::string bytes;
        bytes.resize(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            bytes[i] = static_cast<char>(indices[i].value());
        }
        return bytes;
    }

    std::string workspace_id_;
    SourceKind source_kind_ = SourceKind::Inline;
    std::vector<Index29> indices_;
    std::string fixture_id_;
    std::string source_path_;
    std::string ciphertext_sha256_;
};

#endif // WORKSPACE_CIPHER_HPP
