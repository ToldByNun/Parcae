#ifndef FIXTURE_LOADER_HPP
#define FIXTURE_LOADER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/fixture.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class FixtureLoader {
public:
    [[nodiscard]] static StatusOr<Fixture> load_directory(const std::string& fixture_dir) {
        const std::filesystem::path dir(fixture_dir);
        const std::filesystem::path manifest_path = dir / "manifest.json";

        std::ifstream manifest_input(manifest_path, std::ios::binary);
        if (!manifest_input) {
            return Status::error("Failed to open fixture manifest: " + manifest_path.string());
        }

        nlohmann::json root;
        try {
            manifest_input >> root;
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid fixture manifest JSON: ") + ex.what());
        }

        if (!root.contains("schema") || root.at("schema") != "parcae.fixture_manifest.v0") {
            return Status::error("Fixture schema must be parcae.fixture_manifest.v0");
        }
        if (!root.contains("id") || !root.at("id").is_string()) {
            return Status::error("Fixture id must be a string");
        }

        const std::string id = root.at("id").get<std::string>();
        if (id != dir.filename().string()) {
            return Status::error("Fixture id must match directory name");
        }

        if (!root.contains("files") || !root.at("files").contains("ciphertext") ||
            !root.at("files").contains("plaintext")) {
            return Status::error("Fixture files.ciphertext and files.plaintext are required");
        }

        StatusOr<std::string> ciphertext =
            read_text_file(dir / root.at("files").at("ciphertext").get<std::string>());
        if (!ciphertext.ok()) {
            return ciphertext.status();
        }
        StatusOr<std::string> plaintext =
            read_text_file(dir / root.at("files").at("plaintext").get<std::string>());
        if (!plaintext.ok()) {
            return plaintext.status();
        }

        if (!root.contains("method") || !root.at("method").contains("transform_id")) {
            return Status::error("Fixture method.transform_id is required");
        }

        const std::string transform_id = root.at("method").at("transform_id").get<std::string>();
        const std::string direction = root.at("method").value("direction", std::string("decrypt"));

        std::vector<std::size_t> skip_indices;
        if (root.at("method").contains("interrupt") &&
            root.at("method").at("interrupt").contains("skip_indices")) {
            for (const nlohmann::json& item :
                 root.at("method").at("interrupt").at("skip_indices")) {
                skip_indices.push_back(item.get<std::size_t>());
            }
        }

        std::optional<std::string> key_latin;
        std::optional<std::vector<int>> key_indices;
        nlohmann::json params = nlohmann::json::object();
        if (root.at("method").contains("params")) {
            params = root.at("method").at("params");
            if (params.contains("key_latin") && params.at("key_latin").is_string()) {
                key_latin = params.at("key_latin").get<std::string>();
            }
            if (params.contains("key_indices") && params.at("key_indices").is_array()) {
                std::vector<int> indices;
                for (const nlohmann::json& item : params.at("key_indices")) {
                    indices.push_back(item.get<int>());
                }
                key_indices = std::move(indices);
            }
        }

        if (!root.contains("non_rune_literal_regions") ||
            !root.at("non_rune_literal_regions").is_array()) {
            return Status::error("non_rune_literal_regions must be present as an array");
        }

        std::vector<FixtureLiteralRegion> literals;
        for (const nlohmann::json& item : root.at("non_rune_literal_regions")) {
            literals.emplace_back(item.value("kind", std::string{}),
                                  item.value("role", std::string{}),
                                  item.value("value_file", std::string{}),
                                  item.value("compare", std::string{"exact"}));
        }

        if (!root.contains("verification")) {
            return Status::error("verification object is required");
        }
        const std::string verification_status =
            root.at("verification").value("status", std::string{"draft"});
        const bool recomputed_ok = root.at("verification").value("recomputed_ok", false);

        if (!root.contains("hashes") || !root.at("hashes").is_object()) {
            return Status::error("hashes object is required");
        }

        StatusOr<FixtureHashes> hashes = parse_hashes(root.at("hashes"));
        if (!hashes.ok()) {
            return hashes.status();
        }

        Fixture fixture{
            id,
            normalize_newlines(std::move(ciphertext.value())),
            normalize_newlines(std::move(plaintext.value())),
            transform_id,
            direction,
            verification_status,
            recomputed_ok,
            std::move(skip_indices),
            std::move(literals),
            std::move(key_latin),
            std::move(key_indices),
            std::move(hashes.value()),
            std::move(params),
        };

        Status lock_status = fixture.validate_lock_rules();
        if (!lock_status.ok()) {
            return lock_status;
        }
        return fixture;
    }

private:
    FixtureLoader() = delete;

    [[nodiscard]] static StatusOr<std::optional<std::string>>
    parse_optional_hash(const nlohmann::json& hashes, const char* key) {
        if (!hashes.contains(key) || hashes.at(key).is_null()) {
            return std::optional<std::string>{};
        }
        if (!hashes.at(key).is_string()) {
            return Status::error(std::string(key) + " must be a string or null");
        }
        std::string value = hashes.at(key).get<std::string>();
        if (value.size() != 64) {
            return Status::error(std::string(key) + " must be a 64-char lowercase hex SHA-256");
        }
        for (char ch : value) {
            const bool ok = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
            if (!ok) {
                return Status::error(std::string(key) + " must be lowercase hex");
            }
        }
        return std::optional<std::string>{std::move(value)};
    }

    [[nodiscard]] static StatusOr<FixtureHashes> parse_hashes(const nlohmann::json& hashes) {
        StatusOr<std::optional<std::string>> ciphertext =
            parse_optional_hash(hashes, "ciphertext_sha256");
        if (!ciphertext.ok()) {
            return ciphertext.status();
        }
        StatusOr<std::optional<std::string>> plaintext =
            parse_optional_hash(hashes, "plaintext_sha256");
        if (!plaintext.ok()) {
            return plaintext.status();
        }
        StatusOr<std::optional<std::string>> normalized =
            parse_optional_hash(hashes, "normalized_plaintext_sha256");
        if (!normalized.ok()) {
            return normalized.status();
        }
        return FixtureHashes{
            std::move(ciphertext.value()),
            std::move(plaintext.value()),
            std::move(normalized.value()),
        };
    }

    [[nodiscard]] static StatusOr<std::string> read_text_file(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return Status::error("Failed to open fixture text file: " + path.string());
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
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
};

#endif // FIXTURE_LOADER_HPP
