#ifndef GEMATRIA_PROFILE_LOADER_HPP
#define GEMATRIA_PROFILE_LOADER_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/gematria/gematria_entry.hpp"
#include "parcae/gematria/gematria_profile.hpp"

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class GematriaProfileLoader {
public:
    [[nodiscard]] static StatusOr<GematriaProfile> load_from_string(const std::string& json_text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(json_text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid Gematria profile JSON: ") + ex.what());
        }

        return load_from_json(root);
    }

    [[nodiscard]] static StatusOr<GematriaProfile> load_from_file(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return Status::error("Failed to open Gematria profile file: " + path);
        }

        std::ostringstream buffer;
        buffer << input.rdbuf();
        return load_from_string(buffer.str());
    }

private:
    GematriaProfileLoader() = delete;

    [[nodiscard]] static StatusOr<GematriaProfile> load_from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("Gematria profile root must be a JSON object");
        }

        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != "parcae.gematria_profile.v0") {
            return Status::error("Gematria profile schema must be parcae.gematria_profile.v0");
        }

        if (!root.contains("modulus") || !root.at("modulus").is_number_integer() ||
            root.at("modulus").get<int>() != static_cast<int>(Index29::modulus)) {
            return Status::error("Gematria profile modulus must be 29");
        }

        if (!root.contains("profile_id") || !root.at("profile_id").is_string()) {
            return Status::error("Gematria profile_id must be a string");
        }
        const std::string profile_id = root.at("profile_id").get<std::string>();
        if (profile_id.empty()) {
            return Status::error("Gematria profile_id must be non-empty");
        }

        if (!root.contains("entries") || !root.at("entries").is_array()) {
            return Status::error("Gematria profile entries must be an array");
        }

        std::vector<GematriaEntry> entries;
        entries.reserve(root.at("entries").size());

        for (const nlohmann::json& item : root.at("entries")) {
            StatusOr<GematriaEntry> entry = parse_entry(item);
            if (!entry.ok()) {
                return entry.status();
            }
            entries.push_back(std::move(entry.value()));
        }

        return GematriaProfile::create(profile_id, std::move(entries));
    }

    [[nodiscard]] static StatusOr<GematriaEntry> parse_entry(const nlohmann::json& item) {
        if (!item.is_object()) {
            return Status::error("Gematria profile entry must be an object");
        }

        if (!item.contains("index") || !item.at("index").is_number_integer()) {
            return Status::error("Gematria entry index must be an integer");
        }
        const int index_value = item.at("index").get<int>();
        if (index_value < 0 || index_value >= static_cast<int>(Index29::modulus)) {
            return Status::error("Gematria entry index out of range 0..28");
        }

        if (!item.contains("prime") || !item.at("prime").is_number_integer()) {
            return Status::error("Gematria entry prime must be an integer");
        }
        const auto prime = item.at("prime").get<std::uint32_t>();

        if (!item.contains("rune") || !item.at("rune").is_string()) {
            return Status::error("Gematria entry rune must be a string");
        }
        if (!item.contains("preferred") || !item.at("preferred").is_string()) {
            return Status::error("Gematria entry preferred must be a string");
        }
        if (!item.contains("labels") || !item.at("labels").is_array()) {
            return Status::error("Gematria entry labels must be an array");
        }

        std::vector<std::string> labels;
        for (const nlohmann::json& label : item.at("labels")) {
            if (!label.is_string()) {
                return Status::error("Gematria entry labels must be strings");
            }
            labels.push_back(label.get<std::string>());
        }

        return GematriaEntry{
            Index29{static_cast<std::uint8_t>(index_value)},
            prime,
            item.at("rune").get<std::string>(),
            item.at("preferred").get<std::string>(),
            std::move(labels),
        };
    }
};

#endif // GEMATRIA_PROFILE_LOADER_HPP
