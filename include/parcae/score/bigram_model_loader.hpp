#ifndef BIGRAM_MODEL_LOADER_HPP
#define BIGRAM_MODEL_LOADER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/bigram_model_table.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class BigramModelLoader {
public:
    [[nodiscard]] static StatusOr<BigramModelTable> load_from_string(const std::string& json_text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(json_text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid bigram-model JSON: ") + ex.what());
        }
        return load_from_json(root);
    }

    [[nodiscard]] static StatusOr<BigramModelTable> load_from_file(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return Status::error("Failed to open bigram-model file: " + path);
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return load_from_string(buffer.str());
    }

private:
    [[nodiscard]] static StatusOr<BigramModelTable> load_from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("bigram-model root must be an object");
        }
        if (!root.contains("schema") ||
            root.at("schema").get<std::string>() != "parcae.bigram_model.v0") {
            return Status::error("bigram-model schema must be parcae.bigram_model.v0");
        }
        if (!root.contains("id") || !root.at("id").is_string()) {
            return Status::error("bigram-model missing id");
        }
        if (!root.contains("alphabet_size") ||
            root.at("alphabet_size").get<int>() !=
                static_cast<int>(BigramModelTable::alphabet_size)) {
            return Status::error("bigram-model alphabet_size must be 29");
        }
        if (!root.contains("log_probs") || !root.at("log_probs").is_array()) {
            return Status::error("bigram-model missing log_probs array");
        }
        if (root.at("log_probs").size() != BigramModelTable::cell_count) {
            return Status::error("bigram-model log_probs must have length 841");
        }
        if (!root.contains("raw_counts") || !root.at("raw_counts").is_array()) {
            return Status::error("bigram-model missing raw_counts array");
        }
        if (root.at("raw_counts").size() != BigramModelTable::cell_count) {
            return Status::error("bigram-model raw_counts must have length 841");
        }

        std::string log_base = "ln";
        if (root.contains("log_base")) {
            if (!root.at("log_base").is_string()) {
                return Status::error("bigram-model log_base must be a string");
            }
            log_base = root.at("log_base").get<std::string>();
        }
        if (log_base != "ln") {
            return Status::error("bigram-model log_base must be ln");
        }

        std::string smoothing = "add_one";
        if (root.contains("smoothing")) {
            if (!root.at("smoothing").is_string()) {
                return Status::error("bigram-model smoothing must be a string");
            }
            smoothing = root.at("smoothing").get<std::string>();
        }

        std::array<double, BigramModelTable::cell_count> logs{};
        for (std::size_t i = 0; i < BigramModelTable::cell_count; ++i) {
            if (!root.at("log_probs")[i].is_number()) {
                return Status::error("bigram-model log_prob must be numeric");
            }
            const double value = root.at("log_probs")[i].get<double>();
            if (!std::isfinite(value)) {
                return Status::error("bigram-model log_prob must be finite");
            }
            logs[i] = value;
        }

        std::array<std::uint64_t, BigramModelTable::cell_count> counts{};
        for (std::size_t i = 0; i < BigramModelTable::cell_count; ++i) {
            if (!root.at("raw_counts")[i].is_number_unsigned() &&
                !root.at("raw_counts")[i].is_number_integer()) {
                return Status::error("bigram-model raw_count must be an integer");
            }
            const auto raw = root.at("raw_counts")[i].get<std::int64_t>();
            if (raw < 0) {
                return Status::error("bigram-model raw_count must be >= 0");
            }
            counts[i] = static_cast<std::uint64_t>(raw);
        }

        std::vector<std::string> sources;
        if (root.contains("source_fixture_ids") && root.at("source_fixture_ids").is_array()) {
            for (const auto& item : root.at("source_fixture_ids")) {
                if (!item.is_string()) {
                    return Status::error("source_fixture_ids entries must be strings");
                }
                sources.push_back(item.get<std::string>());
            }
        }

        return BigramModelTable(root.at("id").get<std::string>(), logs, counts,
                                std::move(smoothing), std::move(log_base), std::move(sources));
    }
};

#endif // BIGRAM_MODEL_LOADER_HPP
