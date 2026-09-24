#ifndef EXPECTED_FREQUENCY_LOADER_HPP
#define EXPECTED_FREQUENCY_LOADER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

class ExpectedFrequencyLoader {
public:
    [[nodiscard]] static StatusOr<ExpectedFrequencyTable>
    load_from_string(const std::string& json_text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(json_text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid expected-frequency JSON: ") + ex.what());
        }
        return load_from_json(root);
    }

    [[nodiscard]] static StatusOr<ExpectedFrequencyTable> load_from_file(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return Status::error("Failed to open expected-frequency file: " + path);
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return load_from_string(buffer.str());
    }

private:
    [[nodiscard]] static StatusOr<ExpectedFrequencyTable>
    load_from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("expected-frequency root must be an object");
        }
        if (!root.contains("schema") ||
            root.at("schema").get<std::string>() != "parcae.expected_frequency.v0") {
            return Status::error("expected-frequency schema must be parcae.expected_frequency.v0");
        }
        if (!root.contains("id") || !root.at("id").is_string()) {
            return Status::error("expected-frequency missing id");
        }
        if (!root.contains("alphabet_size") ||
            root.at("alphabet_size").get<int>() !=
                static_cast<int>(ExpectedFrequencyTable::alphabet_size)) {
            return Status::error("expected-frequency alphabet_size must be 29");
        }
        if (!root.contains("probabilities") || !root.at("probabilities").is_array()) {
            return Status::error("expected-frequency missing probabilities array");
        }
        if (root.at("probabilities").size() != ExpectedFrequencyTable::alphabet_size) {
            return Status::error("expected-frequency probabilities must have length 29");
        }

        std::array<double, ExpectedFrequencyTable::alphabet_size> probs{};
        double sum = 0.0;
        for (std::size_t i = 0; i < ExpectedFrequencyTable::alphabet_size; ++i) {
            if (!root.at("probabilities")[i].is_number()) {
                return Status::error("expected-frequency probability must be numeric");
            }
            const double p = root.at("probabilities")[i].get<double>();
            if (!(p > 0.0)) {
                return Status::error("expected-frequency probabilities must be > 0");
            }
            probs[i] = p;
            sum += p;
        }
        if (sum < 0.999 || sum > 1.001) {
            return Status::error("expected-frequency probabilities must sum to ~1");
        }

        std::array<std::uint64_t, ExpectedFrequencyTable::alphabet_size> counts{};
        std::uint64_t total = 0;
        if (root.contains("raw_counts") && root.at("raw_counts").is_array() &&
            root.at("raw_counts").size() == ExpectedFrequencyTable::alphabet_size) {
            for (std::size_t i = 0; i < ExpectedFrequencyTable::alphabet_size; ++i) {
                counts[i] = root.at("raw_counts")[i].get<std::uint64_t>();
                total += counts[i];
            }
        }
        if (root.contains("raw_total") && root.at("raw_total").is_number_unsigned()) {
            total = root.at("raw_total").get<std::uint64_t>();
        }

        std::string smoothing = "none";
        if (root.contains("smoothing") && root.at("smoothing").is_string()) {
            smoothing = root.at("smoothing").get<std::string>();
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

        return ExpectedFrequencyTable(root.at("id").get<std::string>(), probs, counts, total,
                                      std::move(smoothing), std::move(sources));
    }
};

#endif // EXPECTED_FREQUENCY_LOADER_HPP
