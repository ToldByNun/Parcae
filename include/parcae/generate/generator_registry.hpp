#ifndef GENERATOR_REGISTRY_HPP
#define GENERATOR_REGISTRY_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/generator_catalog_entry.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

/// String-id dispatch for bounded candidate generators (`gen_*`).
class GeneratorRegistry {
public:
    [[nodiscard]] static std::vector<GeneratorCatalogEntry> catalog() {
        return {
            {std::string(AtbashCandidateGenerator::generator_id),
             TransformId::atbash().str(),
             AtbashCandidateGenerator::candidate_count,
             false},
            {std::string(CaesarCandidateGenerator::generator_id),
             TransformId::caesar().str(),
             CaesarCandidateGenerator::candidate_count,
             false},
            {std::string(AtbashCaesarCandidateGenerator::generator_id),
             TransformId::compose().str(),
             AtbashCaesarCandidateGenerator::candidate_count,
             false},
            {std::string(AffineCandidateGenerator::generator_id),
             TransformId::affine().str(),
             AffineCandidateGenerator::candidate_count,
             false},
            {std::string(VigenereExplicitKeyCandidateGenerator::generator_id),
             TransformId::vigenere_key().str(),
             0,
             true},
        };
    }

    [[nodiscard]] static std::vector<std::string> list_generator_ids() {
        std::vector<std::string> ids;
        for (const GeneratorCatalogEntry& entry : catalog()) {
            ids.push_back(entry.id());
        }
        return ids;
    }

    [[nodiscard]] static bool is_known(std::string_view generator_id) {
        for (const GeneratorCatalogEntry& entry : catalog()) {
            if (entry.id() == generator_id) {
                return true;
            }
        }
        return false;
    }

    /// Dispatch `gen_*` → candidates. For `gen_vigenere_explicit_keys`, `params`
    /// MUST contain `key_indices_list` (array of Index29 arrays) or `keys`
    /// (array of `{key_indices, key_latin?}` objects).
    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> generate(
        std::string_view generator_id,
        std::span<const Index29> ciphertext,
        TransformDirection direction = TransformDirection::Decrypt,
        const nlohmann::json& params = nlohmann::json::object()) {
        if (generator_id == AtbashCandidateGenerator::generator_id) {
            return AtbashCandidateGenerator::generate(ciphertext, direction);
        }
        if (generator_id == CaesarCandidateGenerator::generator_id) {
            return CaesarCandidateGenerator::generate(ciphertext, direction);
        }
        if (generator_id == AtbashCaesarCandidateGenerator::generator_id) {
            return AtbashCaesarCandidateGenerator::generate(ciphertext, direction);
        }
        if (generator_id == AffineCandidateGenerator::generator_id) {
            return AffineCandidateGenerator::generate(ciphertext, direction);
        }
        if (generator_id == VigenereExplicitKeyCandidateGenerator::generator_id) {
            return generate_vigenere(ciphertext, direction, params);
        }
        return Status::error("Unknown generator_id");
    }

private:
    GeneratorRegistry() = delete;

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> generate_vigenere(
        std::span<const Index29> ciphertext,
        TransformDirection direction,
        const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("gen_vigenere_explicit_keys params must be an object");
        }

        if (params.contains("key_indices_list")) {
            if (!params.at("key_indices_list").is_array()) {
                return Status::error(
                    "gen_vigenere_explicit_keys params.key_indices_list must be an array");
            }
            std::vector<std::vector<Index29>> keys;
            keys.reserve(params.at("key_indices_list").size());
            for (const auto& key_json : params.at("key_indices_list")) {
                StatusOr<std::vector<Index29>> key = parse_index_array(key_json, "key_indices_list");
                if (!key.ok()) {
                    return key.status();
                }
                keys.push_back(std::move(key.value()));
            }
            return VigenereExplicitKeyCandidateGenerator::generate(
                ciphertext, keys, direction);
        }

        if (params.contains("keys")) {
            if (!params.at("keys").is_array()) {
                return Status::error("gen_vigenere_explicit_keys params.keys must be an array");
            }
            std::vector<ExplicitVigenereKey> keys;
            keys.reserve(params.at("keys").size());
            for (const auto& item : params.at("keys")) {
                if (!item.is_object()) {
                    return Status::error(
                        "gen_vigenere_explicit_keys params.keys entries must be objects");
                }
                if (!item.contains("key_indices")) {
                    return Status::error(
                        "gen_vigenere_explicit_keys params.keys[].key_indices is required");
                }
                StatusOr<std::vector<Index29>> indices =
                    parse_index_array(item.at("key_indices"), "keys[].key_indices");
                if (!indices.ok()) {
                    return indices.status();
                }
                ExplicitVigenereKey key;
                key.key_indices = std::move(indices.value());
                if (item.contains("key_latin")) {
                    if (!item.at("key_latin").is_string()) {
                        return Status::error(
                            "gen_vigenere_explicit_keys params.keys[].key_latin must be a string");
                    }
                    key.key_latin = item.at("key_latin").get<std::string>();
                }
                keys.push_back(std::move(key));
            }
            return VigenereExplicitKeyCandidateGenerator::generate(
                ciphertext, keys, direction);
        }

        return Status::error(
            "gen_vigenere_explicit_keys requires params.key_indices_list or params.keys");
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> parse_index_array(
        const nlohmann::json& arr,
        std::string_view field_name) {
        if (!arr.is_array()) {
            return Status::error(
                std::string("gen_vigenere_explicit_keys ") + std::string(field_name) +
                " must be an array");
        }
        std::vector<Index29> out;
        out.reserve(arr.size());
        for (const auto& item : arr) {
            if (!item.is_number_integer()) {
                return Status::error(
                    std::string("gen_vigenere_explicit_keys ") + std::string(field_name) +
                    " entries must be integers");
            }
            const int value = item.get<int>();
            if (value < 0 || value >= static_cast<int>(Index29::modulus)) {
                return Status::error(
                    std::string("gen_vigenere_explicit_keys ") + std::string(field_name) +
                    " entry out of range [0,28]");
            }
            out.push_back(Index29{static_cast<std::uint8_t>(value)});
        }
        return out;
    }
};

#endif  // GENERATOR_REGISTRY_HPP
