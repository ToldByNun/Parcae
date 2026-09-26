#ifndef GENERATOR_REGISTRY_HPP
#define GENERATOR_REGISTRY_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/beaufort_explicit_key_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/compose_recipe_candidate_generator.hpp"
#include "parcae/generate/generator_catalog_entry.hpp"
#include "parcae/generate/hill2_candidate_generator.hpp"
#include "parcae/generate/hill3_candidate_generator.hpp"
#include "parcae/generate/totient_offset_candidate_generator.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// String-id dispatch for bounded candidate generators (`gen_*`).
class GeneratorRegistry {
public:
    [[nodiscard]] static std::vector<GeneratorCatalogEntry> catalog() {
        return {
            {std::string(AtbashCandidateGenerator::generator_id), TransformId::atbash().str(),
             AtbashCandidateGenerator::candidate_count, false},
            {std::string(CaesarCandidateGenerator::generator_id), TransformId::caesar().str(),
             CaesarCandidateGenerator::candidate_count, false},
            {std::string(AtbashCaesarCandidateGenerator::generator_id),
             TransformId::compose().str(), AtbashCaesarCandidateGenerator::candidate_count, false},
            {std::string(AffineCandidateGenerator::generator_id), TransformId::affine().str(),
             AffineCandidateGenerator::candidate_count, false},
            {std::string(Hill2CandidateGenerator::generator_id), TransformId::hill_2().str(), 0,
             true},
            {std::string(Hill3CandidateGenerator::generator_id), TransformId::hill_3().str(), 0,
             true},
            {std::string(VigenereExplicitKeyCandidateGenerator::generator_id),
             TransformId::vigenere_key().str(), 0, true},
            {std::string(BeaufortExplicitKeyCandidateGenerator::generator_id),
             TransformId::beaufort_key().str(), 0, true},
            {std::string(TotientOffsetCandidateGenerator::generator_id),
             TransformId::totient_prime_stream().str(), 0, true},
            {std::string(ComposeRecipeCandidateGenerator::generator_id),
             TransformId::compose().str(), 0, true},
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

    /// Dispatch `gen_*` → candidates.
    ///
    /// Keyed generators (`gen_vigenere_explicit_keys` / `gen_beaufort_explicit_keys`):
    /// `params` MUST contain `key_indices_list` or `keys`.
    /// `gen_totient_offsets`: `params.prime_start_indices` (array of ints) required.
    /// `gen_hill_2` / `gen_hill_3`: empty → seed-bounded sample; or `matrices` /
    /// `max_candidates`+`seed` (see Hill2/Hill3CandidateGenerator).
    /// `gen_compose_recipes`: empty → Atbash∘Caesar 29; or `recipes` / `params_list` /
    /// `stages` / `template`=`atbash_caesar` (see ComposeRecipeCandidateGenerator).
    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::string_view generator_id, std::span<const Index29> ciphertext,
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
        if (generator_id == Hill2CandidateGenerator::generator_id) {
            return Hill2CandidateGenerator::generate(ciphertext, direction, params);
        }
        if (generator_id == Hill3CandidateGenerator::generator_id) {
            return Hill3CandidateGenerator::generate(ciphertext, direction, params);
        }
        if (generator_id == VigenereExplicitKeyCandidateGenerator::generator_id) {
            StatusOr<std::vector<ExplicitVigenereKey>> keys =
                parse_explicit_keys(params, "gen_vigenere_explicit_keys");
            if (!keys.ok()) {
                return keys.status();
            }
            return VigenereExplicitKeyCandidateGenerator::generate(ciphertext, keys.value(),
                                                                   direction);
        }
        if (generator_id == BeaufortExplicitKeyCandidateGenerator::generator_id) {
            StatusOr<std::vector<ExplicitVigenereKey>> keys =
                parse_explicit_keys(params, "gen_beaufort_explicit_keys");
            if (!keys.ok()) {
                return keys.status();
            }
            return BeaufortExplicitKeyCandidateGenerator::generate(ciphertext, keys.value(),
                                                                   direction);
        }
        if (generator_id == TotientOffsetCandidateGenerator::generator_id) {
            return generate_totient(ciphertext, direction, params);
        }
        if (generator_id == ComposeRecipeCandidateGenerator::generator_id) {
            return generate_compose(ciphertext, direction, params);
        }
        return Status::error("Unknown generator_id");
    }

private:
    GeneratorRegistry() = delete;

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate_compose(std::span<const Index29> ciphertext, TransformDirection direction,
                     const nlohmann::json& params) {
        StatusOr<std::vector<nlohmann::json>> recipes =
            ComposeRecipeCandidateGenerator::recipes_from_param_grid(
                params.is_null() ? nlohmann::json::object() : params);
        if (!recipes.ok()) {
            return recipes.status();
        }
        if (ComposeRecipeCandidateGenerator::is_full_atbash_caesar_grid(recipes.value())) {
            return ComposeRecipeCandidateGenerator::generate_atbash_caesar(ciphertext, direction);
        }
        return ComposeRecipeCandidateGenerator::generate(ciphertext, recipes.value(), direction);
    }

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate_totient(std::span<const Index29> ciphertext, TransformDirection direction,
                     const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("gen_totient_offsets params must be an object");
        }
        if (!params.contains("prime_start_indices") ||
            !params.at("prime_start_indices").is_array()) {
            return Status::error("gen_totient_offsets requires params.prime_start_indices array");
        }
        std::vector<std::size_t> starts;
        starts.reserve(params.at("prime_start_indices").size());
        for (const auto& item : params.at("prime_start_indices")) {
            if (!item.is_number_integer()) {
                return Status::error(
                    "gen_totient_offsets prime_start_indices entries must be integers");
            }
            const std::int64_t raw = item.get<std::int64_t>();
            if (raw < 0) {
                return Status::error(
                    "gen_totient_offsets prime_start_indices must be non-negative");
            }
            starts.push_back(static_cast<std::size_t>(raw));
        }
        return TotientOffsetCandidateGenerator::generate(ciphertext, starts, direction);
    }

    [[nodiscard]] static StatusOr<std::vector<ExplicitVigenereKey>>
    parse_explicit_keys(const nlohmann::json& params, std::string_view generator_id) {
        if (!params.is_object()) {
            return Status::error(std::string(generator_id) + " params must be an object");
        }

        if (params.contains("key_indices_list")) {
            if (!params.at("key_indices_list").is_array()) {
                return Status::error(std::string(generator_id) +
                                     " params.key_indices_list must be an array");
            }
            std::vector<ExplicitVigenereKey> keys;
            keys.reserve(params.at("key_indices_list").size());
            for (const auto& key_json : params.at("key_indices_list")) {
                StatusOr<std::vector<Index29>> key =
                    parse_index_array(key_json, generator_id, "key_indices_list");
                if (!key.ok()) {
                    return key.status();
                }
                keys.push_back(ExplicitVigenereKey{std::move(key.value()), std::nullopt});
            }
            return keys;
        }

        if (params.contains("keys")) {
            if (!params.at("keys").is_array()) {
                return Status::error(std::string(generator_id) + " params.keys must be an array");
            }
            std::vector<ExplicitVigenereKey> keys;
            keys.reserve(params.at("keys").size());
            for (const auto& item : params.at("keys")) {
                if (!item.is_object()) {
                    return Status::error(std::string(generator_id) +
                                         " params.keys entries must be objects");
                }
                if (!item.contains("key_indices")) {
                    return Status::error(std::string(generator_id) +
                                         " params.keys[].key_indices is required");
                }
                StatusOr<std::vector<Index29>> indices =
                    parse_index_array(item.at("key_indices"), generator_id, "keys[].key_indices");
                if (!indices.ok()) {
                    return indices.status();
                }
                ExplicitVigenereKey key;
                key.key_indices = std::move(indices.value());
                if (item.contains("key_latin")) {
                    if (!item.at("key_latin").is_string()) {
                        return Status::error(std::string(generator_id) +
                                             " params.keys[].key_latin must be a string");
                    }
                    key.key_latin = item.at("key_latin").get<std::string>();
                }
                keys.push_back(std::move(key));
            }
            return keys;
        }

        return Status::error(std::string(generator_id) +
                             " requires params.key_indices_list or params.keys");
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>>
    parse_index_array(const nlohmann::json& arr, std::string_view generator_id,
                      std::string_view field_name) {
        if (!arr.is_array()) {
            return Status::error(std::string(generator_id) + " " + std::string(field_name) +
                                 " must be an array");
        }
        std::vector<Index29> out;
        out.reserve(arr.size());
        for (const auto& item : arr) {
            if (!item.is_number_integer()) {
                return Status::error(std::string(generator_id) + " " + std::string(field_name) +
                                     " entries must be integers");
            }
            const int value = item.get<int>();
            if (value < 0 || value >= static_cast<int>(Index29::modulus)) {
                return Status::error(std::string(generator_id) + " " + std::string(field_name) +
                                     " entry out of range [0,28]");
            }
            out.push_back(Index29{static_cast<std::uint8_t>(value)});
        }
        return out;
    }
};

#endif // GENERATOR_REGISTRY_HPP
