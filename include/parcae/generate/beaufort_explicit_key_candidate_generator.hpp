#ifndef BEAUFORT_EXPLICIT_KEY_CANDIDATE_GENERATOR_HPP
#define BEAUFORT_EXPLICIT_KEY_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstddef>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Bounded generator `gen_beaufort_explicit_keys`.
///
/// Same contract as `gen_vigenere_explicit_keys`: enumerates a **caller-supplied**
/// key list only (no dictionary / keyspace search). Opt-in search family
/// `beaufort` (requires `SearchJob.allow_extended_families`).
class BeaufortExplicitKeyCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_beaufort_explicit_keys";

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext, const std::vector<ExplicitVigenereKey>& keys,
             TransformDirection direction = TransformDirection::Decrypt,
             const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (keys.empty()) {
            return Status::error(
                "gen_beaufort_explicit_keys requires a non-empty caller-supplied key list");
        }

        const BeaufortKeyTransform transform;
        std::vector<TransformCandidate> out;
        out.reserve(keys.size());

        std::optional<nlohmann::json> interrupt_json;
        if (!interrupt.skip_indices().empty()) {
            interrupt_json = interrupt.to_json();
        }

        for (std::size_t i = 0; i < keys.size(); ++i) {
            const ExplicitVigenereKey& key = keys[i];
            if (key.key_indices.empty()) {
                return Status::error("gen_beaufort_explicit_keys key_indices must be non-empty");
            }

            StatusOr<nlohmann::json> params = make_params(key);
            if (!params.ok()) {
                return params.status();
            }

            StatusOr<std::vector<Index29>> plain =
                transform.apply(ciphertext, params.value(), direction, interrupt);
            if (!plain.ok()) {
                return plain.status();
            }

            out.emplace_back(make_candidate_id(key.key_indices, i), TransformId::beaufort_key(),
                             direction, std::move(params.value()), std::move(plain.value()),
                             interrupt_json);
        }
        return out;
    }

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext,
             const std::vector<std::vector<Index29>>& key_indices_list,
             TransformDirection direction = TransformDirection::Decrypt,
             const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        std::vector<ExplicitVigenereKey> keys;
        keys.reserve(key_indices_list.size());
        for (const std::vector<Index29>& key : key_indices_list) {
            keys.push_back(ExplicitVigenereKey{key, std::nullopt});
        }
        return generate(ciphertext, keys, direction, interrupt);
    }

    [[nodiscard]] static std::string make_candidate_id(const std::vector<Index29>& key_indices,
                                                       std::size_t list_index) {
        std::string id = "beaufort_key:i=" + std::to_string(list_index) + ":key_indices=";
        for (std::size_t i = 0; i < key_indices.size(); ++i) {
            if (i != 0) {
                id.push_back(',');
            }
            id += std::to_string(static_cast<unsigned>(key_indices[i].value()));
        }
        return id;
    }

private:
    [[nodiscard]] static StatusOr<nlohmann::json> make_params(const ExplicitVigenereKey& key) {
        nlohmann::json indices = nlohmann::json::array();
        for (const Index29 idx : key.key_indices) {
            if (idx.value() >= Index29::modulus) {
                return Status::error("beaufort key_indices entry out of range");
            }
            indices.push_back(static_cast<int>(idx.value()));
        }
        nlohmann::json params{{"key_indices", std::move(indices)}};
        if (key.key_latin.has_value()) {
            params["key_latin"] = key.key_latin.value();
        }
        return params;
    }
};

#endif // BEAUFORT_EXPLICIT_KEY_CANDIDATE_GENERATOR_HPP
