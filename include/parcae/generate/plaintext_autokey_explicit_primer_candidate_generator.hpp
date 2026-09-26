#ifndef PLAINTEXT_AUTOKEY_EXPLICIT_PRIMER_CANDIDATE_GENERATOR_HPP
#define PLAINTEXT_AUTOKEY_EXPLICIT_PRIMER_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/plaintext_autokey_transform.hpp"
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

/// Bounded generator `gen_plaintext_autokey_explicit_primers`.
///
/// Enumerates a **caller-supplied** PTAK primer list only (cost = \|primers\|).
/// Same key-list shape as `gen_vigenere_explicit_keys`. No dictionary expansion.
class PlaintextAutokeyExplicitPrimerCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_plaintext_autokey_explicit_primers";

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext, const std::vector<ExplicitVigenereKey>& primers,
             TransformDirection direction = TransformDirection::Decrypt,
             const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (primers.empty()) {
            return Status::error(
                "gen_plaintext_autokey_explicit_primers requires a non-empty primer list");
        }

        const PlaintextAutokeyTransform transform;
        std::vector<TransformCandidate> out;
        out.reserve(primers.size());

        std::optional<nlohmann::json> interrupt_json;
        if (!interrupt.skip_indices().empty()) {
            interrupt_json = interrupt.to_json();
        }

        for (std::size_t i = 0; i < primers.size(); ++i) {
            const ExplicitVigenereKey& primer = primers[i];
            if (primer.key_indices.empty()) {
                return Status::error(
                    "gen_plaintext_autokey_explicit_primers key_indices must be non-empty");
            }

            StatusOr<nlohmann::json> params = make_params(primer);
            if (!params.ok()) {
                return params.status();
            }

            StatusOr<std::vector<Index29>> plain =
                transform.apply(ciphertext, params.value(), direction, interrupt);
            if (!plain.ok()) {
                return plain.status();
            }

            out.emplace_back(make_candidate_id(primer.key_indices, i),
                             TransformId::plaintext_autokey(), direction,
                             std::move(params.value()), std::move(plain.value()), interrupt_json);
        }
        return out;
    }

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext,
             const std::vector<std::vector<Index29>>& primer_indices_list,
             TransformDirection direction = TransformDirection::Decrypt,
             const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        std::vector<ExplicitVigenereKey> primers;
        primers.reserve(primer_indices_list.size());
        for (const std::vector<Index29>& primer : primer_indices_list) {
            primers.push_back(ExplicitVigenereKey{primer, std::nullopt});
        }
        return generate(ciphertext, primers, direction, interrupt);
    }

    [[nodiscard]] static std::string make_candidate_id(const std::vector<Index29>& key_indices,
                                                       std::size_t list_index) {
        std::string id = "plaintext_autokey:i=" + std::to_string(list_index) + ":key_indices=";
        for (std::size_t i = 0; i < key_indices.size(); ++i) {
            if (i != 0) {
                id.push_back(',');
            }
            id += std::to_string(static_cast<unsigned>(key_indices[i].value()));
        }
        return id;
    }

private:
    [[nodiscard]] static StatusOr<nlohmann::json> make_params(const ExplicitVigenereKey& primer) {
        nlohmann::json indices = nlohmann::json::array();
        for (const Index29 idx : primer.key_indices) {
            if (idx.value() >= Index29::modulus) {
                return Status::error("plaintext_autokey key_indices entry out of range");
            }
            indices.push_back(static_cast<int>(idx.value()));
        }
        nlohmann::json params{{"key_indices", std::move(indices)}};
        if (primer.key_latin.has_value()) {
            params["key_latin"] = primer.key_latin.value();
        }
        return params;
    }
};

#endif // PLAINTEXT_AUTOKEY_EXPLICIT_PRIMER_CANDIDATE_GENERATOR_HPP
