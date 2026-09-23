#ifndef COMPOSE_RECIPE_CANDIDATE_GENERATOR_HPP
#define COMPOSE_RECIPE_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Bounded generator `gen_compose_recipes`.
///
/// Enumerates a **caller-supplied** list of compose `params` objects
/// (`{"stages":[…]}`). Empty / `template: "atbash_caesar"` expands to the
/// Koan-1 Atbash∘Caesar 29-shift grid via `AtbashCaesarCandidateGenerator`.
class ComposeRecipeCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_compose_recipes";
    static constexpr std::string_view atbash_caesar_template = "atbash_caesar";

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> generate(
        std::span<const Index29> ciphertext,
        const std::vector<nlohmann::json>& recipes,
        TransformDirection direction = TransformDirection::Decrypt) {
        if (recipes.empty()) {
            return Status::error("gen_compose_recipes requires a non-empty recipes list");
        }

        std::vector<TransformCandidate> out;
        out.reserve(recipes.size());
        for (std::size_t i = 0; i < recipes.size(); ++i) {
            const nlohmann::json& params = recipes[i];
            Status params_ok = require_compose_params(params);
            if (!params_ok.ok()) {
                return Status::error(
                    "gen_compose_recipes recipes[" + std::to_string(i) + "]: " +
                    params_ok.message());
            }

            StatusOr<std::vector<Index29>> plain =
                ComposeTransform{}.apply(ciphertext, params, direction);
            if (!plain.ok()) {
                return Status::error(
                    "gen_compose_recipes apply failed for index " + std::to_string(i) + ": " +
                    plain.status().message());
            }

            out.emplace_back(
                make_candidate_id(i, params),
                TransformId::compose(),
                direction,
                params,
                std::move(plain.value()));
        }
        return out;
    }

    /// Expand Atbash∘Caesar shifts `0..28` (reuses `AtbashCaesarCandidateGenerator`).
    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> generate_atbash_caesar(
        std::span<const Index29> ciphertext,
        TransformDirection direction = TransformDirection::Decrypt) {
        return AtbashCaesarCandidateGenerator::generate(ciphertext, direction);
    }

    [[nodiscard]] static std::vector<nlohmann::json> atbash_caesar_recipes() {
        std::vector<nlohmann::json> out;
        out.reserve(AtbashCaesarCandidateGenerator::candidate_count);
        for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
            out.push_back(ComposeTransform::atbash_then_caesar_params(shift));
        }
        return out;
    }

    /// Resolve `param_grid` → recipe list.
    ///
    /// - empty / null / `{}` → Atbash∘Caesar 29
    /// - `template` / `recipe` == `"atbash_caesar"` → Atbash∘Caesar 29
    /// - `recipes` / `params_list` → explicit compose params objects
    /// - bare `stages` → single recipe
    [[nodiscard]] static StatusOr<std::vector<nlohmann::json>> recipes_from_param_grid(
        const nlohmann::json& param_grid) {
        if (param_grid.is_null() || (param_grid.is_object() && param_grid.empty())) {
            return atbash_caesar_recipes();
        }
        if (!param_grid.is_object()) {
            return Status::error("gen_compose_recipes param_grid must be an object");
        }

        if (param_grid.contains("template") || param_grid.contains("recipe")) {
            const char* key = param_grid.contains("template") ? "template" : "recipe";
            if (!param_grid.at(key).is_string()) {
                return Status::error(
                    std::string("gen_compose_recipes param_grid.") + key +
                    " must be a string");
            }
            const std::string name = param_grid.at(key).get<std::string>();
            if (name != atbash_caesar_template) {
                return Status::error(
                    "gen_compose_recipes unknown template (expected atbash_caesar): " + name);
            }
            return atbash_caesar_recipes();
        }

        if (param_grid.contains("recipes") || param_grid.contains("params_list")) {
            const char* key = param_grid.contains("recipes") ? "recipes" : "params_list";
            if (!param_grid.at(key).is_array() || param_grid.at(key).empty()) {
                return Status::error(
                    std::string("gen_compose_recipes param_grid.") + key +
                    " must be a non-empty array");
            }
            std::vector<nlohmann::json> out;
            out.reserve(param_grid.at(key).size());
            for (const auto& item : param_grid.at(key)) {
                Status ok = require_compose_params(item);
                if (!ok.ok()) {
                    return Status::error(
                        std::string("gen_compose_recipes param_grid.") + key + ": " +
                        ok.message());
                }
                out.push_back(item);
            }
            return out;
        }

        if (param_grid.contains("stages")) {
            Status ok = require_compose_params(param_grid);
            if (!ok.ok()) {
                return ok;
            }
            return std::vector<nlohmann::json>{param_grid};
        }

        return Status::error(
            "gen_compose_recipes param_grid needs template/recipe, recipes/params_list, "
            "or stages (or be empty for atbash_caesar)");
    }

    /// True when `params` match Koan-1 Atbash then Caesar(+shift, encrypt stage).
    [[nodiscard]] static std::optional<std::uint8_t> atbash_caesar_shift(
        const nlohmann::json& params) {
        if (!params.is_object() || !params.contains("stages") || !params.at("stages").is_array()) {
            return std::nullopt;
        }
        const auto& stages = params.at("stages");
        if (stages.size() != 2) {
            return std::nullopt;
        }
        const auto& s0 = stages[0];
        const auto& s1 = stages[1];
        if (!s0.is_object() || !s1.is_object()) {
            return std::nullopt;
        }
        if (!s0.contains("transform_id") || !s0.at("transform_id").is_string() ||
            s0.at("transform_id").get<std::string>() != "atbash") {
            return std::nullopt;
        }
        if (!s1.contains("transform_id") || !s1.at("transform_id").is_string() ||
            s1.at("transform_id").get<std::string>() != "caesar") {
            return std::nullopt;
        }
        if (!s1.contains("direction") || !s1.at("direction").is_string() ||
            s1.at("direction").get<std::string>() != "encrypt") {
            return std::nullopt;
        }
        if (!s1.contains("params") || !s1.at("params").is_object() ||
            !s1.at("params").contains("shift") || !s1.at("params").at("shift").is_number_integer()) {
            return std::nullopt;
        }
        const std::int64_t shift = s1.at("params").at("shift").get<std::int64_t>();
        if (shift < 0 || shift > 28) {
            return std::nullopt;
        }
        return static_cast<std::uint8_t>(shift);
    }

    [[nodiscard]] static bool is_full_atbash_caesar_grid(
        const std::vector<nlohmann::json>& recipes) {
        if (recipes.size() != AtbashCaesarCandidateGenerator::candidate_count) {
            return false;
        }
        std::vector<bool> seen(Index29::modulus, false);
        for (const nlohmann::json& params : recipes) {
            std::optional<std::uint8_t> shift = atbash_caesar_shift(params);
            if (!shift.has_value() || seen[*shift]) {
                return false;
            }
            seen[*shift] = true;
        }
        return true;
    }

    [[nodiscard]] static std::string make_candidate_id(
        std::size_t list_index,
        const nlohmann::json& params) {
        if (std::optional<std::uint8_t> shift = atbash_caesar_shift(params)) {
            return AtbashCaesarCandidateGenerator::make_candidate_id(*shift);
        }
        return "compose:i=" + std::to_string(list_index);
    }

    [[nodiscard]] static Status require_compose_params(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("compose params must be an object");
        }
        if (!params.contains("stages") || !params.at("stages").is_array()) {
            return Status::error("compose params.stages must be an array");
        }
        if (params.at("stages").empty()) {
            return Status::error("compose params.stages must be non-empty");
        }
        // Mirror ComposeTransform: only `stages` key.
        if (params.size() != 1) {
            return Status::error("compose params may only contain stages");
        }
        return Status::success();
    }
};

#endif  // COMPOSE_RECIPE_CANDIDATE_GENERATOR_HPP
