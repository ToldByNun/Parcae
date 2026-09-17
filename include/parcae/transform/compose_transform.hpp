#ifndef COMPOSE_TRANSFORM_HPP
#define COMPOSE_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/identity_transform.hpp"
#include "parcae/transform/totient_prime_stream_transform.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

/// Stage-list composition.
/// Stages describe the **decrypt** pipeline in array order. Outer encrypt applies
/// stages in reverse with inverted per-stage directions.
/// Optional per-stage `direction` overrides the default (`decrypt` on the decrypt
/// path) — Koan 1 uses Caesar with `"direction": "encrypt"` so decrypt does +shift.
/// Nested `compose` is allowed up to `max_depth` (default 8, MUST be ≥ 4).
class ComposeTransform : public Transform {
public:
    static constexpr std::size_t max_depth = 8;

    ComposeTransform() = default;

    [[nodiscard]] TransformId id() const override {
        return TransformId::compose();
    }

    [[nodiscard]] StatusOr<std::vector<Index29>> apply(
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        return apply_with_depth(input, params, direction, interrupt, 0);
    }

    /// Params shape used by the `koan-1` fixture (Atbash then Caesar +shift).
    [[nodiscard]] static nlohmann::json atbash_then_caesar_params(std::uint8_t shift = 3) {
        return nlohmann::json{
            {"stages",
             nlohmann::json::array(
                 {nlohmann::json{{"transform_id", "atbash"}, {"params", nlohmann::json::object()}},
                  nlohmann::json{
                      {"transform_id", "caesar"},
                      {"direction", "encrypt"},
                      {"params", {{"shift", shift}}},
                  }})},
        };
    }

    /// Koan 1 solution path: decrypt = Atbash then +shift; encrypt = −shift then Atbash.
    [[nodiscard]] static StatusOr<std::vector<Index29>> apply_atbash_then_caesar(
        std::span<const Index29> input,
        std::uint8_t shift,
        TransformDirection direction) {
        if (shift > 28) {
            return Status::error("compose atbash_then_caesar shift must be in 0..28");
        }
        return ComposeTransform{}.apply(
            input,
            atbash_then_caesar_params(shift),
            direction,
            InterruptPolicy::none());
    }

private:
    [[nodiscard]] static TransformDirection invert_direction(TransformDirection direction) {
        return direction == TransformDirection::Decrypt ? TransformDirection::Encrypt
                                                        : TransformDirection::Decrypt;
    }

    [[nodiscard]] static StatusOr<TransformDirection> resolve_stage_direction(
        const nlohmann::json& stage,
        TransformDirection recipe_default) {
        if (!stage.contains("direction")) {
            return recipe_default;
        }
        if (!stage.at("direction").is_string()) {
            return Status::error("compose stage direction must be a string");
        }
        return TransformDirectionUtil::from_string(stage.at("direction").get<std::string>());
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply_with_depth(
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt,
        std::size_t depth) {
        if (depth >= max_depth) {
            return Status::error("compose nesting exceeds max_depth");
        }
        if (!params.is_object() || !params.contains("stages") || !params.at("stages").is_array()) {
            return Status::error("compose params.stages must be an array");
        }
        if (params.size() != 1) {
            return Status::error("compose params may only contain stages");
        }

        const nlohmann::json& stages = params.at("stages");
        if (stages.empty()) {
            return Status::error("compose params.stages must be non-empty");
        }

        std::vector<Index29> current(input.begin(), input.end());
        if (direction == TransformDirection::Decrypt) {
            for (const nlohmann::json& stage : stages) {
                StatusOr<TransformDirection> stage_direction =
                    resolve_stage_direction(stage, TransformDirection::Decrypt);
                if (!stage_direction.ok()) {
                    return stage_direction.status();
                }
                StatusOr<std::vector<Index29>> next =
                    apply_stage(current, stage, stage_direction.value(), interrupt, depth);
                if (!next.ok()) {
                    return next.status();
                }
                current = std::move(next.value());
            }
        } else {
            for (auto it = stages.rbegin(); it != stages.rend(); ++it) {
                StatusOr<TransformDirection> recipe_direction =
                    resolve_stage_direction(*it, TransformDirection::Decrypt);
                if (!recipe_direction.ok()) {
                    return recipe_direction.status();
                }
                StatusOr<std::vector<Index29>> next = apply_stage(
                    current,
                    *it,
                    invert_direction(recipe_direction.value()),
                    interrupt,
                    depth);
                if (!next.ok()) {
                    return next.status();
                }
                current = std::move(next.value());
            }
        }
        return current;
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply_stage(
        std::span<const Index29> input,
        const nlohmann::json& stage,
        TransformDirection direction,
        const InterruptPolicy& interrupt,
        std::size_t depth) {
        if (!stage.is_object() || !stage.contains("transform_id")) {
            return Status::error("compose stage requires transform_id");
        }
        StatusOr<TransformId> id =
            TransformId::from_string(stage.at("transform_id").get<std::string>());
        if (!id.ok()) {
            return id.status();
        }

        const nlohmann::json stage_params =
            stage.contains("params") ? stage.at("params") : nlohmann::json::object();

        // Outer interrupt applies only to the outer keyed stage in later families;
        // non-keyed stages ignore it. Nested compose receives none unless the stage
        // carries its own interrupt object.
        InterruptPolicy stage_interrupt = InterruptPolicy::none();
        if (stage.contains("interrupt")) {
            StatusOr<InterruptPolicy> parsed = InterruptPolicy::from_json(stage.at("interrupt"));
            if (!parsed.ok()) {
                return parsed.status();
            }
            stage_interrupt = std::move(parsed.value());
        } else if (depth == 0) {
            stage_interrupt = interrupt;
        }

        if (id.value() == TransformId::identity()) {
            return IdentityTransform{}.apply(input, stage_params, direction, stage_interrupt);
        }
        if (id.value() == TransformId::atbash()) {
            return AtbashTransform{}.apply(input, stage_params, direction, stage_interrupt);
        }
        if (id.value() == TransformId::caesar()) {
            return CaesarTransform{}.apply(input, stage_params, direction, stage_interrupt);
        }
        if (id.value() == TransformId::affine()) {
            return AffineTransform{}.apply(input, stage_params, direction, stage_interrupt);
        }
        if (id.value() == TransformId::vigenere_key()) {
            return VigenereKeyTransform{}.apply(input, stage_params, direction, stage_interrupt);
        }
        if (id.value() == TransformId::beaufort_key()) {
            return BeaufortKeyTransform{}.apply(input, stage_params, direction, stage_interrupt);
        }
        if (id.value() == TransformId::totient_prime_stream()) {
            return TotientPrimeStreamTransform{}.apply(
                input, stage_params, direction, stage_interrupt);
        }
        if (id.value() == TransformId::compose()) {
            return apply_with_depth(input, stage_params, direction, stage_interrupt, depth + 1);
        }
        return Status::error("compose stage transform_id is not supported yet");
    }
};

#endif // COMPOSE_TRANSFORM_HPP
