#ifndef COMPOSE_TRANSFORM_HPP
#define COMPOSE_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/identity_transform.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

/// Stage-list composition. Decrypt applies stages in order; encrypt reverses.
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

    /// Params shape used by the `koan-1` fixture (Atbash then Caesar shift=3).
    [[nodiscard]] static nlohmann::json atbash_then_caesar_params(std::uint8_t shift = 3) {
        return nlohmann::json{
            {"stages",
             nlohmann::json::array(
                 {nlohmann::json{{"transform_id", "atbash"}, {"params", nlohmann::json::object()}},
                  nlohmann::json{
                      {"transform_id", "caesar"},
                      {"params", {{"shift", shift}}},
                  }})},
        };
    }

    /// Koan 1 solution path: decrypt = Atbash then +shift; encrypt = −shift then Atbash.
    /// (Community wiki / solved-methods: `t = 28 - c`, then `p = (t + shift) mod 29`.)
    [[nodiscard]] static StatusOr<std::vector<Index29>> apply_atbash_then_caesar(
        std::span<const Index29> input,
        std::uint8_t shift,
        TransformDirection direction) {
        if (shift > 28) {
            return Status::error("compose atbash_then_caesar shift must be in 0..28");
        }

        const AtbashTransform atbash;
        const CaesarTransform caesar;
        const nlohmann::json caesar_params{{"shift", shift}};

        if (direction == TransformDirection::Decrypt) {
            StatusOr<std::vector<Index29>> reflected =
                atbash.apply(input, nlohmann::json::object(), TransformDirection::Decrypt);
            if (!reflected.ok()) {
                return reflected.status();
            }
            return caesar.apply(
                reflected.value(),
                caesar_params,
                TransformDirection::Encrypt);
        }

        StatusOr<std::vector<Index29>> shifted =
            caesar.apply(input, caesar_params, TransformDirection::Decrypt);
        if (!shifted.ok()) {
            return shifted.status();
        }
        return atbash.apply(
            shifted.value(),
            nlohmann::json::object(),
            TransformDirection::Encrypt);
    }

private:
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
                StatusOr<std::vector<Index29>> next =
                    apply_stage(current, stage, TransformDirection::Decrypt, interrupt, depth);
                if (!next.ok()) {
                    return next.status();
                }
                current = std::move(next.value());
            }
        } else {
            for (auto it = stages.rbegin(); it != stages.rend(); ++it) {
                StatusOr<std::vector<Index29>> next =
                    apply_stage(current, *it, TransformDirection::Encrypt, interrupt, depth);
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
        if (id.value() == TransformId::compose()) {
            return apply_with_depth(input, stage_params, direction, stage_interrupt, depth + 1);
        }
        return Status::error("compose stage transform_id is not supported yet");
    }
};

#endif // COMPOSE_TRANSFORM_HPP
