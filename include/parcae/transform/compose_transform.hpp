#ifndef COMPOSE_TRANSFORM_HPP
#define COMPOSE_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/ciphertext_autokey_transform.hpp"
#include "parcae/transform/hill2_transform.hpp"
#include "parcae/transform/hill3_transform.hpp"
#include "parcae/transform/identity_transform.hpp"
#include "parcae/transform/plaintext_autokey_transform.hpp"
#include "parcae/transform/totient_prime_stream_transform.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"
#include "parcae/transform/variable_delay_autokey_transform.hpp"
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
///
/// Staging uses at most two length-N scratch buffers (ping-pong), not one allocation
/// per stage.
class ComposeTransform : public Transform {
public:
    static constexpr std::size_t max_depth = 8;

    ComposeTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::compose(); }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        return apply_into_with_depth(input, output, params, direction, interrupt, 0);
    }

    /// Params shape used by the `koan-1` fixture (Atbash then Caesar +shift).
    [[nodiscard]] static nlohmann::json atbash_then_caesar_params(std::uint8_t shift = 3) {
        return nlohmann::json{
            {"stages", nlohmann::json::array({nlohmann::json{{"transform_id", "atbash"},
                                                             {"params", nlohmann::json::object()}},
                                              nlohmann::json{
                                                  {"transform_id", "caesar"},
                                                  {"direction", "encrypt"},
                                                  {"params", {{"shift", shift}}},
                                              }})},
        };
    }

    /// Koan 1 solution path: decrypt = Atbash then +shift; encrypt = −shift then Atbash.
    [[nodiscard]] static StatusOr<std::vector<Index29>>
    apply_atbash_then_caesar(std::span<const Index29> input, std::uint8_t shift,
                             TransformDirection direction) {
        if (shift > 28) {
            return Status::error("compose atbash_then_caesar shift must be in 0..28");
        }
        return ComposeTransform{}.apply(input, atbash_then_caesar_params(shift), direction,
                                        InterruptPolicy::none());
    }

private:
    [[nodiscard]] static TransformDirection invert_direction(TransformDirection direction) {
        return direction == TransformDirection::Decrypt ? TransformDirection::Encrypt
                                                        : TransformDirection::Decrypt;
    }

    [[nodiscard]] static StatusOr<TransformDirection>
    resolve_stage_direction(const nlohmann::json& stage, TransformDirection recipe_default) {
        if (!stage.contains("direction")) {
            return recipe_default;
        }
        if (!stage.at("direction").is_string()) {
            return Status::error("compose stage direction must be a string");
        }
        return TransformDirectionUtil::from_string(stage.at("direction").get<std::string>());
    }

    [[nodiscard]] static Status
    apply_into_with_depth(std::span<const Index29> input, std::span<Index29> output,
                          const nlohmann::json& params, TransformDirection direction,
                          const InterruptPolicy& interrupt, std::size_t depth) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
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

        // Ping-pong: at most two N-length scratch buffers for multi-stage pipelines.
        std::vector<Index29> scratch_a;
        std::vector<Index29> scratch_b;
        const bool multi = stages.size() > 1;
        if (multi) {
            scratch_a.assign(input.begin(), input.end());
            scratch_b.resize(input.size());
        }

        std::span<const Index29> current_in = input;
        std::span<Index29> current_out = multi ? std::span<Index29>(scratch_b) : output;
        bool write_to_b = true;

        auto run_stage = [&](const nlohmann::json& stage, TransformDirection stage_dir) -> Status {
            return apply_stage_into(current_in, current_out, stage, stage_dir, interrupt, depth);
        };

        auto advance = [&]() {
            if (!multi) {
                return;
            }
            if (write_to_b) {
                current_in = scratch_b;
                current_out = scratch_a;
                write_to_b = false;
            } else {
                current_in = scratch_a;
                current_out = scratch_b;
                write_to_b = true;
            }
        };

        if (direction == TransformDirection::Decrypt) {
            for (std::size_t s = 0; s < stages.size(); ++s) {
                const bool last = (s + 1 == stages.size());
                if (last) {
                    current_out = output;
                }
                StatusOr<TransformDirection> stage_direction =
                    resolve_stage_direction(stages[s], TransformDirection::Decrypt);
                if (!stage_direction.ok()) {
                    return stage_direction.status();
                }
                Status status = run_stage(stages[s], stage_direction.value());
                if (!status.ok()) {
                    return status;
                }
                if (!last) {
                    advance();
                }
            }
        } else {
            std::size_t remaining = stages.size();
            for (auto it = stages.rbegin(); it != stages.rend(); ++it) {
                --remaining;
                const bool last = (remaining == 0);
                if (last) {
                    current_out = output;
                }
                StatusOr<TransformDirection> recipe_direction =
                    resolve_stage_direction(*it, TransformDirection::Decrypt);
                if (!recipe_direction.ok()) {
                    return recipe_direction.status();
                }
                Status status = run_stage(*it, invert_direction(recipe_direction.value()));
                if (!status.ok()) {
                    return status;
                }
                if (!last) {
                    advance();
                }
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status
    apply_stage_into(std::span<const Index29> input, std::span<Index29> output,
                     const nlohmann::json& stage, TransformDirection direction,
                     const InterruptPolicy& interrupt, std::size_t depth) {
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
            return IdentityTransform{}.apply_into(input, output, stage_params, direction,
                                                  stage_interrupt);
        }
        if (id.value() == TransformId::atbash()) {
            return AtbashTransform{}.apply_into(input, output, stage_params, direction,
                                                stage_interrupt);
        }
        if (id.value() == TransformId::caesar()) {
            return CaesarTransform{}.apply_into(input, output, stage_params, direction,
                                                stage_interrupt);
        }
        if (id.value() == TransformId::affine()) {
            return AffineTransform{}.apply_into(input, output, stage_params, direction,
                                                stage_interrupt);
        }
        if (id.value() == TransformId::hill_2()) {
            return Hill2Transform{}.apply_into(input, output, stage_params, direction,
                                               stage_interrupt);
        }
        if (id.value() == TransformId::hill_3()) {
            return Hill3Transform{}.apply_into(input, output, stage_params, direction,
                                               stage_interrupt);
        }
        if (id.value() == TransformId::vigenere_key()) {
            return VigenereKeyTransform{}.apply_into(input, output, stage_params, direction,
                                                     stage_interrupt);
        }
        if (id.value() == TransformId::ciphertext_autokey()) {
            return CiphertextAutokeyTransform{}.apply_into(input, output, stage_params, direction,
                                                           stage_interrupt);
        }
        if (id.value() == TransformId::plaintext_autokey()) {
            return PlaintextAutokeyTransform{}.apply_into(input, output, stage_params, direction,
                                                          stage_interrupt);
        }
        if (id.value() == TransformId::variable_delay_autokey()) {
            return VariableDelayAutokeyTransform{}.apply_into(input, output, stage_params, direction,
                                                             stage_interrupt);
        }
        if (id.value() == TransformId::beaufort_key()) {
            return BeaufortKeyTransform{}.apply_into(input, output, stage_params, direction,
                                                     stage_interrupt);
        }
        if (id.value() == TransformId::totient_prime_stream()) {
            return TotientPrimeStreamTransform{}.apply_into(input, output, stage_params, direction,
                                                            stage_interrupt);
        }
        if (id.value() == TransformId::compose()) {
            return apply_into_with_depth(input, output, stage_params, direction, stage_interrupt,
                                         depth + 1);
        }
        return Status::error("compose stage transform_id is not supported yet");
    }
};

#endif // COMPOSE_TRANSFORM_HPP
