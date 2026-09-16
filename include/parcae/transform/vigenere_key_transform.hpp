#ifndef VIGENERE_KEY_TRANSFORM_HPP
#define VIGENERE_KEY_TRANSFORM_HPP

#include "parcae/core/z29.hpp"
#include "parcae/transform/transform.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/// Explicit-key Vigenère over Z29.
/// Non-interrupt positions: decrypt `in - key[j]`, encrypt `in + key[j]`; then `j++`.
/// Interrupt positions: pass-through; key cursor does not advance.
class VigenereKeyTransform : public Transform {
public:
    VigenereKeyTransform() = default;

    [[nodiscard]] TransformId id() const override {
        return TransformId::vigenere_key();
    }

    [[nodiscard]] StatusOr<std::vector<Index29>> apply(
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        StatusOr<std::vector<Index29>> key = parse_key_indices(params);
        if (!key.ok()) {
            return key.status();
        }

        Status range = validate_interrupt_range(interrupt, input.size());
        if (!range.ok()) {
            return range;
        }

        const std::vector<Index29>& key_indices = key.value();
        const std::size_t key_len = key_indices.size();

        std::vector<Index29> out;
        out.reserve(input.size());
        std::size_t key_cursor = 0;

        for (std::size_t i = 0; i < input.size(); ++i) {
            if (interrupt.should_skip(i)) {
                out.push_back(input[i]);
                continue;
            }

            const Index29 key_symbol = key_indices[key_cursor % key_len];
            if (direction == TransformDirection::Encrypt) {
                out.push_back(Z29::add(input[i], key_symbol));
            } else {
                out.push_back(Z29::sub(input[i], key_symbol));
            }
            ++key_cursor;
        }

        return out;
    }

private:
    [[nodiscard]] static StatusOr<std::vector<Index29>> parse_key_indices(
        const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("vigenere_key params must be an object");
        }
        if (!params.contains("key_indices")) {
            return Status::error("vigenere_key params.key_indices is required");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error("vigenere_key params.key_indices must be an array");
        }

        // Explicit-key path: key_indices wins; key_latin may be present as metadata.
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "key_indices" && it.key() != "key_latin") {
                return Status::error("vigenere_key params contains unknown field");
            }
        }

        std::vector<Index29> key;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error("vigenere_key key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error("vigenere_key key_indices entries must be in 0..28");
            }
            key.push_back(Index29{static_cast<std::uint8_t>(raw)});
        }
        if (key.empty()) {
            return Status::error("vigenere_key key_indices must be non-empty");
        }
        return key;
    }

    [[nodiscard]] static Status validate_interrupt_range(
        const InterruptPolicy& interrupt,
        std::size_t input_size) {
        for (std::size_t skip : interrupt.skip_indices()) {
            if (skip >= input_size) {
                return Status::error("vigenere_key skip_indices out of range for input");
            }
        }
        return Status::success();
    }
};

#endif // VIGENERE_KEY_TRANSFORM_HPP
