#ifndef PLAINTEXT_AUTOKEY_TRANSFORM_HPP
#define PLAINTEXT_AUTOKEY_TRANSFORM_HPP

#include "parcae/core/z29.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/// Plaintext autokey (PTAK) over \(\mathbb{Z}_{29}\).
/// Primer `key` of length \(L\); after the primer, the keystream is prior plaintext.
///
/// Dense (no interrupts):
/// - encrypt key at \(i\): \(i < L\) → `key[i]`, else `input[i-L]` (plaintext)
/// - decrypt key at \(i\): \(i < L\) → `key[i]`, else `output[i-L]` (recovered plaintext)
/// - encrypt `out = in + key`, decrypt `out = in - key`
///
/// With interrupts: skip positions pass through and do not consume primer/feedback
/// (Vigenère-style cursor over consumed runes only).
class PlaintextAutokeyTransform : public Transform {
public:
    PlaintextAutokeyTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::plaintext_autokey(); }

    /// `key` MUST be non-empty; `skip_indices_sorted` MUST be sorted unique (caller-validated).
    /// In-place OK.
    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       std::span<const Index29> key,
                                       std::span<const std::size_t> skip_indices_sorted,
                                       TransformDirection direction) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        if (key.empty()) {
            return Status::error("plaintext_autokey key must be non-empty");
        }

        const std::size_t primer_len = key.size();

        if (skip_indices_sorted.empty()) {
            for (std::size_t i = 0; i < input.size(); ++i) {
                Index29 key_symbol;
                if (i < primer_len) {
                    key_symbol = key[i];
                } else if (direction == TransformDirection::Encrypt) {
                    key_symbol = input[i - primer_len];
                } else {
                    key_symbol = output[i - primer_len];
                }
                if (direction == TransformDirection::Encrypt) {
                    output[i] = Z29::add(input[i], key_symbol);
                } else {
                    output[i] = Z29::sub(input[i], key_symbol);
                }
            }
            return Status::success();
        }

        std::vector<Index29> feedback;
        feedback.reserve(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (TransformBuffer::should_skip(skip_indices_sorted, i)) {
                output[i] = input[i];
                continue;
            }
            Index29 key_symbol;
            if (feedback.size() < primer_len) {
                key_symbol = key[feedback.size()];
            } else {
                key_symbol = feedback[feedback.size() - primer_len];
            }
            if (direction == TransformDirection::Encrypt) {
                output[i] = Z29::add(input[i], key_symbol);
                feedback.push_back(input[i]);
            } else {
                output[i] = Z29::sub(input[i], key_symbol);
                feedback.push_back(output[i]);
            }
        }
        return Status::success();
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        StatusOr<std::vector<Index29>> key = parse_key_indices(params);
        if (!key.ok()) {
            return key.status();
        }
        Status range = TransformBuffer::validate_interrupt_range(interrupt, input.size(),
                                                                 "plaintext_autokey");
        if (!range.ok()) {
            return range;
        }
        return kernel(input, output, key.value(), TransformBuffer::skip_span(interrupt), direction);
    }

private:
    [[nodiscard]] static StatusOr<std::vector<Index29>>
    parse_key_indices(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("plaintext_autokey params must be an object");
        }
        if (!params.contains("key_indices")) {
            return Status::error("plaintext_autokey params.key_indices is required");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error("plaintext_autokey params.key_indices must be an array");
        }

        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "key_indices" && it.key() != "key_latin") {
                return Status::error("plaintext_autokey params contains unknown field");
            }
        }

        std::vector<Index29> key;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error("plaintext_autokey key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error("plaintext_autokey key_indices entries must be in 0..28");
            }
            key.push_back(Index29{static_cast<std::uint8_t>(raw)});
        }
        if (key.empty()) {
            return Status::error("plaintext_autokey key_indices must be non-empty");
        }
        return key;
    }
};

#endif // PLAINTEXT_AUTOKEY_TRANSFORM_HPP
