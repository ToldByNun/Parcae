#ifndef BEAUFORT_KEY_TRANSFORM_HPP
#define BEAUFORT_KEY_TRANSFORM_HPP

#include "parcae/core/z29.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/// Explicit-key Beaufort over Z29: `out = sub(key[j], in)` on both directions.
///
/// Classic Beaufort is an involution (apply twice recovers the input). Interrupt
/// positions pass through and do not advance the key cursor — same rule as
/// `vigenere_key`.
///
/// Wiki / community sign pitfalls (see also docs/research/solved-methods.md):
/// - Liber Primus Vigenère decrypt is `p = (c - k) mod 29`, not `(k - c)`.
/// - Swapping the operands yields Beaufort and breaks Welcome / Koan-2 oracles.
/// - Atbash `28 - x` equals Beaufort with the constant key index `28` (last rune).
///   Prefer sharing that identity over treating Atbash as an unrelated map.
class BeaufortKeyTransform : public Transform {
public:
    BeaufortKeyTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::beaufort_key(); }

    /// Allocation-free keyed kernel. In-place OK.
    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       std::span<const Index29> key,
                                       std::span<const std::size_t> skip_indices_sorted) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        if (key.empty()) {
            return Status::error("beaufort_key key must be non-empty");
        }

        const std::size_t key_len = key.size();
        std::size_t key_cursor = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (TransformBuffer::should_skip(skip_indices_sorted, i)) {
                output[i] = input[i];
                continue;
            }
            const Index29 key_symbol = key[key_cursor % key_len];
            output[i] = Z29::sub(key_symbol, input[i]);
            ++key_cursor;
        }
        return Status::success();
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection /*direction*/,
               const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        StatusOr<std::vector<Index29>> key = parse_key_indices(params);
        if (!key.ok()) {
            return key.status();
        }
        Status range =
            TransformBuffer::validate_interrupt_range(interrupt, input.size(), "beaufort_key");
        if (!range.ok()) {
            return range;
        }
        return kernel(input, output, key.value(), TransformBuffer::skip_span(interrupt));
    }

private:
    [[nodiscard]] static StatusOr<std::vector<Index29>>
    parse_key_indices(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("beaufort_key params must be an object");
        }
        if (!params.contains("key_indices")) {
            return Status::error("beaufort_key params.key_indices is required");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error("beaufort_key params.key_indices must be an array");
        }

        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "key_indices" && it.key() != "key_latin") {
                return Status::error("beaufort_key params contains unknown field");
            }
        }

        std::vector<Index29> key;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error("beaufort_key key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error("beaufort_key key_indices entries must be in 0..28");
            }
            key.push_back(Index29{static_cast<std::uint8_t>(raw)});
        }
        if (key.empty()) {
            return Status::error("beaufort_key key_indices must be non-empty");
        }
        return key;
    }
};

#endif // BEAUFORT_KEY_TRANSFORM_HPP
