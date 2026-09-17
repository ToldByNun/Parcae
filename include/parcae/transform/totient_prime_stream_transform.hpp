#ifndef TOTIENT_PRIME_STREAM_TRANSFORM_HPP
#define TOTIENT_PRIME_STREAM_TRANSFORM_HPP

#include "parcae/core/z29.hpp"
#include "parcae/math/totient_keystream.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// Totient / prime−1 stream: decrypt `in - s_j`, encrypt `in + s_j`,
/// with `s_j = (p_j - 1) mod 29`. Interrupt positions pass through and do not
/// advance the prime cursor (`InterruptPolicy`).
class TotientPrimeStreamTransform : public Transform {
public:
    TotientPrimeStreamTransform() = default;

    [[nodiscard]] TransformId id() const override {
        return TransformId::totient_prime_stream();
    }

    /// Allocation-free once `shifts` (consumable length) is provided. In-place OK.
    [[nodiscard]] static Status kernel(
        std::span<const Index29> input,
        std::span<Index29> output,
        std::span<const Index29> shifts,
        std::span<const std::size_t> skip_indices_sorted,
        TransformDirection direction) {
        Status sizes = parcae::transform_buf::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }

        std::size_t stream_cursor = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (parcae::transform_buf::should_skip(skip_indices_sorted, i)) {
                output[i] = input[i];
                continue;
            }
            if (stream_cursor >= shifts.size()) {
                return Status::error("totient_prime_stream shifts shorter than consumable count");
            }
            const Index29 shift = shifts[stream_cursor++];
            if (direction == TransformDirection::Encrypt) {
                output[i] = Z29::add(input[i], shift);
            } else {
                output[i] = Z29::sub(input[i], shift);
            }
        }
        if (stream_cursor != shifts.size()) {
            return Status::error("totient_prime_stream shifts longer than consumable count");
        }
        return Status::success();
    }

    [[nodiscard]] Status apply_into(
        std::span<const Index29> input,
        std::span<Index29> output,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        StatusOr<std::size_t> start = parse_prime_start_index(params);
        if (!start.ok()) {
            return start.status();
        }

        Status range = parcae::transform_buf::validate_interrupt_range(
            interrupt, input.size(), "totient_prime_stream");
        if (!range.ok()) {
            return range;
        }

        std::size_t consumable = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (!interrupt.should_skip(i)) {
                ++consumable;
            }
        }

        // Keystream materialization into a single temporary (not per-element).
        std::vector<Index29> shifts(consumable);
        Status filled = TotientKeystream::shifts_into(shifts, start.value());
        if (!filled.ok()) {
            return filled;
        }

        return kernel(
            input,
            output,
            shifts,
            parcae::transform_buf::skip_span(interrupt),
            direction);
    }

private:
    [[nodiscard]] static StatusOr<std::size_t> parse_prime_start_index(
        const nlohmann::json& params) {
        if (params.is_null()) {
            return std::size_t{0};
        }
        if (!params.is_object()) {
            return Status::error("totient_prime_stream params must be an object");
        }

        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "prime_start_index" && it.key() != "shift_mode") {
                return Status::error("totient_prime_stream params contains unknown field");
            }
        }

        if (params.contains("shift_mode")) {
            if (!params.at("shift_mode").is_string()) {
                return Status::error("totient_prime_stream shift_mode must be a string");
            }
            if (params.at("shift_mode").get<std::string>() != "prime_minus_one_mod_29") {
                return Status::error(
                    "totient_prime_stream shift_mode must be prime_minus_one_mod_29");
            }
        }

        std::size_t prime_start_index = 0;
        if (params.contains("prime_start_index")) {
            if (!params.at("prime_start_index").is_number_integer()) {
                return Status::error(
                    "totient_prime_stream prime_start_index must be an integer");
            }
            const auto raw = params.at("prime_start_index").get<std::int64_t>();
            if (raw < 0) {
                return Status::error(
                    "totient_prime_stream prime_start_index must be non-negative");
            }
            prime_start_index = static_cast<std::size_t>(raw);
        }
        return prime_start_index;
    }
};

#endif // TOTIENT_PRIME_STREAM_TRANSFORM_HPP
