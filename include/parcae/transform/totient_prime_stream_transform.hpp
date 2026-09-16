#ifndef TOTIENT_PRIME_STREAM_TRANSFORM_HPP
#define TOTIENT_PRIME_STREAM_TRANSFORM_HPP

#include "parcae/core/z29.hpp"
#include "parcae/math/totient_keystream.hpp"
#include "parcae/transform/transform.hpp"

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

    [[nodiscard]] StatusOr<std::vector<Index29>> apply(
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        StatusOr<std::size_t> start = parse_prime_start_index(params);
        if (!start.ok()) {
            return start.status();
        }

        Status range = validate_interrupt_range(interrupt, input.size());
        if (!range.ok()) {
            return range;
        }

        std::size_t consumable = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (!interrupt.should_skip(i)) {
                ++consumable;
            }
        }

        StatusOr<std::vector<Index29>> stream =
            TotientKeystream::shifts(consumable, start.value());
        if (!stream.ok()) {
            return stream.status();
        }

        std::vector<Index29> out;
        out.reserve(input.size());
        std::size_t stream_cursor = 0;

        for (std::size_t i = 0; i < input.size(); ++i) {
            if (interrupt.should_skip(i)) {
                out.push_back(input[i]);
                continue;
            }

            const Index29 shift = stream.value()[stream_cursor++];
            if (direction == TransformDirection::Encrypt) {
                out.push_back(Z29::add(input[i], shift));
            } else {
                out.push_back(Z29::sub(input[i], shift));
            }
        }

        return out;
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

    [[nodiscard]] static Status validate_interrupt_range(
        const InterruptPolicy& interrupt,
        std::size_t input_size) {
        for (std::size_t skip : interrupt.skip_indices()) {
            if (skip >= input_size) {
                return Status::error(
                    "totient_prime_stream skip_indices out of range for input");
            }
        }
        return Status::success();
    }
};

#endif // TOTIENT_PRIME_STREAM_TRANSFORM_HPP
