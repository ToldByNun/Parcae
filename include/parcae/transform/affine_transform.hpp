#ifndef AFFINE_TRANSFORM_HPP
#define AFFINE_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

/// Affine over Z29: encrypt `a·x + b`, decrypt `inv(a)·(x - b)`.
class AffineTransform : public Transform {
public:
    AffineTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::affine(); }

    /// Allocation-free elementwise kernel (in-place OK).
    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       Index29 a, Index29 b, TransformDirection direction) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        const Index29 inv_a = Z29::inv(a);
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (direction == TransformDirection::Encrypt) {
                output[i] = Z29::add(Z29::mul(a, input[i]), b);
            } else {
                output[i] = Z29::mul(inv_a, Z29::sub(input[i], b));
            }
        }
        return Status::success();
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& /*interrupt*/ = InterruptPolicy::none()) const override {
        StatusOr<Params> parsed = parse_params(params);
        if (!parsed.ok()) {
            return parsed.status();
        }
        return kernel(input, output, parsed.value().a, parsed.value().b, direction);
    }

private:
    struct Params {
        Index29 a;
        Index29 b;
    };

    [[nodiscard]] static StatusOr<Index29> parse_index_field(const nlohmann::json& params,
                                                             const char* key,
                                                             std::uint8_t min_inclusive,
                                                             std::uint8_t max_inclusive) {
        if (!params.contains(key)) {
            return Status::error(std::string("affine params.") + key + " is required");
        }
        if (!params.at(key).is_number_integer()) {
            return Status::error(std::string("affine params.") + key + " must be an integer");
        }
        const auto raw = params.at(key).get<std::int64_t>();
        if (raw < static_cast<std::int64_t>(min_inclusive) ||
            raw > static_cast<std::int64_t>(max_inclusive)) {
            return Status::error(std::string("affine params.") + key + " out of valid range");
        }
        return Index29{static_cast<std::uint8_t>(raw)};
    }

    [[nodiscard]] static StatusOr<Params> parse_params(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("affine params must be an object");
        }
        if (params.size() != 2) {
            return Status::error("affine params may only contain a and b");
        }
        StatusOr<Index29> a = parse_index_field(params, "a", 1, 28);
        if (!a.ok()) {
            return a.status();
        }
        StatusOr<Index29> b = parse_index_field(params, "b", 0, 28);
        if (!b.ok()) {
            return b.status();
        }
        return Params{a.value(), b.value()};
    }
};

#endif // AFFINE_TRANSFORM_HPP
