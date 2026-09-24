#ifndef CAESAR_TRANSFORM_HPP
#define CAESAR_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>

/// Caesar over Z29: encrypt adds `shift`, decrypt subtracts `shift`.
class CaesarTransform : public Transform {
public:
    CaesarTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::caesar(); }

    /// Allocation-free elementwise kernel (in-place OK).
    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       Index29 shift, TransformDirection direction) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (direction == TransformDirection::Encrypt) {
                output[i] = Z29::add(input[i], shift);
            } else {
                output[i] = Z29::sub(input[i], shift);
            }
        }
        return Status::success();
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& /*interrupt*/ = InterruptPolicy::none()) const override {
        StatusOr<Index29> shift = parse_shift(params);
        if (!shift.ok()) {
            return shift.status();
        }
        return kernel(input, output, shift.value(), direction);
    }

private:
    [[nodiscard]] static StatusOr<Index29> parse_shift(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("caesar params must be an object");
        }
        if (!params.contains("shift")) {
            return Status::error("caesar params.shift is required");
        }
        if (!params.at("shift").is_number_integer()) {
            return Status::error("caesar params.shift must be an integer");
        }
        const auto raw = params.at("shift").get<std::int64_t>();
        if (raw < 0 || raw > 28) {
            return Status::error("caesar params.shift must be in 0..28");
        }
        if (params.size() != 1) {
            return Status::error("caesar params may only contain shift");
        }
        return Index29{static_cast<std::uint8_t>(raw)};
    }
};

#endif // CAESAR_TRANSFORM_HPP
