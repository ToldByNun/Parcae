#ifndef ATBASH_TRANSFORM_HPP
#define ATBASH_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

/// `out[i] = 28 - in[i]`. Direction ignored (involution); interrupt unused.
class AtbashTransform : public Transform {
public:
    AtbashTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::atbash(); }

    /// Allocation-free elementwise kernel (in-place OK).
    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = Z29::atbash(input[i]);
        }
        return Status::success();
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection /*direction*/,
               const InterruptPolicy& /*interrupt*/ = InterruptPolicy::none()) const override {
        Status params_status = validate_params(params);
        if (!params_status.ok()) {
            return params_status;
        }
        return kernel(input, output);
    }

private:
    [[nodiscard]] static Status validate_params(const nlohmann::json& params) {
        if (params.is_null()) {
            return Status::success();
        }
        if (!params.is_object()) {
            return Status::error("atbash params must be an object");
        }
        if (!params.empty()) {
            return Status::error("atbash params must be empty");
        }
        return Status::success();
    }
};

#endif // ATBASH_TRANSFORM_HPP
