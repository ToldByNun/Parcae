#ifndef IDENTITY_TRANSFORM_HPP
#define IDENTITY_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/transform/transform.hpp"

#include <vector>

/// `out[i] = in[i]`. Direction and interrupt policy are accepted and ignored.
class IdentityTransform : public Transform {
public:
    IdentityTransform() = default;

    [[nodiscard]] TransformId id() const override {
        return TransformId::identity();
    }

    [[nodiscard]] StatusOr<std::vector<Index29>> apply(
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection /*direction*/,
        const InterruptPolicy& /*interrupt*/ = InterruptPolicy::none()) const override {
        Status params_status = validate_params(params);
        if (!params_status.ok()) {
            return params_status;
        }
        return std::vector<Index29>(input.begin(), input.end());
    }

private:
    [[nodiscard]] static Status validate_params(const nlohmann::json& params) {
        if (params.is_null()) {
            return Status::success();
        }
        if (!params.is_object()) {
            return Status::error("identity params must be an object");
        }
        if (!params.empty()) {
            return Status::error("identity params must be empty");
        }
        return Status::success();
    }
};

#endif // IDENTITY_TRANSFORM_HPP
