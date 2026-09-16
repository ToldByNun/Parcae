#ifndef ATBASH_TRANSFORM_HPP
#define ATBASH_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/transform/transform.hpp"

#include <vector>

/// `out[i] = 28 - in[i]`. Direction ignored (involution); interrupt unused.
class AtbashTransform : public Transform {
public:
    AtbashTransform() = default;

    [[nodiscard]] TransformId id() const override {
        return TransformId::atbash();
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

        std::vector<Index29> out;
        out.reserve(input.size());
        for (Index29 value : input) {
            out.push_back(Z29::atbash(value));
        }
        return out;
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
