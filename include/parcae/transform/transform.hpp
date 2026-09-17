#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <span>
#include <vector>

#include <nlohmann/json.hpp>

/// Pure Index29 kernel: same length out, deterministic in params + interrupt policy.
///
/// Hot path: `apply_into` writes into a caller-provided `output` span (no output
/// allocation). Convenience `apply` allocates once then delegates to `apply_into`.
/// JSON param parsing / keystream materialization may allocate; elementwise kernels
/// themselves do not.
class Transform {
public:
    virtual ~Transform() = default;

    [[nodiscard]] virtual TransformId id() const = 0;

    /// CUDA-ready signature: write `output[i]` for each `input[i]`.
    /// `output.size()` MUST equal `input.size()`. In-place (`output.data() ==
    /// input.data()`) is allowed for single-pass families.
    [[nodiscard]] virtual Status apply_into(
        std::span<const Index29> input,
        std::span<Index29> output,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) const = 0;

    /// Allocating convenience wrapper over `apply_into`.
    [[nodiscard]] StatusOr<std::vector<Index29>> apply(
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) const {
        std::vector<Index29> out(input.size());
        Status status = apply_into(input, out, params, direction, interrupt);
        if (!status.ok()) {
            return status;
        }
        return out;
    }

protected:
    Transform() = default;
    Transform(const Transform&) = default;
    Transform& operator=(const Transform&) = default;
    Transform(Transform&&) = default;
    Transform& operator=(Transform&&) = default;
};

#endif // TRANSFORM_HPP
