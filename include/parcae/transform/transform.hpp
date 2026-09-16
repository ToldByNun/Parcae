#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <span>
#include <vector>

#include <nlohmann/json.hpp>

/// Pure Index29 kernel: same length out, deterministic in params + interrupt policy.
class Transform {
public:
    virtual ~Transform() = default;

    [[nodiscard]] virtual TransformId id() const = 0;

    /// Apply the transform to a consumable Index29 span.
    /// Output length MUST equal `input.size()`. Interrupt policy is ignored by
    /// families that do not advance key/stream state (e.g. identity, atbash).
    [[nodiscard]] virtual StatusOr<std::vector<Index29>> apply(
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) const = 0;

protected:
    Transform() = default;
    Transform(const Transform&) = default;
    Transform& operator=(const Transform&) = default;
    Transform(Transform&&) = default;
    Transform& operator=(Transform&&) = default;
};

#endif // TRANSFORM_HPP
