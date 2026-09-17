#ifndef APPLY_TRANSFORM_HPP
#define APPLY_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/identity_transform.hpp"
#include "parcae/transform/totient_prime_stream_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include <span>
#include <vector>

#include <nlohmann/json.hpp>

/// Dispatch a catalog transform id onto its concrete applicator.
class ApplyTransform {
public:
    [[nodiscard]] static StatusOr<std::vector<Index29>> apply(
        const TransformId& id,
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (id == TransformId::identity()) {
            return IdentityTransform{}.apply(input, params, direction, interrupt);
        }
        if (id == TransformId::atbash()) {
            return AtbashTransform{}.apply(input, params, direction, interrupt);
        }
        if (id == TransformId::caesar()) {
            return CaesarTransform{}.apply(input, params, direction, interrupt);
        }
        if (id == TransformId::affine()) {
            return AffineTransform{}.apply(input, params, direction, interrupt);
        }
        if (id == TransformId::vigenere_key()) {
            return VigenereKeyTransform{}.apply(input, params, direction, interrupt);
        }
        if (id == TransformId::beaufort_key()) {
            return BeaufortKeyTransform{}.apply(input, params, direction, interrupt);
        }
        if (id == TransformId::totient_prime_stream()) {
            return TotientPrimeStreamTransform{}.apply(input, params, direction, interrupt);
        }
        if (id == TransformId::compose()) {
            return ComposeTransform{}.apply(input, params, direction, interrupt);
        }
        return Status::error("transform_id is not supported by ApplyTransform");
    }

private:
    ApplyTransform() = delete;
};

#endif // APPLY_TRANSFORM_HPP
