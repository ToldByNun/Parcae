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

#include <nlohmann/json.hpp>
#include <span>
#include <vector>

/// Dispatch a catalog transform id onto its concrete applicator.
/// Theory URIs (`parcae://theories/…`) are not catalog ids — use
/// `TheoryDispatch` / `ToolApi::apply_to_indices` with a
/// `TheoryEnvelopeBridge::Envelope` instead.
class ApplyTransform {
public:
    [[nodiscard]] static Status
    apply_into(const TransformId& id, std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (id == TransformId::identity()) {
            return IdentityTransform{}.apply_into(input, output, params, direction, interrupt);
        }
        if (id == TransformId::atbash()) {
            return AtbashTransform{}.apply_into(input, output, params, direction, interrupt);
        }
        if (id == TransformId::caesar()) {
            return CaesarTransform{}.apply_into(input, output, params, direction, interrupt);
        }
        if (id == TransformId::affine()) {
            return AffineTransform{}.apply_into(input, output, params, direction, interrupt);
        }
        if (id == TransformId::vigenere_key()) {
            return VigenereKeyTransform{}.apply_into(input, output, params, direction, interrupt);
        }
        if (id == TransformId::beaufort_key()) {
            return BeaufortKeyTransform{}.apply_into(input, output, params, direction, interrupt);
        }
        if (id == TransformId::totient_prime_stream()) {
            return TotientPrimeStreamTransform{}.apply_into(input, output, params, direction,
                                                            interrupt);
        }
        if (id == TransformId::compose()) {
            return ComposeTransform{}.apply_into(input, output, params, direction, interrupt);
        }
        return Status::error("transform_id is not supported by ApplyTransform");
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>>
    apply(const TransformId& id, std::span<const Index29> input, const nlohmann::json& params,
          TransformDirection direction,
          const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        std::vector<Index29> out(input.size());
        Status status = apply_into(id, input, out, params, direction, interrupt);
        if (!status.ok()) {
            return status;
        }
        return out;
    }

private:
    ApplyTransform() = delete;
};

#endif // APPLY_TRANSFORM_HPP
