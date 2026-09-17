#ifndef AFFINE_CANDIDATE_GENERATOR_HPP
#define AFFINE_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

/// Bounded generator `gen_affine`: every invertible affine map over Z/29Z.
///
/// **Cost (documented):** `a ∈ 1..28` × `b ∈ 0..28` → **28 × 29 = 812** candidates.
/// Enumeration is nested `(a ascending, then b ascending)` — deterministic, no
/// dictionary / unbounded search.
class AffineCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_affine";

    /// Count of multipliers `a` (all nonzero residues; 29 is prime ⇒ all invertible).
    static constexpr std::size_t a_count = Index29::modulus - 1;  // 28
    /// Count of additives `b`.
    static constexpr std::size_t b_count = Index29::modulus;  // 29
    /// Total envelopes emitted per generate() call.
    static constexpr std::size_t candidate_count = a_count * b_count;  // 812

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> generate(
        std::span<const Index29> ciphertext,
        TransformDirection direction = TransformDirection::Decrypt) {
        const AffineTransform transform;
        std::vector<TransformCandidate> out;
        out.reserve(candidate_count);

        for (std::uint8_t a = 1; a < Index29::modulus; ++a) {
            for (std::uint8_t b = 0; b < Index29::modulus; ++b) {
                const nlohmann::json params = {
                    {"a", static_cast<int>(a)},
                    {"b", static_cast<int>(b)},
                };
                StatusOr<std::vector<Index29>> plain =
                    transform.apply(ciphertext, params, direction, InterruptPolicy::none());
                if (!plain.ok()) {
                    return plain.status();
                }

                out.emplace_back(
                    make_candidate_id(a, b),
                    TransformId::affine(),
                    direction,
                    params,
                    std::move(plain.value()));
            }
        }
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id(std::uint8_t a, std::uint8_t b) {
        return "affine:a=" + std::to_string(static_cast<unsigned>(a)) +
               ",b=" + std::to_string(static_cast<unsigned>(b));
    }
};

#endif // AFFINE_CANDIDATE_GENERATOR_HPP
