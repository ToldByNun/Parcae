#ifndef ATBASH_CAESAR_CANDIDATE_GENERATOR_HPP
#define ATBASH_CAESAR_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// Bounded generator `gen_atbash_caesar`: Atbash ∘ Caesar(+shift) for every
/// shift in `0..28` (Koan-1 family; Caesar stage uses encrypt-on-decrypt).
/// Enumeration order is ascending shift — 29 candidates.
class AtbashCaesarCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_atbash_caesar";
    static constexpr std::size_t candidate_count = Index29::modulus; // 29

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext,
             TransformDirection direction = TransformDirection::Decrypt) {
        std::vector<TransformCandidate> out;
        out.reserve(candidate_count);

        for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
            const nlohmann::json params = ComposeTransform::atbash_then_caesar_params(shift);
            StatusOr<std::vector<Index29>> plain =
                ComposeTransform::apply_atbash_then_caesar(ciphertext, shift, direction);
            if (!plain.ok()) {
                return plain.status();
            }

            out.emplace_back(make_candidate_id(shift), TransformId::compose(), direction, params,
                             std::move(plain.value()));
        }
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id(std::uint8_t shift) {
        return "atbash_caesar:shift=" + std::to_string(static_cast<unsigned>(shift));
    }
};

#endif // ATBASH_CAESAR_CANDIDATE_GENERATOR_HPP
