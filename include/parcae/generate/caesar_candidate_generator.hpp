#ifndef CAESAR_CANDIDATE_GENERATOR_HPP
#define CAESAR_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// Bounded generator `gen_caesar`: decrypt/encrypt with every shift in `0..28`.
/// Enumeration order is ascending shift (deterministic).
class CaesarCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_caesar";
    static constexpr std::size_t candidate_count = Index29::modulus; // 29

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext,
             TransformDirection direction = TransformDirection::Decrypt) {
        const CaesarTransform transform;
        std::vector<TransformCandidate> out;
        out.reserve(candidate_count);

        for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
            const nlohmann::json params = {{"shift", static_cast<int>(shift)}};
            StatusOr<std::vector<Index29>> plain =
                transform.apply(ciphertext, params, direction, InterruptPolicy::none());
            if (!plain.ok()) {
                return plain.status();
            }

            out.emplace_back(make_candidate_id(shift), TransformId::caesar(), direction, params,
                             std::move(plain.value()));
        }
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id(std::uint8_t shift) {
        return "caesar:shift=" + std::to_string(static_cast<unsigned>(shift));
    }
};

#endif // CAESAR_CANDIDATE_GENERATOR_HPP
