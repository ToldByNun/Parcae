#ifndef ATBASH_CANDIDATE_GENERATOR_HPP
#define ATBASH_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// Bounded generator `gen_atbash`: single involution candidate.
class AtbashCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_atbash";
    static constexpr std::size_t candidate_count = 1;

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext,
             TransformDirection direction = TransformDirection::Decrypt) {
        const AtbashTransform transform;
        const nlohmann::json params = nlohmann::json::object();
        StatusOr<std::vector<Index29>> plain =
            transform.apply(ciphertext, params, direction, InterruptPolicy::none());
        if (!plain.ok()) {
            return plain.status();
        }

        std::vector<TransformCandidate> out;
        out.reserve(1);
        out.emplace_back(make_candidate_id(), TransformId::atbash(), direction, params,
                         std::move(plain.value()));
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id() { return "atbash"; }
};

#endif // ATBASH_CANDIDATE_GENERATOR_HPP
