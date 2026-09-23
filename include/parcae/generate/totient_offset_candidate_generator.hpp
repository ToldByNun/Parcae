#ifndef TOTIENT_OFFSET_CANDIDATE_GENERATOR_HPP
#define TOTIENT_OFFSET_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/totient_prime_stream_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Bounded generator `gen_totient_offsets`.
///
/// Enumerates an explicit list of `prime_start_index` values (cost = |starts|).
/// Default search grids use a small contiguous range (see
/// `GpuCandidateExport::default_totient_start_count`). Opt-in family `totient`
/// requires `SearchJob.allow_extended_families`.
class TotientOffsetCandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_totient_offsets";

    /// Apply totient_prime_stream for each start index. Empty `starts` is an error.
    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>> generate(
        std::span<const Index29> ciphertext,
        const std::vector<std::size_t>& prime_start_indices,
        TransformDirection direction = TransformDirection::Decrypt,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (prime_start_indices.empty()) {
            return Status::error(
                "gen_totient_offsets requires a non-empty prime_start_indices list");
        }

        const TotientPrimeStreamTransform transform;
        std::vector<TransformCandidate> out;
        out.reserve(prime_start_indices.size());

        std::optional<nlohmann::json> interrupt_json;
        if (!interrupt.skip_indices().empty()) {
            interrupt_json = interrupt.to_json();
        }

        for (std::size_t i = 0; i < prime_start_indices.size(); ++i) {
            const std::size_t start = prime_start_indices[i];
            const nlohmann::json params{
                {"prime_start_index", static_cast<std::uint64_t>(start)},
                {"shift_mode", "prime_minus_one_mod_29"},
            };
            StatusOr<std::vector<Index29>> plain =
                transform.apply(ciphertext, params, direction, interrupt);
            if (!plain.ok()) {
                return plain.status();
            }
            out.emplace_back(
                make_candidate_id(start),
                TransformId::totient_prime_stream(),
                direction,
                params,
                std::move(plain.value()),
                interrupt_json);
        }
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id(std::size_t prime_start_index) {
        return "totient_prime_stream:prime_start_index=" +
               std::to_string(prime_start_index);
    }
};

#endif  // TOTIENT_OFFSET_CANDIDATE_GENERATOR_HPP
