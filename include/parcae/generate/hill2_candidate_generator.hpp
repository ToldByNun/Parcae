#ifndef HILL2_CANDIDATE_GENERATOR_HPP
#define HILL2_CANDIDATE_GENERATOR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/math/z29_matrix2.hpp"
#include "parcae/transform/hill2_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Bounded generator `gen_hill_2`.
///
/// Does **not** enumerate all \(29^4\) / \(\mathrm{GL}(2,29)\) matrices. Modes:
/// 1. Explicit: `params.matrices` = array of 4-entry row-major matrices (each MUST be
///    invertible). Cost = \|matrices\|.
/// 2. Sample: `params.max_candidates` (default **256**) + `params.seed` (default **1**)
///    — deterministic LCG walk over \(\mathbb{Z}_{29}^{4}\), keep \(\det\not\equiv 0\).
///
/// Empty params → sample mode with the defaults above. Cipher length MUST be even.
class Hill2CandidateGenerator {
public:
    static constexpr std::string_view generator_id = "gen_hill_2";
    static constexpr std::size_t default_max_candidates = 256;
    static constexpr std::uint32_t default_seed = 1;

    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    generate(std::span<const Index29> ciphertext,
             TransformDirection direction = TransformDirection::Decrypt,
             const nlohmann::json& params = nlohmann::json::object()) {
        if (ciphertext.size() % 2u != 0) {
            return Status::error("gen_hill_2 ciphertext length must be even");
        }

        StatusOr<std::vector<std::array<Index29, 4>>> matrices = resolve_matrices(params);
        if (!matrices.ok()) {
            return matrices.status();
        }
        if (matrices.value().empty()) {
            return Status::error("gen_hill_2 produced an empty matrix list");
        }

        const Hill2Transform transform;
        std::vector<TransformCandidate> out;
        out.reserve(matrices.value().size());

        for (std::size_t i = 0; i < matrices.value().size(); ++i) {
            const std::array<Index29, 4>& entries = matrices.value()[i];
            const nlohmann::json matrix_json = {
                static_cast<int>(entries[0].value()), static_cast<int>(entries[1].value()),
                static_cast<int>(entries[2].value()), static_cast<int>(entries[3].value()),
            };
            const nlohmann::json cand_params{{"matrix", matrix_json}};
            StatusOr<std::vector<Index29>> plain =
                transform.apply(ciphertext, cand_params, direction, InterruptPolicy::none());
            if (!plain.ok()) {
                return plain.status();
            }
            out.emplace_back(make_candidate_id(entries, i), TransformId::hill_2(), direction,
                             cand_params, std::move(plain.value()));
        }
        return out;
    }

    [[nodiscard]] static std::string make_candidate_id(const std::array<Index29, 4>& matrix,
                                                       std::size_t list_index) {
        return "hill_2:i=" + std::to_string(list_index) + ":matrix=" +
               std::to_string(static_cast<unsigned>(matrix[0].value())) + "," +
               std::to_string(static_cast<unsigned>(matrix[1].value())) + "," +
               std::to_string(static_cast<unsigned>(matrix[2].value())) + "," +
               std::to_string(static_cast<unsigned>(matrix[3].value()));
    }

    /// Deterministic invertible sample (used by empty / sample-mode params).
    [[nodiscard]] static std::vector<std::array<Index29, 4>>
    sample_invertible(std::size_t max_candidates, std::uint32_t seed) {
        std::vector<std::array<Index29, 4>> out;
        out.reserve(max_candidates);
        std::uint32_t state = seed == 0 ? 0xA5A5A5A5u : seed;
        // Guard: |GL(2,29)| ≫ 256; worst-case skip rate is fine with a generous try budget.
        const std::size_t try_budget = max_candidates * 64u + 1024u;
        for (std::size_t attempt = 0; attempt < try_budget && out.size() < max_candidates;
             ++attempt) {
            std::array<Index29, 4> entries{
                Index29{next_mod29(state)}, Index29{next_mod29(state)}, Index29{next_mod29(state)},
                Index29{next_mod29(state)},
            };
            const Z29Matrix2 m = Z29Matrix2::from_row_major(entries);
            if (m.det().value() == 0) {
                continue;
            }
            out.push_back(entries);
        }
        return out;
    }

private:
    Hill2CandidateGenerator() = delete;

    [[nodiscard]] static std::uint8_t next_mod29(std::uint32_t& state) noexcept {
        state = state * 1664525u + 1013904223u;
        return static_cast<std::uint8_t>((state >> 16) % Index29::modulus);
    }

    [[nodiscard]] static StatusOr<std::vector<std::array<Index29, 4>>>
    resolve_matrices(const nlohmann::json& params) {
        if (params.is_null()) {
            return sample_invertible(default_max_candidates, default_seed);
        }
        if (!params.is_object()) {
            return Status::error("gen_hill_2 params must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "matrices" && it.key() != "max_candidates" && it.key() != "seed") {
                return Status::error("gen_hill_2 params contains unknown field");
            }
        }

        if (params.contains("matrices")) {
            if (params.contains("max_candidates") || params.contains("seed")) {
                return Status::error(
                    "gen_hill_2 matrices cannot be combined with max_candidates/seed");
            }
            return parse_explicit_matrices(params.at("matrices"));
        }

        std::size_t max_candidates = default_max_candidates;
        std::uint32_t seed = default_seed;
        if (params.contains("max_candidates")) {
            if (!params.at("max_candidates").is_number_integer()) {
                return Status::error("gen_hill_2 max_candidates must be an integer");
            }
            const auto raw = params.at("max_candidates").get<std::int64_t>();
            if (raw < 1) {
                return Status::error("gen_hill_2 max_candidates must be >= 1");
            }
            max_candidates = static_cast<std::size_t>(raw);
        }
        if (params.contains("seed")) {
            if (!params.at("seed").is_number_integer()) {
                return Status::error("gen_hill_2 seed must be an integer");
            }
            const auto raw = params.at("seed").get<std::int64_t>();
            if (raw < 0 || raw > static_cast<std::int64_t>(0xFFFFFFFFu)) {
                return Status::error("gen_hill_2 seed must fit in uint32");
            }
            seed = static_cast<std::uint32_t>(raw);
        }
        return sample_invertible(max_candidates, seed);
    }

    [[nodiscard]] static StatusOr<std::vector<std::array<Index29, 4>>>
    parse_explicit_matrices(const nlohmann::json& matrices) {
        if (!matrices.is_array()) {
            return Status::error("gen_hill_2 matrices must be an array");
        }
        if (matrices.empty()) {
            return Status::error("gen_hill_2 matrices must be non-empty");
        }
        std::vector<std::array<Index29, 4>> out;
        out.reserve(matrices.size());
        for (const nlohmann::json& item : matrices) {
            if (!item.is_array() || item.size() != 4) {
                return Status::error("gen_hill_2 matrices entries must be arrays of length 4");
            }
            std::array<Index29, 4> entries{};
            for (std::size_t i = 0; i < 4; ++i) {
                if (!item[i].is_number_integer()) {
                    return Status::error("gen_hill_2 matrix entries must be integers");
                }
                const auto raw = item[i].get<std::int64_t>();
                if (raw < 0 || raw > 28) {
                    return Status::error("gen_hill_2 matrix entries must be in 0..28");
                }
                entries[i] = Index29{static_cast<std::uint8_t>(raw)};
            }
            const Z29Matrix2 m = Z29Matrix2::from_row_major(entries);
            if (m.det().value() == 0) {
                return Status::error("gen_hill_2 matrix must be invertible (det ≢ 0 mod 29)");
            }
            out.push_back(entries);
        }
        return out;
    }
};

#endif // HILL2_CANDIDATE_GENERATOR_HPP
