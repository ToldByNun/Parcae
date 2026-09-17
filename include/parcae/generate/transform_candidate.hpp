#ifndef TRANSFORM_CANDIDATE_HPP
#define TRANSFORM_CANDIDATE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// One bounded generator hit: stable id + transform envelope + plaintext indices.
class TransformCandidate {
public:
    TransformCandidate(
        std::string candidate_id,
        TransformId transform_id,
        TransformDirection direction,
        nlohmann::json params,
        std::vector<Index29> output_indices)
        : candidate_id_(std::move(candidate_id)),
          transform_id_(std::move(transform_id)),
          direction_(direction),
          params_(std::move(params)),
          output_indices_(std::move(output_indices)) {}

    [[nodiscard]] const std::string& candidate_id() const noexcept {
        return candidate_id_;
    }

    [[nodiscard]] const TransformId& transform_id() const noexcept {
        return transform_id_;
    }

    [[nodiscard]] TransformDirection direction() const noexcept {
        return direction_;
    }

    [[nodiscard]] const nlohmann::json& params() const noexcept {
        return params_;
    }

    [[nodiscard]] const std::vector<Index29>& output_indices() const noexcept {
        return output_indices_;
    }

    /// Spec envelope object (without interrupt — generators default to none).
    [[nodiscard]] nlohmann::json envelope() const {
        return nlohmann::json{
            {"transform_id", transform_id_.str()},
            {"direction", TransformDirectionUtil::to_string(direction_)},
            {"params", params_},
        };
    }

    /// Full output record including `output_indices` as integers 0..28.
    [[nodiscard]] nlohmann::json to_json() const {
        nlohmann::json indices = nlohmann::json::array();
        for (const Index29 idx : output_indices_) {
            indices.push_back(idx.value());
        }
        return nlohmann::json{
            {"candidate_id", candidate_id_},
            {"envelope", envelope()},
            {"output_indices", std::move(indices)},
        };
    }

private:
    std::string candidate_id_;
    TransformId transform_id_;
    TransformDirection direction_;
    nlohmann::json params_;
    std::vector<Index29> output_indices_;
};

#endif // TRANSFORM_CANDIDATE_HPP
