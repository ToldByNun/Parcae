#ifndef TOOL_TRANSFORM_ENVELOPE_HPP
#define TOOL_TRANSFORM_ENVELOPE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <utility>

/// Serializable transform call: id + direction + params + optional interrupt.
class TransformEnvelope {
public:
    TransformEnvelope(TransformId transform_id, TransformDirection direction, nlohmann::json params,
                      InterruptPolicy interrupt = InterruptPolicy::none())
        : transform_id_(std::move(transform_id)), direction_(direction), params_(std::move(params)),
          interrupt_(std::move(interrupt)) {}

    [[nodiscard]] static StatusOr<TransformEnvelope> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("TransformEnvelope must be a JSON object");
        }
        if (!root.contains("transform_id") || !root.at("transform_id").is_string()) {
            return Status::error("TransformEnvelope.transform_id is required");
        }
        const std::string id_text = root.at("transform_id").get<std::string>();
        StatusOr<TransformId> id = TransformId::from_string(id_text);
        if (!id.ok()) {
            StatusOr<TheoryUri> theory = TheoryUri::parse(id_text);
            if (!theory.ok()) {
                return id.status();
            }
            id = TransformId::unchecked(theory.value().to_string());
        }

        TransformDirection direction = TransformDirection::Decrypt;
        if (root.contains("direction")) {
            if (!root.at("direction").is_string()) {
                return Status::error("TransformEnvelope.direction must be a string");
            }
            StatusOr<TransformDirection> parsed =
                TransformDirectionUtil::from_string(root.at("direction").get<std::string>());
            if (!parsed.ok()) {
                return parsed.status();
            }
            direction = parsed.value();
        }

        nlohmann::json params = nlohmann::json::object();
        if (root.contains("params")) {
            if (!root.at("params").is_object()) {
                return Status::error("TransformEnvelope.params must be an object");
            }
            params = root.at("params");
        }

        InterruptPolicy interrupt = InterruptPolicy::none();
        if (root.contains("interrupt")) {
            StatusOr<InterruptPolicy> parsed = InterruptPolicy::from_json(root.at("interrupt"));
            if (!parsed.ok()) {
                return parsed.status();
            }
            interrupt = std::move(parsed.value());
        }

        return TransformEnvelope{id.value(), direction, std::move(params), std::move(interrupt)};
    }

    [[nodiscard]] static StatusOr<TransformEnvelope> from_string(const std::string& json_text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(json_text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid TransformEnvelope JSON: ") + ex.what());
        }
        return from_json(root);
    }

    [[nodiscard]] nlohmann::json to_json() const {
        nlohmann::json root{
            {"transform_id", transform_id_.str()},
            {"direction", TransformDirectionUtil::to_string(direction_)},
            {"params", params_},
        };
        if (!interrupt_.skip_indices().empty()) {
            root["interrupt"] = interrupt_.to_json();
        }
        return root;
    }

    [[nodiscard]] const TransformId& transform_id() const noexcept { return transform_id_; }

    [[nodiscard]] TransformDirection direction() const noexcept { return direction_; }

    [[nodiscard]] const nlohmann::json& params() const noexcept { return params_; }

    [[nodiscard]] const InterruptPolicy& interrupt() const noexcept { return interrupt_; }

private:
    TransformId transform_id_;
    TransformDirection direction_;
    nlohmann::json params_;
    InterruptPolicy interrupt_;
};

#endif // TOOL_TRANSFORM_ENVELOPE_HPP
