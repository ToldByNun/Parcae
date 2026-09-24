#ifndef PARITY_RECORD_HPP
#define PARITY_RECORD_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/apply_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// CPU↔CUDA parity transcript (`parcae.parity_record.v0`).
///
/// Digests:
/// - `params_hash_sha256` — SHA-256 of compact `params.dump()` (no whitespace)
/// - `input_sha256` / `output_sha256` — SHA-256 of raw `uint8_t` Index29 values
/// - `interrupt_sha256` — SHA-256 of compact `interrupt.to_json().dump()`
class ParityRecord {
public:
    static constexpr std::string_view schema_id = "parcae.parity_record.v0";

    ParityRecord() = default;

    [[nodiscard]] const std::string& transform_id() const noexcept { return transform_id_; }

    [[nodiscard]] const std::string& params_hash_sha256() const noexcept {
        return params_hash_sha256_;
    }

    [[nodiscard]] const std::string& input_sha256() const noexcept { return input_sha256_; }

    [[nodiscard]] const std::string& output_sha256() const noexcept { return output_sha256_; }

    [[nodiscard]] const std::string& interrupt_sha256() const noexcept { return interrupt_sha256_; }

    [[nodiscard]] const std::string& backend() const noexcept { return backend_; }

    [[nodiscard]] nlohmann::json to_json() const {
        return nlohmann::json{
            {"parity_schema", std::string(schema_id)},
            {"transform_id", transform_id_},
            {"params_hash_sha256", params_hash_sha256_},
            {"input_sha256", input_sha256_},
            {"output_sha256", output_sha256_},
            {"interrupt_sha256", interrupt_sha256_},
            {"backend", backend_},
        };
    }

    [[nodiscard]] static StatusOr<ParityRecord> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("parity record must be a JSON object");
        }
        if (!root.contains("parity_schema") ||
            root.at("parity_schema").get<std::string>() != schema_id) {
            return Status::error("unsupported or missing parity_schema");
        }
        for (const char* key : {"transform_id", "params_hash_sha256", "input_sha256",
                                "output_sha256", "interrupt_sha256", "backend"}) {
            if (!root.contains(key) || !root.at(key).is_string()) {
                return Status::error(std::string("parity record missing string field: ") + key);
            }
        }
        ParityRecord record;
        record.transform_id_ = root.at("transform_id").get<std::string>();
        record.params_hash_sha256_ = root.at("params_hash_sha256").get<std::string>();
        record.input_sha256_ = root.at("input_sha256").get<std::string>();
        record.output_sha256_ = root.at("output_sha256").get<std::string>();
        record.interrupt_sha256_ = root.at("interrupt_sha256").get<std::string>();
        record.backend_ = root.at("backend").get<std::string>();
        return record;
    }

    /// Build a record from already-computed buffers (no transform apply).
    [[nodiscard]] static ParityRecord
    capture(std::string_view transform_id, const nlohmann::json& params,
            const InterruptPolicy& interrupt, std::span<const Index29> input,
            std::span<const Index29> output, std::string_view backend = "cpu") {
        ParityRecord record;
        record.transform_id_ = std::string(transform_id);
        record.params_hash_sha256_ = hash_json(params);
        record.input_sha256_ = hash_indices(input);
        record.output_sha256_ = hash_indices(output);
        record.interrupt_sha256_ = hash_json(interrupt.to_json());
        record.backend_ = std::string(backend);
        return record;
    }

    /// Apply a catalog transform, then capture the parity transcript.
    [[nodiscard]] static StatusOr<std::pair<std::vector<Index29>, ParityRecord>>
    apply_and_capture(const TransformId& id, std::span<const Index29> input,
                      const nlohmann::json& params, TransformDirection direction,
                      const InterruptPolicy& interrupt = InterruptPolicy::none(),
                      std::string_view backend = "cpu") {
        StatusOr<std::vector<Index29>> output =
            ApplyTransform::apply(id, input, params, direction, interrupt);
        if (!output.ok()) {
            return output.status();
        }
        ParityRecord record = capture(id.str(), params, interrupt, input, output.value(), backend);
        return std::make_pair(std::move(output.value()), std::move(record));
    }

    [[nodiscard]] static std::string hash_json(const nlohmann::json& value) {
        return Sha256::hex_digest(value.dump());
    }

    [[nodiscard]] static std::string hash_indices(std::span<const Index29> indices) {
        std::vector<std::uint8_t> bytes(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            bytes[i] = indices[i].value();
        }
        return Sha256::hex_digest(bytes.data(), bytes.size());
    }

    [[nodiscard]] bool operator==(const ParityRecord&) const noexcept = default;

private:
    std::string transform_id_;
    std::string params_hash_sha256_;
    std::string input_sha256_;
    std::string output_sha256_;
    std::string interrupt_sha256_;
    std::string backend_{"cpu"};
};

#endif // PARITY_RECORD_HPP
