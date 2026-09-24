#ifndef TOOL_RESPONSE_HPP
#define TOOL_RESPONSE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

/// Stable `error.code` values for `parcae.tool_response.v0` (docs/spec/agent-tools.md).
enum class ToolErrorCode : std::uint8_t {
    Usage = 0,
    Io = 1,
    Schema = 2,
    Policy = 3,
    NotBuilt = 4,
    Validation = 5,
    Internal = 6,
};

class ToolErrorCodeUtil {
public:
    [[nodiscard]] static constexpr std::string_view to_string(ToolErrorCode code) noexcept {
        switch (code) {
        case ToolErrorCode::Usage:
            return "usage";
        case ToolErrorCode::Io:
            return "io";
        case ToolErrorCode::Schema:
            return "schema";
        case ToolErrorCode::Policy:
            return "policy";
        case ToolErrorCode::NotBuilt:
            return "not_built";
        case ToolErrorCode::Validation:
            return "validation";
        case ToolErrorCode::Internal:
            return "internal";
        }
        return "internal";
    }

    [[nodiscard]] static StatusOr<ToolErrorCode> from_string(std::string_view text) {
        if (text == "usage") {
            return ToolErrorCode::Usage;
        }
        if (text == "io") {
            return ToolErrorCode::Io;
        }
        if (text == "schema") {
            return ToolErrorCode::Schema;
        }
        if (text == "policy") {
            return ToolErrorCode::Policy;
        }
        if (text == "not_built") {
            return ToolErrorCode::NotBuilt;
        }
        if (text == "validation") {
            return ToolErrorCode::Validation;
        }
        if (text == "internal") {
            return ToolErrorCode::Internal;
        }
        return Status::error("unknown tool error code");
    }

    /// Agent-mode process exit: 1 = soft (`validation`), 2 = hard (everything else).
    [[nodiscard]] static constexpr int exit_status(ToolErrorCode code) noexcept {
        return code == ToolErrorCode::Validation ? 1 : 2;
    }

private:
    ToolErrorCodeUtil() = delete;
};

/// Builder / validator for the agent JSON envelope `parcae.tool_response.v0`.
class ToolResponse {
public:
    static constexpr std::string_view schema_id = "parcae.tool_response.v0";

    /// Success envelope: `ok=true`, `error=null`, `result` is an object (MAY be empty).
    [[nodiscard]] static nlohmann::json
    success(std::string_view tool, std::optional<std::string> backend, nlohmann::json result) {
        if (!result.is_object()) {
            result = nlohmann::json::object();
        }
        return nlohmann::json{
            {"schema", std::string(schema_id)},
            {"ok", true},
            {"tool", std::string(tool)},
            {"backend", backend.has_value() ? nlohmann::json(*backend) : nlohmann::json(nullptr)},
            {"result", std::move(result)},
            {"error", nullptr},
        };
    }

    /// Failure envelope: `ok=false`, `result=null`, `error={code,message[,details]}`.
    [[nodiscard]] static nlohmann::json failure(std::string_view tool,
                                                std::optional<std::string> backend,
                                                ToolErrorCode code, std::string message,
                                                nlohmann::json details = nlohmann::json(nullptr)) {
        nlohmann::json error{
            {"code", std::string(ToolErrorCodeUtil::to_string(code))},
            {"message", std::move(message)},
        };
        if (!details.is_null()) {
            error["details"] = std::move(details);
        }
        return nlohmann::json{
            {"schema", std::string(schema_id)},
            {"ok", false},
            {"tool", std::string(tool)},
            {"backend", backend.has_value() ? nlohmann::json(*backend) : nlohmann::json(nullptr)},
            {"result", nullptr},
            {"error", std::move(error)},
        };
    }

    [[nodiscard]] static constexpr int exit_success() noexcept { return 0; }

    [[nodiscard]] static constexpr int exit_failure(ToolErrorCode code) noexcept {
        return ToolErrorCodeUtil::exit_status(code);
    }

    static void write(std::ostream& out, const nlohmann::json& envelope) {
        out << envelope.dump() << '\n';
    }

    /// Validate envelope shape; does not deep-validate `result` contents.
    [[nodiscard]] static Status validate(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("tool response must be a JSON object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error("unsupported or missing tool response schema");
        }
        if (!root.contains("ok") || !root.at("ok").is_boolean()) {
            return Status::error("tool response.ok must be a boolean");
        }
        if (!root.contains("tool") || !root.at("tool").is_string() ||
            root.at("tool").get<std::string>().empty()) {
            return Status::error("tool response.tool must be a non-empty string");
        }
        if (!root.contains("backend") ||
            !(root.at("backend").is_null() || root.at("backend").is_string())) {
            return Status::error("tool response.backend must be string or null");
        }
        if (root.at("backend").is_string()) {
            const std::string backend = root.at("backend").get<std::string>();
            if (backend != "cpu" && backend != "cuda") {
                return Status::error("tool response.backend must be cpu, cuda, or null");
            }
        }
        if (!root.contains("result") || !root.contains("error")) {
            return Status::error("tool response must contain result and error");
        }

        const bool ok = root.at("ok").get<bool>();
        if (ok) {
            if (!root.at("result").is_object()) {
                return Status::error("successful tool response.result must be an object");
            }
            if (!root.at("error").is_null()) {
                return Status::error("successful tool response.error must be null");
            }
            return Status::success();
        }

        if (!root.at("result").is_null()) {
            return Status::error("failed tool response.result must be null");
        }
        if (!root.at("error").is_object()) {
            return Status::error("failed tool response.error must be an object");
        }
        const nlohmann::json& err = root.at("error");
        if (!err.contains("code") || !err.at("code").is_string()) {
            return Status::error("tool response.error.code must be a string");
        }
        if (!err.contains("message") || !err.at("message").is_string()) {
            return Status::error("tool response.error.message must be a string");
        }
        if (err.contains("details") && err.at("details").is_null()) {
            return Status::error("tool response.error.details must not be null when present");
        }
        StatusOr<ToolErrorCode> code =
            ToolErrorCodeUtil::from_string(err.at("code").get<std::string>());
        if (!code.ok()) {
            return code.status();
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<nlohmann::json> parse(std::string_view text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid tool response JSON: ") + ex.what());
        }
        Status status = validate(root);
        if (!status.ok()) {
            return status;
        }
        return root;
    }

private:
    ToolResponse() = delete;
};

#endif // TOOL_RESPONSE_HPP
