#ifndef TOOL_CLI_JSON_HPP
#define TOOL_CLI_JSON_HPP

#include "parcae/tool/tool_response.hpp"

#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

/// Emit `parcae.tool_response.v0` on stdout for agent `--json` CLI mode.
class ToolCliJson {
public:
    [[nodiscard]] static int ok(std::string_view tool, std::optional<std::string> backend,
                                nlohmann::json result) {
        ToolResponse::write(std::cout,
                            ToolResponse::success(tool, std::move(backend), std::move(result)));
        return ToolResponse::exit_success();
    }

    [[nodiscard]] static int err(std::string_view tool, std::optional<std::string> backend,
                                 ToolErrorCode code, std::string message,
                                 nlohmann::json details = nlohmann::json(nullptr)) {
        std::cerr << message << '\n';
        ToolResponse::write(std::cout,
                            ToolResponse::failure(tool, std::move(backend), code,
                                                  std::move(message), std::move(details)));
        return ToolResponse::exit_failure(code);
    }

private:
    ToolCliJson() = delete;
};

#endif // TOOL_CLI_JSON_HPP
