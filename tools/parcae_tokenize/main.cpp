#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/corpus/token_kind.hpp"
#include "parcae/tool/api.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "tokenize";

void print_help() {
    std::cerr
        << "Usage: parcae-tokenize [--json] [--strict|--no-strict] [--data-dir <path>] <file|->\n"
        << "  Tokenize Liber Primus UTF-8 transcript text.\n"
        << "  --json       Machine-readable JSON on stdout (parcae.tool_response.v0)\n"
        << "  --strict     Reject unknown symbols (default)\n"
        << "  --no-strict  Allow unknown symbols as Unknown tokens\n"
        << "  --data-dir   Parcae data/ root (else PARCAE_DATA_DIR or ./data)\n"
        << "  -h, --help   Show this help\n";
}

[[nodiscard]] int fail(
    bool json_mode,
    ToolErrorCode code,
    std::string message,
    int plain_exit) {
    if (json_mode) {
        return ToolCliJson::err(kTool, std::nullopt, code, std::move(message));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? CliIo::kExitUsage : CliIo::kExitOk;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");
    const bool no_strict = CliIo::has_flag(args, "--no-strict");
    const bool strict = !no_strict;
    const std::string data_dir = CliIo::optional_option(args, "--data-dir");

    std::string input_path;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--json" || args[i] == "--strict" || args[i] == "--no-strict" ||
            args[i] == "-h" || args[i] == "--help") {
            continue;
        }
        if (args[i] == "--data-dir") {
            ++i;
            continue;
        }
        if (!args[i].empty() && args[i][0] == '-') {
            print_help();
            return fail(json_mode, ToolErrorCode::Usage, "Unknown option: " + args[i], CliIo::kExitUsage);
        }
        if (!input_path.empty()) {
            return fail(json_mode, ToolErrorCode::Usage, "Multiple input paths provided", CliIo::kExitUsage);
        }
        input_path = args[i];
    }
    if (input_path.empty()) {
        print_help();
        return fail(
            json_mode, ToolErrorCode::Usage, "Missing input path (file or -)", CliIo::kExitUsage);
    }

    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, ToolErrorCode::Io, ctx.status().message(), CliIo::kExitUsage);
    }

    StatusOr<std::string> source = CliIo::read_all_utf8(input_path);
    if (!source.ok()) {
        return fail(json_mode, ToolErrorCode::Io, source.status().message(), CliIo::kExitUsage);
    }

    StatusOr<TokenStream> stream =
        ToolApi::tokenize(ctx.value(), source.value(), "rtkd-separator-grammar-v0", strict);
    if (!stream.ok()) {
        return fail(json_mode, ToolErrorCode::Internal, stream.status().message(), CliIo::kExitFail);
    }

    if (!json_mode) {
        for (std::size_t i = 0; i < stream.value().size(); ++i) {
            const Token& token = stream.value().at(i);
            std::cout << TokenKindUtil::to_string(token.kind());
            if (token.index29().has_value()) {
                std::cout << '\t' << static_cast<unsigned>(token.index29()->value());
            } else {
                std::cout << "\t-";
            }
            if (token.consumable_index().has_value()) {
                std::cout << '\t' << token.consumable_index().value();
            } else {
                std::cout << "\t-";
            }
            std::cout << '\t' << token.text() << '\n';
        }
        return CliIo::kExitOk;
    }

    nlohmann::json tokens = nlohmann::json::array();
    for (std::size_t i = 0; i < stream.value().size(); ++i) {
        const Token& token = stream.value().at(i);
        nlohmann::json row{
            {"kind", TokenKindUtil::to_string(token.kind())},
            {"text", token.text()},
            {"byte_begin", token.byte_begin()},
            {"byte_end", token.byte_end()},
        };
        if (token.index29().has_value()) {
            row["index29"] = token.index29()->value();
        } else {
            row["index29"] = nullptr;
        }
        if (token.consumable_index().has_value()) {
            row["consumable_index"] = token.consumable_index().value();
        } else {
            row["consumable_index"] = nullptr;
        }
        tokens.push_back(std::move(row));
    }
    return ToolCliJson::ok(
        kTool,
        std::nullopt,
        nlohmann::json{
            {"token_count", stream.value().size()},
            {"consumable_count", stream.value().consumable_count()},
            {"tokens", std::move(tokens)},
        });
}
