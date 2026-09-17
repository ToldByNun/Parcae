#include "cli_io.hpp"

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

void print_help() {
    std::cerr
        << "Usage: parcae-tokenize [--json] [--strict|--no-strict] [--data-dir <path>] <file|->\n"
        << "  Tokenize Liber Primus UTF-8 transcript text.\n"
        << "  --json       Machine-readable JSON on stdout\n"
        << "  --strict     Reject unknown symbols (default)\n"
        << "  --no-strict  Allow unknown symbols as Unknown tokens\n"
        << "  --data-dir   Parcae data/ root (else PARCAE_DATA_DIR or ./data)\n"
        << "  -h, --help   Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? kExitUsage : kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const bool no_strict = has_flag(args, "--no-strict");
    const bool strict = !no_strict;
    const std::string data_dir = optional_option(args, "--data-dir");

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
            std::cerr << "Unknown option: " << args[i] << '\n';
            print_help();
            return kExitUsage;
        }
        if (!input_path.empty()) {
            std::cerr << "Multiple input paths provided\n";
            return kExitUsage;
        }
        input_path = args[i];
    }
    if (input_path.empty()) {
        std::cerr << "Missing input path (file or -)\n";
        print_help();
        return kExitUsage;
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return kExitUsage;
    }

    StatusOr<std::string> source = read_all_utf8(input_path);
    if (!source.ok()) {
        std::cerr << source.status().message() << '\n';
        return kExitUsage;
    }

    StatusOr<TokenStream> stream =
        parcae::tool::tokenize(ctx.value(), source.value(), "rtkd-separator-grammar-v0", strict);
    if (!stream.ok()) {
        std::cerr << stream.status().message() << '\n';
        return kExitFail;
    }

    if (json_mode) {
        nlohmann::json tokens = nlohmann::json::array();
        for (std::size_t i = 0; i < stream.value().size(); ++i) {
            const Token& token = stream.value().at(i);
            nlohmann::json row{
                {"kind", TokenKindUtil::to_string(token.kind())},
                {"text", token.text()},
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
        std::cout << nlohmann::json{{"tokens", std::move(tokens)}}.dump(2) << '\n';
    } else {
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
    }

    return kExitOk;
}
