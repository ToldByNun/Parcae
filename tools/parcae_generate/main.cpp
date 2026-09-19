#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/generate/generator_registry.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/tool/generate_candidates.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "generate";

void print_help() {
    std::cerr
        << "Usage: parcae-generate --generator-id <id> --input <file|->\n"
        << "                       [--latin|--runes|--indices] [--direction decrypt|encrypt]\n"
        << "                       [--params-json <json>] [--json] [--data-dir <path>]\n"
        << "       parcae-generate --list [--json] [--data-dir <path>]\n"
        << "\n"
        << "Emit TransformCandidate JSON from a bounded gen_* generator.\n"
        << "\n"
        << "Input modes (exactly one; default --runes):\n"
        << "  --latin     Preferred/alias Latin letters (non-letters stripped)\n"
        << "  --runes     Tokenize Liber Primus UTF-8; use consumable runes\n"
        << "  --indices   Comma/whitespace-separated Index29 values 0..28\n"
        << "\n"
        << "  --generator-id   e.g. gen_caesar, gen_atbash, gen_affine\n"
        << "  --direction      decrypt|encrypt (default decrypt)\n"
        << "  --params-json    Generator params (required for gen_vigenere_explicit_keys)\n"
        << "  --list           Print known generator ids and exit\n"
        << "  --json           JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir       Parcae data/ root\n"
        << "  -h, --help       Show this help\n";
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
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? kExitUsage : kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const std::string data_dir = optional_option(args, "--data-dir");

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, ToolErrorCode::Io, ctx.status().message(), kExitUsage);
    }

    if (has_flag(args, "--list")) {
        const std::vector<std::string> ids = GenerateCandidates::list_generator_ids();
        if (!json_mode) {
            for (const std::string& id : ids) {
                std::cout << id << '\n';
            }
            return kExitOk;
        }
        return ToolCliJson::ok(
            kTool,
            std::nullopt,
            nlohmann::json{{"generator_ids", ids}});
    }

    StatusOr<std::string> generator_id = require_option(args, "--generator-id");
    if (!generator_id.ok()) {
        print_help();
        return fail(json_mode, ToolErrorCode::Usage, generator_id.status().message(), kExitUsage);
    }
    if (!GeneratorRegistry::is_known(generator_id.value())) {
        return fail(
            json_mode,
            ToolErrorCode::Usage,
            "Unknown generator_id: " + generator_id.value(),
            kExitUsage);
    }

    StatusOr<std::string> input_path = require_option(args, "--input");
    if (!input_path.ok()) {
        print_help();
        return fail(json_mode, ToolErrorCode::Usage, input_path.status().message(), kExitUsage);
    }

    const bool mode_latin = has_flag(args, "--latin");
    const bool mode_runes = has_flag(args, "--runes");
    const bool mode_indices = has_flag(args, "--indices");
    const int modes =
        static_cast<int>(mode_latin) + static_cast<int>(mode_runes) +
        static_cast<int>(mode_indices);
    if (modes > 1) {
        return fail(
            json_mode,
            ToolErrorCode::Usage,
            "Choose at most one of --latin, --runes, --indices",
            kExitUsage);
    }
    const std::string_view input_mode =
        mode_latin ? "latin" : (mode_indices ? "indices" : "runes");

    const std::string direction_text = optional_option(args, "--direction", "decrypt");
    StatusOr<TransformDirection> direction =
        TransformDirectionUtil::from_string(direction_text);
    if (!direction.ok()) {
        return fail(json_mode, ToolErrorCode::Usage, direction.status().message(), kExitUsage);
    }

    nlohmann::json params = nlohmann::json::object();
    const std::string params_json = optional_option(args, "--params-json");
    if (!params_json.empty()) {
        try {
            params = nlohmann::json::parse(params_json);
        } catch (const nlohmann::json::exception& ex) {
            return fail(
                json_mode,
                ToolErrorCode::Schema,
                std::string("Invalid --params-json: ") + ex.what(),
                kExitUsage);
        }
        if (!params.is_object()) {
            return fail(
                json_mode,
                ToolErrorCode::Schema,
                "--params-json must be an object",
                kExitUsage);
        }
    }

    StatusOr<std::string> source = read_all_utf8(input_path.value());
    if (!source.ok()) {
        return fail(json_mode, ToolErrorCode::Io, source.status().message(), kExitUsage);
    }

    StatusOr<std::vector<TransformCandidate>> candidates = GenerateCandidates::from_source(
        ctx.value(),
        generator_id.value(),
        source.value(),
        input_mode,
        direction.value(),
        params);
    if (!candidates.ok()) {
        return fail(
            json_mode,
            ToolErrorCode::Internal,
            candidates.status().message(),
            kExitFail);
    }

    if (!json_mode) {
        for (const TransformCandidate& c : candidates.value()) {
            std::cout << c.candidate_id() << '\t' << c.transform_id().str() << '\t'
                      << c.params().dump() << '\n';
        }
        std::cout << "count\t" << candidates.value().size() << '\n';
        return kExitOk;
    }

    nlohmann::json rows = nlohmann::json::array();
    for (const TransformCandidate& c : candidates.value()) {
        rows.push_back(c.to_json());
    }
    return ToolCliJson::ok(
        kTool,
        std::nullopt,
        nlohmann::json{
            {"generator_id", generator_id.value()},
            {"direction", TransformDirectionUtil::to_string(direction.value())},
            {"input_mode", std::string(input_mode)},
            {"count", candidates.value().size()},
            {"candidates", std::move(rows)},
        });
}
