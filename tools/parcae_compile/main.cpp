#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/core/version.hpp"
#include "parcae/dsl/dsl_ast_json_version.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"

#include <filesystem>
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

constexpr std::string_view kTool = "compile";

void print_help() {
    std::cerr
        << "Usage: parcae-compile --status [--json] [--data-dir <path>]\n"
        << "       parcae-compile <theory.py> [--json] [--data-dir <path>]\n"
        << "\n"
        << "Compile a theory DSL source into a versioned artifact under data/theories/.\n"
        << "Normative: docs/spec/dsl.md, docs/spec/theory-artifact.md\n"
        << "\n"
        << "  --status       Report toolchain / DSL versions (stub-safe; no compile)\n"
        << "  --json         JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir     Parcae data/ root (theories land under <data>/theories/)\n"
        << "  -h, --help     Show this help\n"
        << "\n"
        << "Stub note: full AST→IR→verify→emit is not wired yet. Compiling a .py file\n"
        << "returns error code not_built until later commits land the pipeline.\n";
}

[[nodiscard]] int fail(
    bool json_mode,
    ToolErrorCode code,
    std::string message,
    int plain_exit,
    nlohmann::json details = nlohmann::json(nullptr)) {
    if (json_mode) {
        return ToolCliJson::err(kTool, std::nullopt, code, std::move(message), std::move(details));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

[[nodiscard]] nlohmann::json status_result(const std::filesystem::path& data_root) {
    return nlohmann::json{
        {"stub", true},
        {"pipeline_ready", false},
        {"toolkit_version", PARCAE_VERSION_STRING},
        {"dsl_spec_version", std::string(DslSpecVersion::current_string)},
        {"dsl_ast_json_version", std::string(DslAstJsonVersion::current_string)},
        {"dsl_ast_json_schema", std::string(DslAstJsonVersion::schema_id)},
        {"theory_artifact_schema", "parcae.theory_artifact.v0"},
        {"data_dir", data_root.string()},
        {"theories_dir", (data_root / "theories").string()},
        {"message",
         "parcae-compile stub: use --status for versions; compile pipeline lands in later commits"},
    };
}

[[nodiscard]] std::vector<std::string> positional_args(const std::vector<std::string>& args) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--json" || a == "--status" || a == "-h" || a == "--help") {
            continue;
        }
        if (a == "--data-dir") {
            if (i + 1 < args.size()) {
                ++i;
            }
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            continue;
        }
        out.push_back(a);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help")) {
        print_help();
        return kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const bool want_status = has_flag(args, "--status");
    const std::string data_dir = optional_option(args, "--data-dir");
    const std::vector<std::string> positionals = positional_args(args);

    // Reject unknown long options that look like flags (not values).
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--json" || a == "--status" || a == "-h" || a == "--help" || a == "--data-dir") {
            if (a == "--data-dir" && i + 1 < args.size()) {
                ++i;
            }
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            return fail(
                json_mode,
                ToolErrorCode::Usage,
                "Unknown option: " + a,
                kExitUsage);
        }
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, ToolErrorCode::Io, ctx.status().message(), kExitUsage);
    }
    const std::filesystem::path data_root = ctx.value().data_root();

    if (want_status) {
        if (!positionals.empty()) {
            return fail(
                json_mode,
                ToolErrorCode::Usage,
                "--status does not take a theory path",
                kExitUsage);
        }
        nlohmann::json result = status_result(data_root);
        if (json_mode) {
            return ToolCliJson::ok(kTool, std::nullopt, std::move(result));
        }
        std::cout << "parcae-compile stub\n"
                  << "  toolkit_version:      " << PARCAE_VERSION_STRING << '\n'
                  << "  dsl_spec_version:     " << DslSpecVersion::current_string << '\n'
                  << "  dsl_ast_json_version: " << DslAstJsonVersion::current_string << '\n'
                  << "  pipeline_ready:       false\n"
                  << "  data_dir:             " << data_root.string() << '\n'
                  << "  theories_dir:         " << (data_root / "theories").string() << '\n';
        return kExitOk;
    }

    if (positionals.empty()) {
        if (json_mode) {
            return fail(
                json_mode,
                ToolErrorCode::Usage,
                "Usage: parcae-compile --status [--json] | parcae-compile <theory.py> [--json]",
                kExitUsage);
        }
        print_help();
        return kExitUsage;
    }

    if (positionals.size() != 1) {
        return fail(
            json_mode,
            ToolErrorCode::Usage,
            "Expected exactly one theory.py path",
            kExitUsage);
    }

    const std::filesystem::path theory_path = positionals[0];
    if (!std::filesystem::is_regular_file(theory_path)) {
        return fail(
            json_mode,
            ToolErrorCode::Io,
            "Theory source is not a readable file: " + theory_path.string(),
            kExitFail,
            nlohmann::json{{"path", theory_path.string()}});
    }

    // Stub: pipeline not implemented yet.
    return fail(
        json_mode,
        ToolErrorCode::NotBuilt,
        "parcae-compile pipeline not built yet (AST dump → IR → verify → emit). "
        "Use --status for DSL versions; see docs/architecture/python-transpiler.md",
        ToolErrorCodeUtil::exit_status(ToolErrorCode::NotBuilt),
        nlohmann::json{
            {"path", theory_path.string()},
            {"stub", true},
            {"pipeline_ready", false},
            {"dsl_spec_version", std::string(DslSpecVersion::current_string)},
        });
}
