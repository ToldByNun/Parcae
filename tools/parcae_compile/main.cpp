#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/core/version.hpp"
#include "parcae/dsl/dsl_ast_json_version.hpp"
#include "parcae/dsl/dsl_compile.hpp"
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

#ifndef PARCAE_PYTHON_DIR
#define PARCAE_PYTHON_DIR ""
#endif

#ifndef PARCAE_PYTHON_EXE
#define PARCAE_PYTHON_EXE "python"
#endif

namespace {

constexpr std::string_view kTool = "compile";

void print_help() {
    std::cerr
        << "Usage: parcae-compile --status [--json] [--data-dir <path>]\n"
        << "       parcae-compile <theory.py> [--json] [--data-dir <path>] "
           "[--allow-dsl-ignores]\n"
        << "\n"
        << "Compile a theory DSL source into a versioned artifact under data/theories/.\n"
        << "Pipeline: ast_dump → ingest → gate → IR → verify → emit → TheoryArtifact.\n"
        << "Normative: docs/spec/dsl.md, docs/spec/theory-artifact.md\n"
        << "\n"
        << "  --status             Report toolchain / DSL versions (no compile)\n"
        << "  --json               JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir           Parcae data/ root (theories land under <data>/theories/)\n"
        << "  --allow-dsl-ignores  Honor #ignore DSL_FLAG (emits W010; off by default)\n"
        << "  -h, --help           Show this help\n";
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

[[nodiscard]] DslCompile::Options make_compile_options(bool allow_dsl_ignores) {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    (void)opt.set_allow_dsl_ignores(allow_dsl_ignores);
    return opt;
}

[[nodiscard]] nlohmann::json status_result(const std::filesystem::path& data_root) {
    const DslCompile::Options opt = make_compile_options(false);
    const bool ready = DslCompile::pipeline_ready(opt);
    return nlohmann::json{
        {"stub", false},
        {"pipeline_ready", ready},
        {"toolkit_version", PARCAE_VERSION_STRING},
        {"dsl_spec_version", std::string(DslSpecVersion::current_string)},
        {"dsl_ast_json_version", std::string(DslAstJsonVersion::current_string)},
        {"dsl_ast_json_schema", std::string(DslAstJsonVersion::schema_id)},
        {"theory_artifact_schema", "parcae.theory_artifact.v0"},
        {"data_dir", data_root.string()},
        {"theories_dir", (data_root / "theories").string()},
        {"python_exe", opt.python_exe()},
        {"python_path", opt.python_path()},
        {"message",
         ready ? "parcae-compile pipeline ready (ast_dump → IR → verify → artifact)"
               : "parcae-compile: ast_dump frontend missing; check PARCAE_PYTHON_DIR"},
    };
}

[[nodiscard]] std::vector<std::string> positional_args(const std::vector<std::string>& args) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--json" || a == "--status" || a == "-h" || a == "--help" ||
            a == "--allow-dsl-ignores") {
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
    const bool allow_dsl_ignores = has_flag(args, "--allow-dsl-ignores");
    const std::string data_dir = optional_option(args, "--data-dir");
    const std::vector<std::string> positionals = positional_args(args);

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--json" || a == "--status" || a == "-h" || a == "--help" || a == "--data-dir" ||
            a == "--allow-dsl-ignores") {
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
        std::cout << "parcae-compile\n"
                  << "  toolkit_version:      " << PARCAE_VERSION_STRING << '\n'
                  << "  dsl_spec_version:     " << DslSpecVersion::current_string << '\n'
                  << "  dsl_ast_json_version: " << DslAstJsonVersion::current_string << '\n'
                  << "  pipeline_ready:       "
                  << (result.at("pipeline_ready").get<bool>() ? "true" : "false") << '\n'
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

    const DslCompile::Options opt = make_compile_options(allow_dsl_ignores);
    if (!DslCompile::pipeline_ready(opt)) {
        return fail(
            json_mode,
            ToolErrorCode::NotBuilt,
            "parcae-compile: ast_dump frontend not found (PYTHONPATH / PARCAE_PYTHON_DIR)",
            ToolErrorCodeUtil::exit_status(ToolErrorCode::NotBuilt),
            nlohmann::json{
                {"path", theory_path.string()},
                {"python_path", opt.python_path()},
                {"pipeline_ready", false},
            });
    }

    StatusOr<DslCompile::Result> compiled =
        DslCompile::compile_file(theory_path, data_root / "theories", opt);
    if (!compiled.ok()) {
        return fail(
            json_mode,
            ToolErrorCode::Validation,
            compiled.status().message(),
            ToolErrorCodeUtil::exit_status(ToolErrorCode::Validation),
            nlohmann::json{
                {"path", theory_path.string()},
                {"dsl_spec_version", std::string(DslSpecVersion::current_string)},
            });
    }

    for (const DslDiag& w : compiled.value().warnings()) {
        std::cerr << w.format() << '\n';
    }

    nlohmann::json uris = nlohmann::json::array();
    for (const TheoryArtifact& a : compiled.value().artifacts()) {
        uris.push_back(a.uri().to_string());
    }
    nlohmann::json warnings_json = nlohmann::json::array();
    for (const DslDiag& w : compiled.value().warnings()) {
        nlohmann::json entry{
            {"rule", w.rule_id()},
            {"message", w.message()},
        };
        if (w.lineno().has_value()) {
            entry["lineno"] = *w.lineno();
        }
        if (w.col().has_value()) {
            entry["col"] = *w.col();
        }
        warnings_json.push_back(std::move(entry));
    }
    nlohmann::json result{
        {"uris", std::move(uris)},
        {"count", compiled.value().artifacts().size()},
        {"source_sha256", compiled.value().source_sha256()},
        {"dsl_spec_version", std::string(DslSpecVersion::current_string)},
        {"theories_dir", (data_root / "theories").string()},
        {"path", theory_path.string()},
        {"warnings", std::move(warnings_json)},
        {"dsl_ignores_applied", compiled.value().dsl_ignores_applied()},
        {"allow_dsl_ignores", allow_dsl_ignores},
    };

    if (json_mode) {
        return ToolCliJson::ok(kTool, std::nullopt, std::move(result));
    }
    std::cout << "compiled " << compiled.value().artifacts().size() << " theor"
              << (compiled.value().artifacts().size() == 1 ? "y" : "ies") << '\n';
    for (const TheoryArtifact& a : compiled.value().artifacts()) {
        std::cout << "  " << a.uri().to_string() << '\n';
    }
    return kExitOk;
}
