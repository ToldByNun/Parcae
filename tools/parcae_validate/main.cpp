#include "parcae/dsl/theory_validate.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/validate/validation_report.hpp"

#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "validate";

void print_help() {
    std::cerr << "Usage: parcae-validate --id <fixture_id|path> [--require-locked] [--json]\n"
              << "       parcae-validate --all [--require-locked] [--json]\n"
              << "       parcae-validate --theory <uri|name@version|dir> [--json]\n"
              << "       parcae-validate --theories [--json]\n"
              << "\n"
              << "Fixture mode (solved corpus):\n"
              << "  --id <id|path>     One solved fixture id or fixture directory\n"
              << "  --all              All fixtures under data/fixtures/solved/\n"
              << "                     (with --require-locked: locked fixtures only)\n"
              << "  --require-locked   Fail (or skip under --all) non-locked fixtures\n"
              << "\n"
              << "Theory artifact mode (docs/spec/theory-artifact.md):\n"
              << "  --theory <ref>     One theory: parcae://theories/<name>@<ver>,\n"
              << "                     <name>@<ver>, or artifact directory / manifest.json\n"
              << "  --theories         All theory artifacts under data/theories/\n"
              << "                     (stale dsl_spec_version → FAIL)\n"
              << "\n"
              << "Common:\n"
              << "  --json             JSON envelope on stdout (parcae.tool_response.v0)\n"
              << "  --data-dir         Parcae data/ root\n"
              << "  -h, --help         Show this help\n"
              << "\n"
              << "Exit: 0 all selected passed; 1 validation failure; 2 usage/I/O error\n";
}

[[nodiscard]] int fail(bool json_mode, ToolErrorCode code, std::string message, int plain_exit,
                       nlohmann::json details = nlohmann::json(nullptr)) {
    if (json_mode) {
        return ToolCliJson::err(kTool, std::nullopt, code, std::move(message), std::move(details));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

[[nodiscard]] nlohmann::json report_to_json(const ValidationReport& report) {
    nlohmann::json checks = nlohmann::json::array();
    for (const ValidationCheck& check : report.checks()) {
        checks.push_back(nlohmann::json{
            {"name", check.name()},
            {"ok", check.ok()},
            {"message", check.message()},
        });
    }
    nlohmann::json root{
        {"ok", report.ok()},
        {"fixture_id", report.fixture_id()},
        {"checks", std::move(checks)},
    };
    if (report.diff_excerpt().has_value()) {
        root["diff_excerpt"] = report.diff_excerpt().value();
    } else {
        root["diff_excerpt"] = nullptr;
    }
    return root;
}

[[nodiscard]] nlohmann::json theory_report_to_json(const TheoryValidate::Report& report) {
    nlohmann::json checks = nlohmann::json::array();
    for (const TheoryValidate::Check& check : report.checks()) {
        checks.push_back(nlohmann::json{
            {"name", check.name()},
            {"ok", check.ok()},
            {"message", check.message()},
        });
    }
    nlohmann::json root{
        {"ok", report.ok()},
        {"kind", "theory"},
        {"uri", report.uri()},
        {"checks", std::move(checks)},
    };
    if (report.dsl_spec_version().has_value()) {
        root["dsl_spec_version"] = *report.dsl_spec_version();
    } else {
        root["dsl_spec_version"] = nullptr;
    }
    return root;
}

void print_report_human(const ValidationReport& report) {
    std::cout << (report.ok() ? "PASS" : "FAIL") << '\t' << report.fixture_id() << '\n';
    for (const ValidationCheck& check : report.checks()) {
        std::cout << "  " << (check.ok() ? "ok" : "FAIL") << '\t' << check.name() << '\t'
                  << check.message() << '\n';
    }
    if (report.diff_excerpt().has_value()) {
        std::cout << "  diff\t" << report.diff_excerpt().value() << '\n';
    }
}

void print_theory_report_human(const TheoryValidate::Report& report) {
    std::cout << (report.ok() ? "PASS" : "FAIL") << '\t' << report.uri() << '\n';
    for (const TheoryValidate::Check& check : report.checks()) {
        std::cout << "  " << (check.ok() ? "ok" : "FAIL") << '\t' << check.name() << '\t'
                  << check.message() << '\n';
    }
}

[[nodiscard]] StatusOr<bool> fixture_is_locked(const std::filesystem::path& dir) {
    const std::filesystem::path manifest = dir / "manifest.json";
    StatusOr<std::string> text = CliIo::read_file_utf8(manifest);
    if (!text.ok()) {
        return text.status();
    }
    try {
        const nlohmann::json root = nlohmann::json::parse(text.value());
        if (!root.contains("verification") || !root.at("verification").contains("status")) {
            return false;
        }
        return root.at("verification").at("status").get<std::string>() == "locked";
    } catch (const nlohmann::json::exception& ex) {
        return Status::error(std::string("Invalid manifest: ") + ex.what());
    }
}

[[nodiscard]] StatusOr<std::vector<std::string>> list_solved_fixture_ids(const Context& ctx,
                                                                         bool locked_only) {
    const std::filesystem::path solved = ctx.data_root() / "fixtures" / "solved";
    if (!std::filesystem::is_directory(solved)) {
        return Status::error("Missing fixtures/solved under data root");
    }

    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(solved)) {
        if (!entry.is_directory()) {
            continue;
        }
        if (!std::filesystem::is_regular_file(entry.path() / "manifest.json")) {
            continue;
        }
        if (locked_only) {
            StatusOr<bool> locked = fixture_is_locked(entry.path());
            if (!locked.ok()) {
                return locked.status();
            }
            if (!locked.value()) {
                continue;
            }
        }
        ids.push_back(entry.path().filename().string());
    }
    std::sort(ids.begin(), ids.end());
    if (ids.empty()) {
        return Status::error(locked_only ? "No locked fixtures found under fixtures/solved"
                                         : "No fixtures found under fixtures/solved");
    }
    return ids;
}

[[nodiscard]] int run_fixture_mode(bool json_mode, bool require_locked, bool all_mode, bool has_id,
                                   const std::vector<std::string>& args, const Context& ctx) {
    if (static_cast<int>(all_mode) + static_cast<int>(has_id) != 1) {
        print_help();
        return fail(json_mode, ToolErrorCode::Usage, "Choose exactly one of --id or --all",
                    CliIo::kExitUsage);
    }

    std::vector<std::string> targets;
    if (all_mode) {
        StatusOr<std::vector<std::string>> ids =
            list_solved_fixture_ids(ctx, /*locked_only=*/require_locked);
        if (!ids.ok()) {
            return fail(json_mode, ToolErrorCode::Io, ids.status().message(), CliIo::kExitUsage);
        }
        targets = std::move(ids.value());
    } else {
        StatusOr<std::string> id = CliIo::require_option(args, "--id");
        if (!id.ok()) {
            print_help();
            return fail(json_mode, ToolErrorCode::Usage, id.status().message(), CliIo::kExitUsage);
        }
        targets.push_back(id.value());
    }

    nlohmann::json reports = nlohmann::json::array();
    bool all_ok = true;

    for (const std::string& target : targets) {
        const ValidationReport report = ToolApi::validate_fixture(ctx, target, require_locked);
        if (!report.ok()) {
            all_ok = false;
        }
        if (json_mode) {
            reports.push_back(report_to_json(report));
        } else {
            print_report_human(report);
        }
    }

    if (json_mode) {
        // Keep fixture result shape stable for goldens/agents:
        // { ok, count, reports:[{ fixture_id, ok, checks, diff_excerpt }] }
        nlohmann::json result{
            {"ok", all_ok},
            {"count", reports.size()},
            {"reports", std::move(reports)},
        };
        if (all_ok) {
            return ToolCliJson::ok(kTool, std::nullopt, std::move(result));
        }
        return ToolCliJson::err(kTool, std::nullopt, ToolErrorCode::Validation,
                                "One or more fixtures failed validation", std::move(result));
    }
    return all_ok ? CliIo::kExitOk : CliIo::kExitFail;
}

[[nodiscard]] int run_theory_mode(bool json_mode, bool theories_all, bool has_theory,
                                  const std::vector<std::string>& args, const Context& ctx) {
    if (static_cast<int>(theories_all) + static_cast<int>(has_theory) != 1) {
        print_help();
        return fail(json_mode, ToolErrorCode::Usage, "Choose exactly one of --theory or --theories",
                    CliIo::kExitUsage);
    }

    const std::filesystem::path theories_root = ctx.data_root() / "theories";
    std::vector<TheoryValidate::Report> reports;

    if (theories_all) {
        StatusOr<std::vector<TheoryValidate::Report>> all =
            TheoryValidate::validate_all(theories_root);
        if (!all.ok()) {
            return fail(json_mode, ToolErrorCode::Io, all.status().message(), CliIo::kExitUsage);
        }
        reports = std::move(all.value());
    } else {
        StatusOr<std::string> ref = CliIo::require_option(args, "--theory");
        if (!ref.ok()) {
            print_help();
            return fail(json_mode, ToolErrorCode::Usage, ref.status().message(), CliIo::kExitUsage);
        }
        reports.push_back(TheoryValidate::validate_target(theories_root, ref.value()));
    }

    nlohmann::json json_reports = nlohmann::json::array();
    bool all_ok = true;
    for (const TheoryValidate::Report& report : reports) {
        if (!report.ok()) {
            all_ok = false;
        }
        if (json_mode) {
            json_reports.push_back(theory_report_to_json(report));
        } else {
            print_theory_report_human(report);
        }
    }

    if (json_mode) {
        nlohmann::json result{
            {"mode", "theory"},
            {"theories_dir", theories_root.string()},
            {"reports", std::move(json_reports)},
            {"all_ok", all_ok},
            {"count", reports.size()},
        };
        if (all_ok) {
            return ToolCliJson::ok(kTool, std::nullopt, std::move(result));
        }
        return ToolCliJson::err(kTool, std::nullopt, ToolErrorCode::Validation,
                                "One or more theory artifacts failed validation",
                                std::move(result));
    }
    if (reports.empty() && theories_all) {
        std::cout << "PASS\t(no theory artifacts under " << theories_root.string() << ")\n";
    }
    return all_ok ? CliIo::kExitOk : CliIo::kExitFail;
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? CliIo::kExitUsage : CliIo::kExitOk;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");
    const bool require_locked = CliIo::has_flag(args, "--require-locked");
    const bool all_mode = CliIo::has_flag(args, "--all");
    const bool has_id = CliIo::has_flag(args, "--id");
    const bool theories_all = CliIo::has_flag(args, "--theories");
    const bool has_theory = CliIo::has_flag(args, "--theory");
    const std::string data_dir = CliIo::optional_option(args, "--data-dir");

    const bool fixture_mode = all_mode || has_id;
    const bool theory_mode = theories_all || has_theory;
    if (fixture_mode && theory_mode) {
        print_help();
        return fail(json_mode, ToolErrorCode::Usage,
                    "Do not mix fixture flags (--id/--all) with theory flags (--theory/--theories)",
                    CliIo::kExitUsage);
    }
    if (!fixture_mode && !theory_mode) {
        print_help();
        return fail(json_mode, ToolErrorCode::Usage,
                    "Choose fixture mode (--id/--all) or theory mode (--theory/--theories)",
                    CliIo::kExitUsage);
    }
    if (require_locked && theory_mode) {
        return fail(json_mode, ToolErrorCode::Usage,
                    "--require-locked applies only to fixture mode", CliIo::kExitUsage);
    }

    // Reject unknown options.
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--json" || arg == "--all" || arg == "--require-locked" || arg == "--theories" ||
            arg == "-h" || arg == "--help") {
            continue;
        }
        if (arg == "--id" || arg == "--theory" || arg == "--data-dir") {
            ++i;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            print_help();
            return fail(json_mode, ToolErrorCode::Usage, "Unknown option: " + arg,
                        CliIo::kExitUsage);
        }
    }

    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, ToolErrorCode::Io, ctx.status().message(), CliIo::kExitUsage);
    }

    if (theory_mode) {
        return run_theory_mode(json_mode, theories_all, has_theory, args, ctx.value());
    }
    return run_fixture_mode(json_mode, require_locked, all_mode, has_id, args, ctx.value());
}
