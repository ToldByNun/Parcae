#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/dsl/theory_sweep.hpp"

#include <cstdlib>
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

constexpr std::string_view kTool = "sweep";

void print_help() {
    std::cerr
        << "Usage: parcae-sweep --theory <uri|name@version> [--limit <n>] [--json]\n"
        << "\n"
        << "Expand TheoryArtifact.sweep.param_grid into a candidate plan.\n"
        << "Loads via TheoryRegistry (stale dsl_spec_version → fail).\n"
        << "Plan-only in v0 — does not apply/score (TheoryDispatch comes later).\n"
        << "Normative: docs/spec/theory-artifact.md\n"
        << "\n"
        << "  --theory <ref>  parcae://theories/<name>@<ver> or <name>@<ver>\n"
        << "  --limit <n>     Cap expanded candidates (default 100000; 0 = unlimited)\n"
        << "  --json          JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir      Parcae data/ root (theories under <data>/theories/)\n"
        << "  -h, --help      Show this help\n";
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

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? CliIo::kExitUsage : CliIo::kExitOk;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");
    const std::string data_dir = CliIo::optional_option(args, "--data-dir");

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--json" || arg == "-h" || arg == "--help") {
            continue;
        }
        if (arg == "--theory" || arg == "--limit" || arg == "--data-dir") {
            ++i;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            print_help();
            return fail(json_mode, ToolErrorCode::Usage, "Unknown option: " + arg, CliIo::kExitUsage);
        }
        print_help();
        return fail(json_mode, ToolErrorCode::Usage, "Unexpected argument: " + arg, CliIo::kExitUsage);
    }

    StatusOr<std::string> theory = CliIo::require_option(args, "--theory");
    if (!theory.ok()) {
        print_help();
        return fail(json_mode, ToolErrorCode::Usage, theory.status().message(), CliIo::kExitUsage);
    }

    TheorySweep::Options opt;
    if (CliIo::has_flag(args, "--limit")) {
        StatusOr<std::string> lim = CliIo::require_option(args, "--limit");
        if (!lim.ok()) {
            return fail(json_mode, ToolErrorCode::Usage, lim.status().message(), CliIo::kExitUsage);
        }
        char* end = nullptr;
        const unsigned long long v = std::strtoull(lim.value().c_str(), &end, 10);
        if (end == lim.value().c_str() || (end && *end != '\0')) {
            return fail(json_mode, ToolErrorCode::Usage, "Invalid --limit", CliIo::kExitUsage);
        }
        opt.set_limit(static_cast<std::size_t>(v));
    }

    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, ToolErrorCode::Io, ctx.status().message(), CliIo::kExitUsage);
    }

    const std::filesystem::path theories_root = ctx.value().data_root() / "theories";
    StatusOr<TheorySweep::Plan> plan =
        TheorySweep::plan_uri(theories_root, theory.value(), opt);
    if (!plan.ok()) {
        const std::string& msg = plan.status().message();
        ToolErrorCode code = ToolErrorCode::Validation;
        if (msg.find("DSL spec") != std::string::npos ||
            msg.find("re-run parcae-compile") != std::string::npos) {
            code = ToolErrorCode::Validation;
        } else if (msg.find("Failed to open") != std::string::npos ||
                   msg.find("not a directory") != std::string::npos) {
            code = ToolErrorCode::Io;
        } else if (msg.find("must be") != std::string::npos ||
                   msg.find("ref must") != std::string::npos ||
                   msg.find("invalid") != std::string::npos) {
            code = ToolErrorCode::Schema;
        }
        return fail(
            json_mode,
            code,
            msg,
            ToolErrorCodeUtil::exit_status(code));
    }

    if (json_mode) {
        return ToolCliJson::ok(kTool, std::nullopt, plan.value().to_json());
    }

    const TheorySweep::Plan& p = plan.value();
    std::cout << "uri\t" << p.uri() << '\n';
    std::cout << "corpus\t" << p.corpus() << '\n';
    if (p.compare_against().has_value()) {
        std::cout << "compare_against\t" << *p.compare_against() << '\n';
    }
    std::cout << "metrics\t";
    for (std::size_t i = 0; i < p.record_metrics().size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << p.record_metrics()[i];
    }
    std::cout << '\n';
    std::cout << "count\t" << p.candidates().size();
    if (p.truncated()) {
        std::cout << "\t(truncated from " << p.total_before_limit() << ")";
    }
    std::cout << '\n';
    std::cout << "note\tplan-only; apply/score requires TheoryDispatch\n";
    constexpr std::size_t kPreview = 8;
    for (std::size_t i = 0; i < p.candidates().size() && i < kPreview; ++i) {
        std::cout << "  candidate[" << i << "]\t" << p.candidates()[i].dump() << '\n';
    }
    if (p.candidates().size() > kPreview) {
        std::cout << "  …\t(" << (p.candidates().size() - kPreview) << " more)\n";
    }
    return CliIo::kExitOk;
}
