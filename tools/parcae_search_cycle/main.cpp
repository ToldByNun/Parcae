#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/core/version.hpp"
#include "parcae/search/batch_artifact.hpp"
#include "parcae/search/search_job.hpp"
#include "parcae/search/search_prior.hpp"
#include "parcae/search/search_scheduler.hpp"
#include "parcae/tool/tool_backend.hpp"

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

/// Agent allow-list tool name (`docs/spec/agent-tools.md`).
constexpr std::string_view kTool = "search_cycle";

void print_help() {
    std::cerr
        << "Usage: parcae-search-cycle --status [--json] [--data-dir <path>]\n"
        << "\n"
        << "Workspace closed-loop search: job → BatchArtifact → hypotheses\n"
        << "(SearchScheduler). Normative: docs/spec/search-loop.md\n"
        << "Not a substitute for deny-listed parcae-search-run (metrics only).\n"
        << "\n"
        << "Scaffold (H27):\n"
        << "  --status       Report toolkit / schema / backend readiness (no cycle)\n"
        << "  --json         JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir     Parcae data/ root\n"
        << "  -h, --help     Show this help\n"
        << "\n"
        << "Planned run flags (later commits): --workspace --job --backend\n"
        << "  --iterations --omit-timing --with-agent\n";
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
        {"stub", false},
        {"scheduler_ready", true},
        {"run_ready", false},
        {"toolkit_version", PARCAE_VERSION_STRING},
        {"search_job_schema", std::string(SearchJob::schema_id)},
        {"search_prior_schema", std::string(SearchPrior::schema_id)},
        {"batch_artifact_schema", std::string(BatchArtifact::schema_id)},
        {"search_cycle_result_schema", std::string(SearchScheduler::result_schema_id)},
        {"cuda_built", parcae::tool::BackendUtil::cuda_built()},
        {"data_dir", data_root.string()},
        {"message",
         "parcae-search-cycle --status ok; cycle run flags (--workspace / --job) land in "
         "later commits"},
    };
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
            return fail(
                json_mode,
                ToolErrorCode::Usage,
                "Unknown option: " + a + " (scaffold supports --status / --json / --data-dir)",
                kExitUsage);
        }
        return fail(
            json_mode,
            ToolErrorCode::Usage,
            "Unexpected argument: " + a,
            kExitUsage);
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, ToolErrorCode::Io, ctx.status().message(), kExitUsage);
    }
    const std::filesystem::path data_root = ctx.value().data_root();

    if (want_status) {
        nlohmann::json result = status_result(data_root);
        if (json_mode) {
            return ToolCliJson::ok(kTool, std::nullopt, std::move(result));
        }
        std::cout << "parcae-search-cycle\n"
                  << "  toolkit_version:              " << PARCAE_VERSION_STRING << '\n'
                  << "  search_cycle_result_schema:   " << SearchScheduler::result_schema_id
                  << '\n'
                  << "  search_job_schema:            " << SearchJob::schema_id << '\n'
                  << "  scheduler_ready:              true\n"
                  << "  run_ready:                    false (use --status only for now)\n"
                  << "  cuda_built:                   "
                  << (parcae::tool::BackendUtil::cuda_built() ? "true" : "false") << '\n'
                  << "  data_dir:                     " << data_root.string() << '\n';
        return kExitOk;
    }

    if (json_mode) {
        return fail(
            json_mode,
            ToolErrorCode::Usage,
            "Usage: parcae-search-cycle --status [--json] [--data-dir <path>]",
            kExitUsage);
    }
    print_help();
    return kExitUsage;
}
