#include "agent_policy_cli.hpp"
#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/core/version.hpp"
#include "parcae/hypothesis/workspace_manifest.hpp"
#include "parcae/search/batch_artifact.hpp"
#include "parcae/search/search_job.hpp"
#include "parcae/search/search_prior.hpp"
#include "parcae/search/search_scheduler.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

/// Agent allow-list tool name (`docs/spec/agent-tools.md`).
constexpr std::string_view kTool = "search_cycle";
constexpr std::size_t kDefaultK = 16;
constexpr std::uint32_t kDefaultSeed = 1;
constexpr std::size_t kDefaultMaxCandidates = 4096;

void print_help() {
    std::cerr
        << "Usage: parcae-search-cycle --status [--json] [--data-dir <path>]\n"
        << "       parcae-search-cycle --workspace <id> --job <file>\n"
        << "                           [--backend cpu|cuda] [--allow-cuda]\n"
        << "                           [--iterations <n>] [--created-utc <rfc3339>]\n"
        << "                           [--json] [--omit-timing] [--data-dir <path>]\n"
        << "       parcae-search-cycle --workspace <id> --family <id> [--k <n>]\n"
        << "                           [--seed <u32>] [--score-id <id>]\n"
        << "                           [--backend cpu|cuda] [--allow-cuda]\n"
        << "                           [--iterations <n>] [--created-utc <rfc3339>]\n"
        << "                           [--json] [--omit-timing] [--data-dir <path>]\n"
        << "\n"
        << "Workspace closed-loop search: job → BatchArtifact → hypotheses\n"
        << "(SearchScheduler). Normative: docs/spec/search-loop.md\n"
        << "Not a substitute for deny-listed parcae-search-run (metrics only).\n"
        << "\n"
        << "  --status         Toolkit / schema / backend readiness (no cycle)\n"
        << "  --workspace      Workspace id under data/workspaces/ (required for run)\n"
        << "  --job            Path to parcae.search_job.v0 JSON\n"
        << "  --family         Build a job when --job is omitted (caesar|atbash|…;\n"
        << "                   beaufort|totient need --allow-extended-families)\n"
        << "  --k              Top-k (default 16; with --family)\n"
        << "  --seed           Replay seed (default 1; with --family)\n"
        << "  --score-id       Score id (default: workspace default_score_id)\n"
        << "  --backend        cpu|cuda (default cpu; cuda needs --allow-cuda)\n"
        << "  --allow-cuda     AgentPolicy opt-in for --backend cuda\n"
        << "  --allow-extended-families  Opt-in for beaufort|totient families\n"
        << "  --iterations     run_loop count (default 1, >= 1)\n"
        << "  --created-utc    Fixed RFC3339 UTC (YYYY-MM-DDTHH:MM:SSZ) for replayable\n"
        << "                   batch/prior digests; default = wall clock\n"
        << "  --json           JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --omit-timing    Agent-safe: no timing fields / no report.json (requires --json)\n"
        << "  --data-dir       Parcae data/ root\n"
        << "  -h, --help       Show this help\n"
        << "\n"
        << "Planned: --with-agent\n";
}

[[nodiscard]] int fail(
    bool json_mode,
    const std::optional<std::string>& backend,
    ToolErrorCode code,
    std::string message,
    int plain_exit,
    nlohmann::json details = nlohmann::json(nullptr)) {
    if (json_mode) {
        return ToolCliJson::err(kTool, backend, code, std::move(message), std::move(details));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

[[nodiscard]] nlohmann::json status_result(const std::filesystem::path& data_root) {
    return nlohmann::json{
        {"stub", false},
        {"scheduler_ready", true},
        {"run_ready", true},
        {"toolkit_version", PARCAE_VERSION_STRING},
        {"search_job_schema", std::string(SearchJob::schema_id)},
        {"search_prior_schema", std::string(SearchPrior::schema_id)},
        {"batch_artifact_schema", std::string(BatchArtifact::schema_id)},
        {"search_cycle_result_schema", std::string(SearchScheduler::result_schema_id)},
        {"cuda_built", parcae::tool::BackendUtil::cuda_built()},
        {"data_dir", data_root.string()},
        {"message",
         "parcae-search-cycle ready: --workspace + (--job | --family) runs SearchScheduler"},
    };
}

[[nodiscard]] std::string utc_now_rfc3339() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    if (gmtime_s(&tm, &t) != 0) {
        return "1970-01-01T00:00:00Z";
    }
#else
    if (gmtime_r(&t, &tm) == nullptr) {
        return "1970-01-01T00:00:00Z";
    }
#endif
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) {
        return "1970-01-01T00:00:00Z";
    }
    return std::string(buf);
}

[[nodiscard]] StatusOr<std::size_t> parse_size(
    const std::vector<std::string>& args,
    std::string_view flag,
    std::size_t default_value) {
    using namespace parcae::cli;
    const std::string text = optional_option(args, flag);
    if (text.empty()) {
        return default_value;
    }
    try {
        const unsigned long long v = std::stoull(text);
        return static_cast<std::size_t>(v);
    } catch (const std::exception&) {
        return Status::error(std::string(flag) + " must be a non-negative integer");
    }
}

[[nodiscard]] StatusOr<std::uint32_t> parse_u32(
    const std::vector<std::string>& args,
    std::string_view flag,
    std::uint32_t default_value) {
    using namespace parcae::cli;
    const std::string text = optional_option(args, flag);
    if (text.empty()) {
        return default_value;
    }
    try {
        const unsigned long long v = std::stoull(text);
        if (v > 0xFFFFFFFFull) {
            return Status::error(std::string(flag) + " out of uint32 range");
        }
        return static_cast<std::uint32_t>(v);
    } catch (const std::exception&) {
        return Status::error(std::string(flag) + " must be a non-negative integer");
    }
}

[[nodiscard]] StatusOr<SearchJob> job_with_backend(
    const SearchJob& job,
    parcae::tool::Backend backend,
    bool allow_extended_families) {
    const bool extended = job.allow_extended_families() || allow_extended_families;
    if (job.backend() == backend && job.allow_extended_families() == extended) {
        return job;
    }
    return SearchJob::make(
        job.workspace_id(),
        job.family(),
        job.score_id(),
        job.k(),
        job.seed(),
        backend,
        job.max_candidates(),
        job.direction(),
        job.param_grid(),
        job.prior(),
        job.score_version(),
        extended);
}

[[nodiscard]] StatusOr<SearchJob> resolve_job(
    const std::filesystem::path& data_root,
    const std::vector<std::string>& args,
    std::string_view workspace_id,
    parcae::tool::Backend backend) {
    using namespace parcae::cli;
    const std::string job_path = optional_option(args, "--job");
    const std::string family = optional_option(args, "--family");
    const bool allow_extended = has_flag(args, "--allow-extended-families");

    if (!job_path.empty() && !family.empty()) {
        return Status::error("Use either --job or --family, not both");
    }

    if (!job_path.empty()) {
        StatusOr<SearchJob> loaded = SearchJob::load_file(job_path);
        if (!loaded.ok()) {
            return loaded.status();
        }
        if (loaded.value().workspace_id() != workspace_id) {
            return Status::error(
                "SearchJob.workspace_id (" + loaded.value().workspace_id() +
                ") does not match --workspace (" + std::string(workspace_id) + ")");
        }
        return job_with_backend(loaded.value(), backend, allow_extended);
    }

    if (family.empty()) {
        return Status::error("Cycle run requires --job <file> or --family <id>");
    }

    StatusOr<WorkspaceManifest> manifest = WorkspaceManifest::load(data_root, workspace_id);
    if (!manifest.ok()) {
        return manifest.status();
    }
    std::string score_id = optional_option(args, "--score-id");
    if (score_id.empty()) {
        score_id = manifest.value().default_score_id().empty()
                       ? std::string("chi2_english_gp_v0")
                       : manifest.value().default_score_id();
    }
    std::string score_version = manifest.value().default_score_version();
    if (score_version.empty()) {
        score_version = "v0";
    }

    StatusOr<std::size_t> k = parse_size(args, "--k", kDefaultK);
    if (!k.ok()) {
        return k.status();
    }
    if (k.value() == 0) {
        return Status::error("--k must be >= 1");
    }
    StatusOr<std::uint32_t> seed = parse_u32(args, "--seed", kDefaultSeed);
    if (!seed.ok()) {
        return seed.status();
    }
    StatusOr<std::size_t> max_candidates =
        parse_size(args, "--max-candidates", kDefaultMaxCandidates);
    if (!max_candidates.ok()) {
        return max_candidates.status();
    }
    std::size_t max_c = max_candidates.value();
    if (max_c < k.value()) {
        max_c = k.value();
    }

    return SearchJob::make(
        workspace_id,
        family,
        score_id,
        k.value(),
        seed.value(),
        backend,
        max_c,
        TransformDirection::Decrypt,
        nlohmann::json::object(),
        std::nullopt,
        score_version,
        has_flag(args, "--allow-extended-families"));
}

[[nodiscard]] bool is_known_flag(std::string_view a) {
    return a == "--json" || a == "--status" || a == "-h" || a == "--help" || a == "--data-dir" ||
           a == "--workspace" || a == "--job" || a == "--family" || a == "--k" || a == "--seed" ||
           a == "--score-id" || a == "--max-candidates" || a == "--backend" || a == "--allow-cuda" ||
           a == "--allow-extended-families" || a == "--iterations" || a == "--omit-timing" ||
           a == "--created-utc";
}

[[nodiscard]] bool flag_takes_value(std::string_view a) {
    return a == "--data-dir" || a == "--workspace" || a == "--job" || a == "--family" ||
           a == "--k" || a == "--seed" || a == "--score-id" || a == "--max-candidates" ||
           a == "--backend" || a == "--iterations" || a == "--created-utc";
}

[[nodiscard]] StatusOr<std::string> resolve_created_utc(const std::vector<std::string>& args) {
    using namespace parcae::cli;
    const std::string text = optional_option(args, "--created-utc");
    if (text.empty()) {
        return utc_now_rfc3339();
    }
    // Validate YYYY-MM-DDTHH:MM:SSZ via scheduler helper (advance by 0).
    StatusOr<std::string> ok = SearchScheduler::advance_utc_seconds(text, 0);
    if (!ok.ok()) {
        return Status::error(
            "--created-utc must be RFC3339 UTC of the form YYYY-MM-DDTHH:MM:SSZ");
    }
    return ok.value();
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
    const bool omit_timing_flag = has_flag(args, "--omit-timing");
    const std::string data_dir = optional_option(args, "--data-dir");
    std::optional<std::string> backend_label;

    if (omit_timing_flag && !json_mode) {
        std::cerr << "--omit-timing requires --json\n";
        print_help();
        return kExitUsage;
    }

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (is_known_flag(a)) {
            if (flag_takes_value(a) && i + 1 < args.size()) {
                ++i;
            }
            continue;
        }
        if (!a.empty() && a[0] == '-') {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "Unknown option: " + a,
                kExitUsage);
        }
        return fail(
            json_mode,
            std::nullopt,
            ToolErrorCode::Usage,
            "Unexpected argument: " + a,
            kExitUsage);
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, std::nullopt, ToolErrorCode::Io, ctx.status().message(), kExitUsage);
    }
    const std::filesystem::path data_root = ctx.value().data_root();

    if (want_status) {
        if (!optional_option(args, "--workspace").empty() ||
            !optional_option(args, "--job").empty() ||
            !optional_option(args, "--family").empty()) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "--status does not take cycle-run flags",
                kExitUsage);
        }
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
                  << "  run_ready:                    true\n"
                  << "  cuda_built:                   "
                  << (parcae::tool::BackendUtil::cuda_built() ? "true" : "false") << '\n'
                  << "  data_dir:                     " << data_root.string() << '\n';
        return kExitOk;
    }

    const std::string workspace = optional_option(args, "--workspace");
    if (workspace.empty()) {
        if (json_mode) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "Cycle run requires --workspace <id> (or use --status)",
                kExitUsage);
        }
        print_help();
        return kExitUsage;
    }

    const AgentPolicy policy = AgentPolicyCli::make(ctx.value(), args);
    StatusOr<parcae::tool::Backend> backend = AgentPolicyCli::resolve_backend(policy, args);
    if (!backend.ok()) {
        const ToolErrorCode code = AgentPolicyCli::backend_error_code(backend.status());
        backend_label = optional_option(args, "--backend", "cpu");
        return fail(
            json_mode,
            backend_label,
            code,
            backend.status().message(),
            code == ToolErrorCode::NotBuilt ? ToolErrorCodeUtil::exit_status(code) : kExitUsage);
    }
    backend_label = std::string(parcae::tool::BackendUtil::to_string(backend.value()));

    // Gate writes before touching the workspace (fixtures / path escape → policy).
    Status write_gate = policy.allow_workspace_write(workspace, "batches");
    if (!write_gate.ok()) {
        return fail(
            json_mode,
            backend_label,
            AgentPolicy::error_code_for(write_gate),
            write_gate.message(),
            kExitUsage);
    }

    StatusOr<std::size_t> iterations = parse_size(args, "--iterations", 1);
    if (!iterations.ok()) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Usage, iterations.status().message(),
            kExitUsage);
    }
    if (iterations.value() == 0) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Usage, "--iterations must be >= 1",
            kExitUsage);
    }

    StatusOr<SearchJob> job = resolve_job(data_root, args, workspace, backend.value());
    if (!job.ok()) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Usage, job.status().message(), kExitUsage);
    }
    Status ws_ok = job.value().require_workspace_dir(data_root);
    if (!ws_ok.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Io, ws_ok.message(), kExitFail);
    }

    StatusOr<std::string> created_utc = resolve_created_utc(args);
    if (!created_utc.ok()) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Usage, created_utc.status().message(),
            kExitUsage);
    }

    // Agent / --json defaults to omit_timing; human runs may write a digest-only report.
    const bool omit_timing = json_mode || omit_timing_flag;

    SearchScheduler::LoopOptions loop;
    loop.created_utc = std::move(created_utc.value());
    loop.max_iterations = iterations.value();
    loop.omit_timing = omit_timing;

    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_loop(data_root, ctx.value(), job.value(), loop);
    if (!cycle.ok()) {
        ToolErrorCode code = ToolErrorCode::Validation;
        const std::string& msg = cycle.status().message();
        if (msg.find("CUDA") != std::string::npos) {
            code = ToolErrorCode::NotBuilt;
        } else if (msg.find("missing") != std::string::npos ||
                   msg.find("Failed to open") != std::string::npos) {
            code = ToolErrorCode::Io;
        }
        return fail(
            json_mode,
            backend_label,
            code,
            cycle.status().message(),
            ToolErrorCodeUtil::exit_status(code));
    }

    nlohmann::json result = cycle.value().to_json();
    if (json_mode) {
        return ToolCliJson::ok(kTool, backend_label, std::move(result));
    }

    std::cout << "search_cycle\n"
              << "  workspace:            " << cycle.value().workspace_id() << '\n'
              << "  iterations:           " << cycle.value().iterations() << '\n'
              << "  stop_reason:          " << cycle.value().stop_reason() << '\n'
              << "  hypotheses_written:   " << cycle.value().hypotheses_written() << '\n'
              << "  batches:              " << cycle.value().batches().size() << '\n'
              << "  backend:              " << *backend_label << '\n';
    for (const SearchScheduler::BatchSummary& b : cycle.value().batches()) {
        std::cout << "    - " << b.batch_id() << "  candidates=" << b.candidate_count() << '\n';
    }
    return kExitOk;
}
