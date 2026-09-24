#include "parcae/run/search_run.hpp"
#include "parcae/run/search_run_console.hpp"
#include "parcae/run/search_run_metrics.hpp"
#include "parcae/tool/tool_backend.hpp"

#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "search_run";

void print_help() {
    std::cerr << "Usage: parcae-search-run [--backend cpu|cuda] [--family caesar]\n"
              << "                         [--seed <u32>] [--stream-length <n>]\n"
              << "                         [--repeats <n>] [--score-id <id>]\n"
              << "                         [--no-compare] [--json] [--omit-timing]\n"
              << "                         [--data-dir <path>]\n"
              << "\n"
              << "AI-style search-run dashboard: throughput (runes/s), sweep scores,\n"
              << "and locked-fixture scorer eval. Timing covers transform+score only\n"
              << "(setup/init excluded). tok_per_sec is intentionally non-deterministic;\n"
              << "use --omit-timing with --json for replayable agent output (params per step).\n"
              << "\nv0 families: caesar|atbash|atbash_caesar|affine|vigenere\n";
}

[[nodiscard]] int fail(bool json_mode, const std::optional<std::string>& backend,
                       ToolErrorCode code, std::string message, int plain_exit) {
    if (json_mode) {
        return ToolCliJson::err(kTool, backend, code, std::move(message));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help")) {
        print_help();
        return CliIo::kExitOk;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");
    const bool omit_timing = CliIo::has_flag(args, "--omit-timing");
    if (omit_timing && !json_mode) {
        std::cerr << "--omit-timing requires --json\n";
        print_help();
        return CliIo::kExitUsage;
    }

    const std::string data_dir = CliIo::optional_option(args, "--data-dir");
    std::optional<std::string> backend_label;

    StatusOr<Backend> backend =
        BackendUtil::from_string(CliIo::optional_option(args, "--backend", "cpu"));
    if (!backend.ok()) {
        print_help();
        return fail(json_mode, std::nullopt, ToolErrorCode::Usage, backend.status().message(),
                    CliIo::kExitUsage);
    }
    backend_label = std::string(BackendUtil::to_string(backend.value()));

    Status backend_ok = BackendUtil::ensure_usable(backend.value());
    if (!backend_ok.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::NotBuilt, backend_ok.message(),
                    CliIo::kExitUsage);
    }

    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Io, ctx.status().message(),
                    CliIo::kExitUsage);
    }

    SearchRun::Options options;
    options.backend = backend.value();
    options.family = CliIo::optional_option(args, "--family", "caesar");
    options.score_id = CliIo::optional_option(args, "--score-id", "chi2_english_gp_v0");
    options.compare_cpu_cuda = !CliIo::has_flag(args, "--no-compare");

    if (backend.value() == Backend::Cuda) {
        if (options.family == "affine") {
            options.stream_length = 1u << 18;
            options.throughput_repeats = 16;
        } else {
            options.stream_length = 1u << 20;
            options.throughput_repeats = 64;
        }
    }

    const std::string seed_text = CliIo::optional_option(args, "--seed");
    if (!seed_text.empty()) {
        try {
            options.seed = static_cast<std::uint32_t>(std::stoul(seed_text));
        } catch (const std::exception&) {
            return fail(json_mode, backend_label, ToolErrorCode::Usage, "Invalid --seed",
                        CliIo::kExitUsage);
        }
    }

    const std::string len_text = CliIo::optional_option(args, "--stream-length");
    if (!len_text.empty()) {
        try {
            options.stream_length = static_cast<std::size_t>(std::stoull(len_text));
        } catch (const std::exception&) {
            return fail(json_mode, backend_label, ToolErrorCode::Usage, "Invalid --stream-length",
                        CliIo::kExitUsage);
        }
    }

    const std::string reps_text = CliIo::optional_option(args, "--repeats");
    if (!reps_text.empty()) {
        try {
            options.throughput_repeats = static_cast<std::size_t>(std::stoull(reps_text));
        } catch (const std::exception&) {
            return fail(json_mode, backend_label, ToolErrorCode::Usage, "Invalid --repeats",
                        CliIo::kExitUsage);
        }
    }

    StatusOr<SearchRunMetrics> metrics = SearchRun::run(ctx.value(), options);
    if (!metrics.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Internal, metrics.status().message(),
                    CliIo::kExitFail);
    }

    const bool eval_ok = metrics.value().eval_set_pass_rate() >= 1.0;
    const bool parity_ok =
        !metrics.value().cpu_cuda_pass().has_value() || metrics.value().cpu_cuda_pass().value();

    if (json_mode) {
        nlohmann::json result = metrics.value().to_json(omit_timing);
        if (eval_ok && parity_ok) {
            return ToolCliJson::ok(kTool, backend_label, std::move(result));
        }
        return ToolCliJson::err(kTool, backend_label, ToolErrorCode::Validation,
                                !eval_ok ? "fixture eval did not all-pass"
                                         : "CPU↔CUDA score parity failed",
                                std::move(result));
    }

    std::cout << SearchRunConsole::format(metrics.value());
    if (!eval_ok || !parity_ok) {
        return CliIo::kExitFail;
    }
    return CliIo::kExitOk;
}
