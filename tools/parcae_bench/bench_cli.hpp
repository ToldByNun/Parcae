#ifndef BENCH_CLI_HPP
#define BENCH_CLI_HPP

#include "parcae/core/version.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <iostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

/// Help / status helpers for `parcae-bench`.
class BenchCli {
public:
    static void print_help() {
        std::cerr
            << "Usage: parcae-bench --status [--json] [--data-dir <path>]\n"
            << "       parcae-bench --suite slo [--extended] --allow-cuda\n"
            << "                    [--json] [--omit-timing] [--data-dir <path>]\n"
            << "       parcae-bench --suite accuracy [--allow-cuda]\n"
            << "                    [--json] [--omit-timing] [--data-dir <path>]\n"
            << "       parcae-bench --suite hardware [--backend cpu|cuda|both]\n"
            << "                    [--allow-cuda|--require-cuda] [--allow-skip]\n"
            << "                    [--cpu-full] [--json] [--omit-timing]\n"
            << "                    [--data-dir <path>]\n"
            << "       parcae-bench --suite probe --probe-cmd <cmd>\n"
            << "                    [--probe-tiers T1,T2,T3]\n"
            << "                    [--probe-timeout-ms N] [--compare-builtin]\n"
            << "                    [--json] [--omit-timing] [--data-dir <path>]\n"
            << "       parcae-bench --suite all [flags for each leg…]\n"
            << "\n"
            << "Benchmark & diagnostics umbrella (toolkit bench target).\n"
            << "\n"
            << "  --status              Toolkit / suite / CUDA readiness (no measurement)\n"
            << "  --suite slo           Fused CUDA SLO tiers T1–T3 (default when not --status)\n"
            << "  --suite accuracy      Statistical validation (CPU; CUDA extras with --allow-cuda)\n"
            << "  --suite hardware      CPU vs CUDA T1–T3 compare (scaled CPU smoke by default)\n"
            << "  --suite probe         External JSON subprocess probes (see docs/spec/bench-probe.md)\n"
            << "  --suite all           accuracy → slo → hardware → probe (probe only if --probe-cmd)\n"
            << "  --extended            Also run F.* and C.* rows (with --suite slo)\n"
            << "  --backend MODE        hardware only: cpu|cuda|both (default both)\n"
            << "  --allow-cuda          Required for slo; CUDA leg for accuracy/hardware\n"
            << "  --require-cuda        hardware: fail if CUDA unavailable\n"
            << "  --allow-skip          hardware: OK when CUDA rows are skipped_not_built\n"
            << "  --cpu-full            hardware: full T1–T3 CPU C/T/reps (not smoke)\n"
            << "  --probe-cmd CMD       External command; `{tier}` substituted per tier\n"
            << "  --probe-tiers LIST    Comma list (default T1,T2,T3)\n"
            << "  --probe-timeout-ms N  Per-tier spawn timeout (default 120000)\n"
            << "  --compare-builtin     Probe: require config match BenchTierSpec; ratio vs SLO\n"
            << "  --json                Emit parcae.tool_response.v0 on stdout\n"
            << "  --omit-timing         Drop rate/wall fields (requires --json)\n"
            << "  --data-dir PATH       Parcae data root (profiles / fixtures)\n"
            << "\n"
            << "Compat: `parcae-throughput-tiers` runs the same suite as\n"
            << "  --suite slo --extended --allow-cuda (no allow-cuda flag on that binary).\n";
    }

    [[nodiscard]] static nlohmann::json status_result(std::string_view data_dir) {
        return nlohmann::json{
            {"toolkit_version", PARCAE_VERSION_STRING},
            {"cuda_built", BackendUtil::cuda_built()},
            {"suites",
             nlohmann::json{
                 {"slo", true},
                 {"accuracy", true},
                 {"hardware", true},
                 {"probe", true},
                 {"all", true},
             }},
            {"data_dir", std::string(data_dir)},
            {"message",
             "parcae-bench ready: --status | --suite slo|accuracy|hardware|probe|all"},
        };
    }

private:
    BenchCli() = delete;
};

#endif // BENCH_CLI_HPP
