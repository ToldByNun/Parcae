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
            << "\n"
            << "Benchmark & diagnostics umbrella (toolkit bench target).\n"
            << "\n"
            << "  --status         Toolkit / suite / CUDA readiness (no measurement)\n"
            << "  --suite slo      Fused CUDA SLO tiers T1–T3 (default when not --status)\n"
            << "  --suite accuracy Statistical validation (CPU; CUDA extras with --allow-cuda)\n"
            << "  --extended       Also run F.* and C.* rows (with --suite slo)\n"
            << "  --allow-cuda     Required for --suite slo; optional CUDA checks for accuracy\n"
            << "  --json           Emit parcae.tool_response.v0 on stdout\n"
            << "  --omit-timing    Drop rate/wall fields (requires --json)\n"
            << "  --data-dir PATH  Parcae data root (profiles / fixtures)\n"
            << "\n"
            << "Compat: `parcae-throughput-tiers` runs the same suite as\n"
            << "  --suite slo --extended --allow-cuda (no allow-cuda flag on that binary).\n"
            << "\n"
            << "Suites hardware|probe|all are reserved for follow-up work.\n";
    }

    [[nodiscard]] static nlohmann::json status_result(std::string_view data_dir) {
        return nlohmann::json{
            {"toolkit_version", PARCAE_VERSION_STRING},
            {"cuda_built", parcae::tool::BackendUtil::cuda_built()},
            {"suites",
             nlohmann::json{
                 {"slo", true},
                 {"accuracy", true},
                 {"hardware", false},
                 {"probe", false},
                 {"all", false},
             }},
            {"data_dir", std::string(data_dir)},
            {"message",
             "parcae-bench ready: --status | --suite slo --allow-cuda | --suite accuracy"},
        };
    }

private:
    BenchCli() = delete;
};

#endif // BENCH_CLI_HPP
