#include "cli_io.hpp"
#include "throughput_tiers_cli.hpp"

#include "parcae/run/throughput_tiers.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <iostream>
#include <string>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

int main(int argc, char** argv) {
    const std::vector<std::string> args = parcae::cli::argv_tail(argc, argv);
    if (parcae::cli::has_flag(args, "-h") || parcae::cli::has_flag(args, "--help")) {
        ThroughputTiersCli::print_help();
        return parcae::cli::kExitOk;
    }

#if !defined(PARCAE_HAS_CUDA)
    std::cerr << "parcae-throughput-tiers requires a CUDA build\n";
    return parcae::cli::kExitUsage;
#else
    Status backend_ok =
        parcae::tool::BackendUtil::ensure_usable(parcae::tool::Backend::Cuda);
    if (!backend_ok.ok()) {
        std::cerr << backend_ok.message() << '\n';
        return parcae::cli::kExitUsage;
    }

    const std::string data_dir = parcae::cli::optional_option(args, "--data-dir");
    StatusOr<parcae::tool::Context> ctx =
        parcae::cli::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return parcae::cli::kExitUsage;
    }
    StatusOr<ExpectedFrequencyTable> freqs = ctx.value().load_english_gp_expected();
    if (!freqs.ok()) {
        std::cerr << freqs.status().message() << '\n';
        return parcae::cli::kExitFail;
    }

    StatusOr<ThroughputTiers::Report> report = ThroughputTiers::run(freqs.value());
    if (!report.ok()) {
        std::cerr << report.status().message() << '\n';
        return parcae::cli::kExitFail;
    }
    std::cout << ThroughputTiers::format(report.value());
    return report.value().all_pass ? parcae::cli::kExitOk : parcae::cli::kExitFail;
#endif
}
