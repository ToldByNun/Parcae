#include "cli_io.hpp"
#include "throughput_tiers_cli.hpp"

#include "parcae/bench/bench_formatter.hpp"
#include "parcae/bench/bench_slo_suite.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <iostream>
#include <string>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

/// Compatibility entry point: same suite as
/// `parcae-bench --suite slo --extended --allow-cuda` (no --allow-cuda flag
/// here so existing scripts keep working).
int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help")) {
        ThroughputTiersCli::print_help();
        return CliIo::kExitOk;
    }

#if !defined(PARCAE_HAS_CUDA)
    std::cerr << "parcae-throughput-tiers requires a CUDA build\n";
    std::cerr << "Use: parcae-bench --status --json   (CPU) or a CUDA-linked build\n";
    return CliIo::kExitUsage;
#else
    Status backend_ok =
        BackendUtil::ensure_usable(Backend::Cuda);
    if (!backend_ok.ok()) {
        std::cerr << backend_ok.message() << '\n';
        return CliIo::kExitUsage;
    }

    const std::string data_dir = CliIo::optional_option(args, "--data-dir");
    StatusOr<Context> ctx =
        CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return CliIo::kExitUsage;
    }
    StatusOr<ExpectedFrequencyTable> freqs = ctx.value().load_english_gp_expected();
    if (!freqs.ok()) {
        std::cerr << freqs.status().message() << '\n';
        return CliIo::kExitFail;
    }

    // Historical full suite = primary T1–T3 + extended F.* / C.* rows.
    BenchSloSuite::Options opts(/*extended=*/true);
    StatusOr<BenchReport::Document> doc = BenchSloSuite::run(freqs.value(), opts);
    if (!doc.ok()) {
        std::cerr << doc.status().message() << '\n';
        return CliIo::kExitFail;
    }
    std::cout << BenchFormatter::format(doc.value());
    return doc.value().all_pass() ? CliIo::kExitOk : CliIo::kExitFail;
#endif
}
