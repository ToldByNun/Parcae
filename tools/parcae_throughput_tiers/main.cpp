#include "cli_io.hpp"

#include "parcae/run/throughput_tiers.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <iostream>
#include <string>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

void print_help() {
    std::cerr
        << "Usage: parcae-throughput-tiers [--data-dir <path>] [-h|--help]\n"
        << "\n"
        << "CUDA fused throughput gates:\n"
        << "  T1 simple sub/vig chi2     >= 15B runes/s (band 15-35B)\n"
        << "  T2 multi-key/autokey/dyn   >= 3B  runes/s (band 3-10B)\n"
        << "  T3 n-gram + dictionary     >= 1B  runes/s\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help")) {
        print_help();
        return kExitOk;
    }

#if !defined(PARCAE_HAS_CUDA)
    std::cerr << "parcae-throughput-tiers requires a CUDA build\n";
    return kExitUsage;
#else
    Status backend_ok =
        parcae::tool::BackendUtil::ensure_usable(parcae::tool::Backend::Cuda);
    if (!backend_ok.ok()) {
        std::cerr << backend_ok.message() << '\n';
        return kExitUsage;
    }

    const std::string data_dir = optional_option(args, "--data-dir");
    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return kExitUsage;
    }
    StatusOr<ExpectedFrequencyTable> freqs = ctx.value().load_english_gp_expected();
    if (!freqs.ok()) {
        std::cerr << freqs.status().message() << '\n';
        return kExitFail;
    }

    StatusOr<ThroughputTiers::Report> report = ThroughputTiers::run(freqs.value());
    if (!report.ok()) {
        std::cerr << report.status().message() << '\n';
        return kExitFail;
    }
    std::cout << ThroughputTiers::format(report.value());
    return report.value().all_pass ? kExitOk : kExitFail;
#endif
}
