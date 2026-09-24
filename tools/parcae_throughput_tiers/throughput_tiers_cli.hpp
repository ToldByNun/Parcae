#ifndef THROUGHPUT_TIERS_CLI_HPP
#define THROUGHPUT_TIERS_CLI_HPP

#include <iostream>

/// Compat CLI help for `parcae-throughput-tiers` (thin wrapper of BenchSloSuite).
class ThroughputTiersCli {
public:
    static void print_help() {
        std::cerr << "Usage: parcae-throughput-tiers [--data-dir <path>] [-h|--help]\n"
                  << "\n"
                  << "Compatibility wrapper for the fused CUDA SLO suite.\n"
                  << "Equivalent to:\n"
                  << "  parcae-bench --suite slo --extended --allow-cuda [--data-dir …]\n"
                  << "\n"
                  << "Runs the same BenchSloSuite path (T1–T3 + F.* + C.*):\n"
                  << "  T1..T3   SLO tiers\n"
                  << "  F.*      transform families (atbash/affine/vigenere/beaufort/totient)\n"
                  << "  C.*      compose recipes (Koan-1 fused + staged)\n"
                  << "\n"
                  << "Prefer `parcae-bench` for new scripts (JSON / --omit-timing / --status).\n";
    }

private:
    ThroughputTiersCli() = delete;
};

#endif // THROUGHPUT_TIERS_CLI_HPP
