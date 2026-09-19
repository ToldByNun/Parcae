#ifndef THROUGHPUT_TIERS_CLI_HPP
#define THROUGHPUT_TIERS_CLI_HPP

#include <iostream>

class ThroughputTiersCli {
public:
    static void print_help() {
        std::cerr
            << "Usage: parcae-throughput-tiers [--data-dir <path>] [-h|--help]\n"
            << "\n"
            << "CUDA fused throughput gates:\n"
            << "  T1 simple sub/vig chi2     >= 15B runes/s (band 15-35B)\n"
            << "  T2 multi-key/autokey/dyn   >= 3B  runes/s (band 3-10B)\n"
            << "  T3 n-gram + dictionary     >= 1B  runes/s\n";
    }

private:
    ThroughputTiersCli() = delete;
};

#endif  // THROUGHPUT_TIERS_CLI_HPP
