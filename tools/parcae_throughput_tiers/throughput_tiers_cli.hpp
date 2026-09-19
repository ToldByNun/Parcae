#ifndef THROUGHPUT_TIERS_CLI_HPP
#define THROUGHPUT_TIERS_CLI_HPP

#include <iostream>

class ThroughputTiersCli {
public:
    static void print_help() {
        std::cerr
            << "Usage: parcae-throughput-tiers [--data-dir <path>] [-h|--help]\n"
            << "\n"
            << "CUDA fused throughput gates (≥90% practical peak):\n"
            << "  T1..T3   SLO tiers\n"
            << "  F.*      transform families (atbash/affine/vigenere/beaufort/totient)\n"
            << "  C.*      compose recipes (Koan-1 fused + staged)\n";
    }

private:
    ThroughputTiersCli() = delete;
};

#endif  // THROUGHPUT_TIERS_CLI_HPP
