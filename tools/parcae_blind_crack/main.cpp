#include "cli_io.hpp"

#include "parcae/run/blind_crack.hpp"

#include <iostream>
#include <string>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

void print_help() {
    std::cerr
        << "Usage: parcae-blind-crack [--data-dir <path>] [-h|--help]\n"
        << "\n"
        << "Blind-crack bench on locked Tier-A fixtures (ciphertext only).\n"
        << "Enumerates identity/atbash/caesar/atbash_caesar/affine, ranks by\n"
        << "chi2_english_gp_v0, then oracle-checks against known plaintext.\n"
        << "No unsolved LP2 transcripts ship in-repo — this is the real\n"
        << "additive-family foothold test on Liber Primus-length streams.\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help")) {
        print_help();
        return kExitOk;
    }

    const std::string data_dir = optional_option(args, "--data-dir");
    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return kExitUsage;
    }

    StatusOr<BlindCrack::Report> report = BlindCrack::run_all_locked(ctx.value());
    if (!report.ok()) {
        std::cerr << report.status().message() << '\n';
        return kExitFail;
    }

    std::cout << BlindCrack::format(report.value());
    return kExitOk;
}
