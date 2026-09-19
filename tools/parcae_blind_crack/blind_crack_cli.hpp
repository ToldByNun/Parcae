#ifndef BLIND_CRACK_CLI_HPP
#define BLIND_CRACK_CLI_HPP

#include <iostream>

class BlindCrackCli {
public:
    static void print_help() {
        std::cerr
            << "Usage: parcae-blind-crack [--data-dir <path>] [-h|--help]\n"
            << "\n"
            << "Blind-crack bench on locked Tier-A fixtures (ciphertext only).\n"
            << "Enumerates identity/atbash/caesar/atbash_caesar/affine, ranks by\n"
            << "chi2_english_gp_v0, then oracle-checks against known plaintext.\n"
            << "No unsolved LP2 transcripts ship in-repo — this is the real\n"
            << "additive-family foothold test on Liber Primus-length streams.\n";
    }

private:
    BlindCrackCli() = delete;
};

#endif  // BLIND_CRACK_CLI_HPP
