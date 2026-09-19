#include "blind_crack_cli.hpp"
#include "cli_io.hpp"

#include "parcae/run/blind_crack.hpp"

#include <iostream>
#include <string>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

int main(int argc, char** argv) {
    const std::vector<std::string> args = parcae::cli::argv_tail(argc, argv);
    if (parcae::cli::has_flag(args, "-h") || parcae::cli::has_flag(args, "--help")) {
        BlindCrackCli::print_help();
        return parcae::cli::kExitOk;
    }

    const std::string data_dir = parcae::cli::optional_option(args, "--data-dir");
    StatusOr<parcae::tool::Context> ctx =
        parcae::cli::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return parcae::cli::kExitUsage;
    }

    StatusOr<BlindCrack::Report> report = BlindCrack::run_all_locked(ctx.value());
    if (!report.ok()) {
        std::cerr << report.status().message() << '\n';
        return parcae::cli::kExitFail;
    }

    std::cout << BlindCrack::format(report.value());
    return parcae::cli::kExitOk;
}
