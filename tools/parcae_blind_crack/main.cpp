#include "parcae/run/blind_crack.hpp"

#include "blind_crack_cli.hpp"
#include "cli_io.hpp"

#include <iostream>
#include <string>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help")) {
        BlindCrackCli::print_help();
        return CliIo::kExitOk;
    }

    const std::string data_dir = CliIo::optional_option(args, "--data-dir");
    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return CliIo::kExitUsage;
    }

    StatusOr<BlindCrack::Report> report = BlindCrack::run_all_locked(ctx.value());
    if (!report.ok()) {
        std::cerr << report.status().message() << '\n';
        return CliIo::kExitFail;
    }

    std::cout << BlindCrack::format(report.value());
    return CliIo::kExitOk;
}
