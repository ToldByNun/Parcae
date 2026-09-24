#include "bench_cli.hpp"
#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/bench/bench_accuracy_suite.hpp"
#include "parcae/bench/bench_formatter.hpp"
#include "parcae/bench/bench_slo_suite.hpp"
#include "parcae/core/version.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "bench";

[[nodiscard]] int fail(
    bool json_mode,
    const std::optional<std::string>& backend,
    ToolErrorCode code,
    std::string message,
    int plain_exit) {
    if (json_mode) {
        return ToolCliJson::err(kTool, backend, code, std::move(message));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

[[nodiscard]] bool is_known_suite(std::string_view suite) {
    return suite == "slo" || suite == "accuracy" || suite == "hardware" || suite == "probe" ||
           suite == "all";
}

[[nodiscard]] int emit_doc(
    bool json_mode,
    bool omit_timing,
    const std::optional<std::string>& backend,
    const BenchReport::Document& doc) {
    if (json_mode) {
        return ToolCliJson::ok(kTool, backend, doc.to_json(omit_timing));
    }
    std::cout << BenchFormatter::format(doc);
    return doc.all_pass() ? parcae::cli::kExitOk : parcae::cli::kExitFail;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help")) {
        BenchCli::print_help();
        return kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const bool omit_timing = has_flag(args, "--omit-timing");
    if (omit_timing && !json_mode) {
        return fail(
            false,
            std::nullopt,
            ToolErrorCode::Usage,
            "--omit-timing requires --json",
            kExitUsage);
    }

    const std::string data_dir = optional_option(args, "--data-dir");
    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, std::nullopt, ToolErrorCode::Io, ctx.status().message(), kExitUsage);
    }

    const bool want_status = has_flag(args, "--status");
    const std::string suite = optional_option(args, "--suite", want_status ? "" : "slo");

    if (want_status) {
        if (has_flag(args, "--suite") || has_flag(args, "--extended") ||
            has_flag(args, "--allow-cuda")) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "--status does not take suite-run flags",
                kExitUsage);
        }
        nlohmann::json result = BenchCli::status_result(ctx.value().data_root().string());
        if (json_mode) {
            return ToolCliJson::ok(kTool, std::nullopt, std::move(result));
        }
        std::cout << "parcae-bench status\n"
                  << "  toolkit_version: " << result.at("toolkit_version").get<std::string>()
                  << '\n'
                  << "  cuda_built:      "
                  << (result.at("cuda_built").get<bool>() ? "true" : "false") << '\n'
                  << "  suites.slo:      ready\n"
                  << "  suites.accuracy: ready\n"
                  << "  data_dir:        " << result.at("data_dir").get<std::string>() << '\n'
                  << "  " << result.at("message").get<std::string>() << '\n';
        return kExitOk;
    }

    if (suite.empty() || !is_known_suite(suite)) {
        BenchCli::print_help();
        return fail(
            json_mode,
            std::nullopt,
            ToolErrorCode::Usage,
            "unknown or missing --suite (use slo|accuracy|hardware|probe|all)",
            kExitUsage);
    }

    if (suite == "accuracy") {
        if (has_flag(args, "--extended")) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "--extended applies only to --suite slo",
                kExitUsage);
        }
        BenchAccuracySuite::Options opts;
        opts.set_allow_cuda(has_flag(args, "--allow-cuda"));
        StatusOr<BenchReport::Document> doc = BenchAccuracySuite::run(ctx.value(), opts);
        if (!doc.ok()) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Internal,
                doc.status().message(),
                kExitFail);
        }
        std::optional<std::string> backend = opts.allow_cuda() ? std::optional<std::string>("cuda")
                                                               : std::optional<std::string>("cpu");
        return emit_doc(json_mode, omit_timing, backend, doc.value());
    }

    if (suite != "slo") {
        return fail(
            json_mode,
            std::nullopt,
            ToolErrorCode::Usage,
            "suite '" + suite + "' is not implemented yet (supports slo|accuracy)",
            kExitUsage);
    }

    if (!has_flag(args, "--allow-cuda")) {
        return fail(
            json_mode,
            std::string("cuda"),
            ToolErrorCode::Policy,
            "BenchSloSuite requires --allow-cuda",
            kExitUsage);
    }

    Status backend_ok =
        parcae::tool::BackendUtil::ensure_usable(parcae::tool::Backend::Cuda);
    if (!backend_ok.ok()) {
        return fail(
            json_mode,
            std::string("cuda"),
            ToolErrorCode::NotBuilt,
            backend_ok.message(),
            kExitUsage);
    }

    StatusOr<ExpectedFrequencyTable> freqs = ctx.value().load_english_gp_expected();
    if (!freqs.ok()) {
        return fail(
            json_mode,
            std::string("cuda"),
            ToolErrorCode::Io,
            freqs.status().message(),
            kExitFail);
    }

    BenchSloSuite::Options opts(has_flag(args, "--extended"));
    StatusOr<BenchReport::Document> doc = BenchSloSuite::run(freqs.value(), opts);
    if (!doc.ok()) {
        return fail(
            json_mode,
            std::string("cuda"),
            ToolErrorCode::Internal,
            doc.status().message(),
            kExitFail);
    }
    return emit_doc(json_mode, omit_timing, std::string("cuda"), doc.value());
}
