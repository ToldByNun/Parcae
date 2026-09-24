#include "bench_cli.hpp"
#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/bench/bench_accuracy_suite.hpp"
#include "parcae/bench/bench_config.hpp"
#include "parcae/bench/bench_formatter.hpp"
#include "parcae/bench/bench_hardware_suite.hpp"
#include "parcae/bench/bench_probe_runner.hpp"
#include "parcae/bench/bench_slo_suite.hpp"
#include "parcae/core/version.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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
    return doc.all_pass() ? CliIo::kExitOk : CliIo::kExitFail;
}

[[nodiscard]] void append_rows(BenchReport::Document& dest, const BenchReport::Document& src) {
    for (const BenchReport::Row& row : src.rows()) {
        dest.add_row(row);
    }
}

[[nodiscard]] StatusOr<BenchConfig> load_probe_config(const std::vector<std::string>& args) {
    BenchConfig cfg;
    const std::string cmd = CliIo::optional_option(args, "--probe-cmd");
    if (!cmd.empty()) {
        cfg.set_probe_cmd(cmd);
    }
    const std::string tiers_raw = CliIo::optional_option(args, "--probe-tiers");
    if (!tiers_raw.empty()) {
        StatusOr<std::vector<std::string>> tiers = BenchConfig::parse_probe_tiers(tiers_raw);
        if (!tiers.ok()) {
            return tiers.status();
        }
        cfg.set_probe_tiers(std::move(tiers.value()));
    }
    const std::string timeout_raw = CliIo::optional_option(args, "--probe-timeout-ms");
    if (!timeout_raw.empty()) {
        StatusOr<std::uint32_t> ms = BenchConfig::parse_timeout_ms(timeout_raw);
        if (!ms.ok()) {
            return ms.status();
        }
        cfg.set_probe_timeout_ms(ms.value());
    }
    cfg.set_compare_builtin(CliIo::has_flag(args, "--compare-builtin"));
    return cfg;
}

[[nodiscard]] StatusOr<BenchReport::Document> run_accuracy(
    const Context& ctx, const std::vector<std::string>& args) {
    BenchAccuracySuite::Options opts;
    opts.set_allow_cuda(CliIo::has_flag(args, "--allow-cuda"));
    return BenchAccuracySuite::run(ctx, opts);
}

[[nodiscard]] StatusOr<BenchReport::Document> run_slo(
    const Context& ctx, const std::vector<std::string>& args) {
    if (!CliIo::has_flag(args, "--allow-cuda")) {
        return Status::error("BenchSloSuite requires --allow-cuda");
    }
    Status backend_ok =
        BackendUtil::ensure_usable(Backend::Cuda);
    if (!backend_ok.ok()) {
        return backend_ok;
    }
    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    if (!freqs.ok()) {
        return freqs.status();
    }
    BenchSloSuite::Options opts(CliIo::has_flag(args, "--extended"));
    return BenchSloSuite::run(freqs.value(), opts);
}

[[nodiscard]] StatusOr<BenchReport::Document> run_hardware(
    const Context& ctx, const std::vector<std::string>& args) {
    BenchHardwareSuite::Options opts;
    const std::string backend_arg = CliIo::optional_option(args, "--backend", "both");
    StatusOr<BenchHardwareSuite::BackendSelect> backend =
        BenchHardwareSuite::parse_backend(backend_arg);
    if (!backend.ok()) {
        return backend.status();
    }
    opts.set_backend(backend.value());
    opts.set_allow_cuda(CliIo::has_flag(args, "--allow-cuda"));
    opts.set_require_cuda(CliIo::has_flag(args, "--require-cuda"));
    opts.set_allow_skip(CliIo::has_flag(args, "--allow-skip"));
    opts.set_cpu_full(CliIo::has_flag(args, "--cpu-full"));
    return BenchHardwareSuite::run(ctx, opts);
}

[[nodiscard]] StatusOr<BenchReport::Document> run_probe(const std::vector<std::string>& args) {
    StatusOr<BenchConfig> cfg = load_probe_config(args);
    if (!cfg.ok()) {
        return cfg.status();
    }
    if (!cfg.value().has_probe_cmd()) {
        return Status::error("BenchProbeRunner: --probe-cmd is required for --suite probe");
    }
    return BenchProbeRunner::run(cfg.value());
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help")) {
        BenchCli::print_help();
        return CliIo::kExitOk;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");
    const bool omit_timing = CliIo::has_flag(args, "--omit-timing");
    if (omit_timing && !json_mode) {
        return fail(
            false,
            std::nullopt,
            ToolErrorCode::Usage,
            "--omit-timing requires --json",
            CliIo::kExitUsage);
    }

    const std::string data_dir = CliIo::optional_option(args, "--data-dir");
    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, std::nullopt, ToolErrorCode::Io, ctx.status().message(), CliIo::kExitUsage);
    }

    const bool want_status = CliIo::has_flag(args, "--status");
    const std::string suite = CliIo::optional_option(args, "--suite", want_status ? "" : "slo");

    if (want_status) {
        if (CliIo::has_flag(args, "--suite") || CliIo::has_flag(args, "--extended") ||
            CliIo::has_flag(args, "--allow-cuda") || CliIo::has_flag(args, "--require-cuda") ||
            CliIo::has_flag(args, "--allow-skip") || CliIo::has_flag(args, "--cpu-full") ||
            CliIo::has_flag(args, "--backend") || CliIo::has_flag(args, "--probe-cmd") ||
            CliIo::has_flag(args, "--probe-tiers") || CliIo::has_flag(args, "--probe-timeout-ms") ||
            CliIo::has_flag(args, "--compare-builtin")) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "--status does not take suite-run flags",
                CliIo::kExitUsage);
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
                  << "  suites.hardware: ready\n"
                  << "  suites.probe:    ready\n"
                  << "  suites.all:      ready\n"
                  << "  data_dir:        " << result.at("data_dir").get<std::string>() << '\n'
                  << "  " << result.at("message").get<std::string>() << '\n';
        return CliIo::kExitOk;
    }

    if (suite.empty() || !is_known_suite(suite)) {
        BenchCli::print_help();
        return fail(
            json_mode,
            std::nullopt,
            ToolErrorCode::Usage,
            "unknown or missing --suite (use slo|accuracy|hardware|probe|all)",
            CliIo::kExitUsage);
    }

    if (suite == "accuracy") {
        if (CliIo::has_flag(args, "--extended") || CliIo::has_flag(args, "--backend") ||
            CliIo::has_flag(args, "--require-cuda") || CliIo::has_flag(args, "--allow-skip") ||
            CliIo::has_flag(args, "--cpu-full") || CliIo::has_flag(args, "--probe-cmd")) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "accuracy does not take slo/hardware/probe-only flags",
                CliIo::kExitUsage);
        }
        StatusOr<BenchReport::Document> doc = run_accuracy(ctx.value(), args);
        if (!doc.ok()) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Internal,
                doc.status().message(),
                CliIo::kExitFail);
        }
        std::optional<std::string> backend =
            CliIo::has_flag(args, "--allow-cuda") ? std::optional<std::string>("cuda")
                                           : std::optional<std::string>("cpu");
        return emit_doc(json_mode, omit_timing, backend, doc.value());
    }

    if (suite == "hardware") {
        if (CliIo::has_flag(args, "--extended") || CliIo::has_flag(args, "--probe-cmd")) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "hardware does not take --extended/--probe-cmd",
                CliIo::kExitUsage);
        }
        StatusOr<BenchReport::Document> doc = run_hardware(ctx.value(), args);
        if (!doc.ok()) {
            const bool policy = doc.status().message().find("require-cuda") != std::string::npos ||
                                doc.status().message().find("CUDA unavailable") != std::string::npos ||
                                doc.status().message().find("requires --allow-cuda") !=
                                    std::string::npos;
            return fail(
                json_mode,
                std::nullopt,
                policy ? ToolErrorCode::Policy : ToolErrorCode::Internal,
                doc.status().message(),
                policy ? CliIo::kExitUsage : CliIo::kExitFail);
        }
        return emit_doc(json_mode, omit_timing, std::string("both"), doc.value());
    }

    if (suite == "probe") {
        if (CliIo::has_flag(args, "--extended")) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Usage,
                "--extended applies only to --suite slo",
                CliIo::kExitUsage);
        }
        StatusOr<BenchReport::Document> doc = run_probe(args);
        if (!doc.ok()) {
            const bool usage = doc.status().message().find("--probe-cmd") != std::string::npos ||
                               doc.status().message().find("unknown probe tier") !=
                                   std::string::npos ||
                               doc.status().message().find("timeout-ms") != std::string::npos;
            return fail(
                json_mode,
                std::nullopt,
                usage ? ToolErrorCode::Usage : ToolErrorCode::Internal,
                doc.status().message(),
                usage ? CliIo::kExitUsage : CliIo::kExitFail);
        }
        return emit_doc(json_mode, omit_timing, std::string("probe"), doc.value());
    }

    if (suite == "all") {
        BenchReport::Document merged(BenchReport::Suite::All);

        StatusOr<BenchReport::Document> accuracy = run_accuracy(ctx.value(), args);
        if (!accuracy.ok()) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Internal,
                accuracy.status().message(),
                CliIo::kExitFail);
        }
        append_rows(merged, accuracy.value());

        if (CliIo::has_flag(args, "--allow-cuda")) {
            StatusOr<BenchReport::Document> slo = run_slo(ctx.value(), args);
            if (!slo.ok()) {
                // Soft-skip SLO when CUDA unavailable under --suite all.
                if (slo.status().message().find("CUDA") != std::string::npos ||
                    slo.status().message().find("cuda") != std::string::npos ||
                    slo.status().message().find("not built") != std::string::npos) {
                    // omit SLO rows
                } else {
                    return fail(
                        json_mode,
                        std::string("cuda"),
                        ToolErrorCode::Internal,
                        slo.status().message(),
                        CliIo::kExitFail);
                }
            } else {
                append_rows(merged, slo.value());
            }
        }

        StatusOr<BenchReport::Document> hardware = run_hardware(ctx.value(), args);
        if (!hardware.ok()) {
            return fail(
                json_mode,
                std::nullopt,
                ToolErrorCode::Internal,
                hardware.status().message(),
                CliIo::kExitFail);
        }
        append_rows(merged, hardware.value());

        if (CliIo::has_flag(args, "--probe-cmd")) {
            StatusOr<BenchReport::Document> probe = run_probe(args);
            if (!probe.ok()) {
                return fail(
                    json_mode,
                    std::nullopt,
                    ToolErrorCode::Internal,
                    probe.status().message(),
                    CliIo::kExitFail);
            }
            append_rows(merged, probe.value());
        }

        merged.recompute_all_pass();
        return emit_doc(json_mode, omit_timing, std::string("all"), merged);
    }

    // suite == slo
    if (CliIo::has_flag(args, "--probe-cmd") || CliIo::has_flag(args, "--backend") ||
        CliIo::has_flag(args, "--cpu-full")) {
        return fail(
            json_mode,
            std::nullopt,
            ToolErrorCode::Usage,
            "slo does not take --probe-cmd/--backend/--cpu-full",
            CliIo::kExitUsage);
    }

    StatusOr<BenchReport::Document> doc = run_slo(ctx.value(), args);
    if (!doc.ok()) {
        const bool policy = doc.status().message().find("--allow-cuda") != std::string::npos ||
                            doc.status().message().find("CUDA") != std::string::npos ||
                            doc.status().message().find("cuda") != std::string::npos;
        return fail(
            json_mode,
            std::string("cuda"),
            policy ? ToolErrorCode::Policy : ToolErrorCode::Internal,
            doc.status().message(),
            policy ? CliIo::kExitUsage : CliIo::kExitFail);
    }
    return emit_doc(json_mode, omit_timing, std::string("cuda"), doc.value());
}
