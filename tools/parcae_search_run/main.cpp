#include "cli_io.hpp"

#include "parcae/run/search_run.hpp"
#include "parcae/run/search_run_console.hpp"
#include "parcae/run/search_run_metrics.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

void print_help() {
    std::cerr
        << "Usage: parcae-search-run [--backend cpu|cuda] [--family caesar]\n"
        << "                         [--seed <u32>] [--stream-length <n>]\n"
        << "                         [--repeats <n>] [--score-id <id>]\n"
        << "                         [--no-compare] [--json] [--data-dir <path>]\n"
        << "\n"
        << "AI-style search-run dashboard: throughput (runes/s), sweep scores,\n"
        << "and locked-fixture scorer eval. Timing covers transform+score only\n"
        << "(setup/init excluded). tok_per_sec is intentionally non-deterministic.\n"
        << "\nv0 families: caesar|atbash|atbash_caesar|affine|vigenere\n";
}

[[nodiscard]] nlohmann::json metrics_to_json(const SearchRunMetrics& metrics) {
    nlohmann::json steps = nlohmann::json::array();
    for (const SearchRunStep& step : metrics.steps()) {
        steps.push_back({
            {"step_id", step.step_id()},
            {"transform_id", step.transform_id()},
            {"param_hash", step.param_hash()},
            {"score", step.score()},
        });
    }
    nlohmann::json out = {
        {"transform_id", metrics.transform_id()},
        {"parameters", metrics.parameters_label()},
        {"seed", metrics.seed()},
        {"backend", metrics.backend()},
        {"score_id", metrics.score_id()},
        {"tok_per_sec", metrics.tok_per_sec()},
        {"score_mean", metrics.score_mean()},
        {"score_std", metrics.score_std()},
        {"eval_passed", metrics.eval_passed()},
        {"eval_total", metrics.eval_total()},
        {"eval_set_pass_rate", metrics.eval_set_pass_rate()},
        {"steps", std::move(steps)},
    };
    if (metrics.cpu_cuda_pass().has_value()) {
        out["cpu_cuda_pass"] = metrics.cpu_cuda_pass().value();
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help")) {
        print_help();
        return kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const std::string data_dir = optional_option(args, "--data-dir");

    StatusOr<parcae::tool::Backend> backend = parcae::tool::BackendUtil::from_string(
        optional_option(args, "--backend", "cpu"));
    if (!backend.ok()) {
        std::cerr << backend.status().message() << '\n';
        print_help();
        return kExitUsage;
    }
    Status backend_ok = parcae::tool::BackendUtil::ensure_usable(backend.value());
    if (!backend_ok.ok()) {
        std::cerr << backend_ok.message() << '\n';
        return kExitUsage;
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return kExitUsage;
    }

    SearchRun::Options options;
    options.backend = backend.value();
    options.family = optional_option(args, "--family", "caesar");
    options.score_id = optional_option(args, "--score-id", "chi2_english_gp_v0");
    options.compare_cpu_cuda = !has_flag(args, "--no-compare");

    // CUDA defaults: large stream so fused kernels are not launch-bound.
    if (backend.value() == parcae::tool::Backend::Cuda) {
        if (options.family == "affine") {
            options.stream_length = 1u << 18;  // 256k (812 candidates)
            options.throughput_repeats = 16;
        } else {
            options.stream_length = 1u << 20;  // 1M
            options.throughput_repeats = 64;
        }
    }

    const std::string seed_text = optional_option(args, "--seed");
    if (!seed_text.empty()) {
        try {
            options.seed = static_cast<std::uint32_t>(std::stoul(seed_text));
        } catch (const std::exception&) {
            std::cerr << "Invalid --seed\n";
            return kExitUsage;
        }
    }

    const std::string len_text = optional_option(args, "--stream-length");
    if (!len_text.empty()) {
        try {
            options.stream_length = static_cast<std::size_t>(std::stoull(len_text));
        } catch (const std::exception&) {
            std::cerr << "Invalid --stream-length\n";
            return kExitUsage;
        }
    }

    const std::string reps_text = optional_option(args, "--repeats");
    if (!reps_text.empty()) {
        try {
            options.throughput_repeats = static_cast<std::size_t>(std::stoull(reps_text));
        } catch (const std::exception&) {
            std::cerr << "Invalid --repeats\n";
            return kExitUsage;
        }
    }

    StatusOr<SearchRunMetrics> metrics = SearchRun::run(ctx.value(), options);
    if (!metrics.ok()) {
        std::cerr << metrics.status().message() << '\n';
        return kExitFail;
    }

    if (json_mode) {
        std::cout << metrics_to_json(metrics.value()).dump(2) << '\n';
    } else {
        std::cout << SearchRunConsole::format(metrics.value());
    }

    if (metrics.value().eval_set_pass_rate() < 1.0) {
        return kExitFail;
    }
    if (metrics.value().cpu_cuda_pass().has_value() &&
        !metrics.value().cpu_cuda_pass().value()) {
        return kExitFail;
    }
    return kExitOk;
}
