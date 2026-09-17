#include "cli_io.hpp"

#include "parcae/tool/api.hpp"
#include "parcae/validate/validation_report.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

void print_help() {
    std::cerr
        << "Usage: parcae-validate --id <fixture_id|path> [--require-locked] [--json]\n"
        << "       parcae-validate --all [--require-locked] [--json]\n"
        << "\n"
        << "  --id <id|path>     One solved fixture id or fixture directory\n"
        << "  --all              All fixtures under data/fixtures/solved/\n"
        << "                     (with --require-locked: locked fixtures only)\n"
        << "  --require-locked   Fail (or skip under --all) non-locked fixtures\n"
        << "  --json             Machine-readable JSON on stdout\n"
        << "  --data-dir         Parcae data/ root\n"
        << "  -h, --help         Show this help\n"
        << "\n"
        << "Exit: 0 all selected passed; 1 validation failure; 2 usage/I/O error\n";
}

[[nodiscard]] nlohmann::json report_to_json(const ValidationReport& report) {
    nlohmann::json checks = nlohmann::json::array();
    for (const ValidationCheck& check : report.checks()) {
        checks.push_back(nlohmann::json{
            {"name", check.name()},
            {"ok", check.ok()},
            {"message", check.message()},
        });
    }
    nlohmann::json root{
        {"ok", report.ok()},
        {"fixture_id", report.fixture_id()},
        {"checks", std::move(checks)},
    };
    if (report.diff_excerpt().has_value()) {
        root["diff_excerpt"] = report.diff_excerpt().value();
    } else {
        root["diff_excerpt"] = nullptr;
    }
    return root;
}

void print_report_human(const ValidationReport& report) {
    std::cout << (report.ok() ? "PASS" : "FAIL") << '\t' << report.fixture_id() << '\n';
    for (const ValidationCheck& check : report.checks()) {
        std::cout << "  " << (check.ok() ? "ok" : "FAIL") << '\t' << check.name() << '\t'
                  << check.message() << '\n';
    }
    if (report.diff_excerpt().has_value()) {
        std::cout << "  diff\t" << report.diff_excerpt().value() << '\n';
    }
}

[[nodiscard]] StatusOr<bool> fixture_is_locked(const std::filesystem::path& dir) {
    const std::filesystem::path manifest = dir / "manifest.json";
    StatusOr<std::string> text = parcae::cli::read_file_utf8(manifest);
    if (!text.ok()) {
        return text.status();
    }
    try {
        const nlohmann::json root = nlohmann::json::parse(text.value());
        if (!root.contains("verification") || !root.at("verification").contains("status")) {
            return false;
        }
        return root.at("verification").at("status").get<std::string>() == "locked";
    } catch (const nlohmann::json::exception& ex) {
        return Status::error(std::string("Invalid manifest: ") + ex.what());
    }
}

[[nodiscard]] StatusOr<std::vector<std::string>> list_solved_fixture_ids(
    const parcae::tool::Context& ctx,
    bool locked_only) {
    const std::filesystem::path solved = ctx.data_root() / "fixtures" / "solved";
    if (!std::filesystem::is_directory(solved)) {
        return Status::error("Missing fixtures/solved under data root");
    }

    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(solved)) {
        if (!entry.is_directory()) {
            continue;
        }
        if (!std::filesystem::is_regular_file(entry.path() / "manifest.json")) {
            continue;
        }
        if (locked_only) {
            StatusOr<bool> locked = fixture_is_locked(entry.path());
            if (!locked.ok()) {
                return locked.status();
            }
            if (!locked.value()) {
                continue;
            }
        }
        ids.push_back(entry.path().filename().string());
    }
    std::sort(ids.begin(), ids.end());
    if (ids.empty()) {
        return Status::error(
            locked_only ? "No locked fixtures found under fixtures/solved"
                        : "No fixtures found under fixtures/solved");
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? kExitUsage : kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const bool require_locked = has_flag(args, "--require-locked");
    const bool all_mode = has_flag(args, "--all");
    const bool has_id = has_flag(args, "--id");
    const std::string data_dir = optional_option(args, "--data-dir");

    if (static_cast<int>(all_mode) + static_cast<int>(has_id) != 1) {
        std::cerr << "Choose exactly one of --id or --all\n";
        print_help();
        return kExitUsage;
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return kExitUsage;
    }

    std::vector<std::string> targets;
    if (all_mode) {
        StatusOr<std::vector<std::string>> ids =
            list_solved_fixture_ids(ctx.value(), /*locked_only=*/require_locked);
        if (!ids.ok()) {
            std::cerr << ids.status().message() << '\n';
            return kExitUsage;
        }
        targets = std::move(ids.value());
    } else {
        StatusOr<std::string> id = require_option(args, "--id");
        if (!id.ok()) {
            std::cerr << id.status().message() << '\n';
            print_help();
            return kExitUsage;
        }
        targets.push_back(id.value());
    }

    nlohmann::json reports = nlohmann::json::array();
    bool all_ok = true;

    for (const std::string& target : targets) {
        const ValidationReport report =
            parcae::tool::validate_fixture(ctx.value(), target, require_locked);
        if (!report.ok()) {
            all_ok = false;
        }
        if (json_mode) {
            reports.push_back(report_to_json(report));
        } else {
            print_report_human(report);
        }
    }

    if (json_mode) {
        std::cout << nlohmann::json{
                         {"ok", all_ok},
                         {"count", reports.size()},
                         {"reports", std::move(reports)},
                     }
                         .dump(2)
                  << '\n';
    } else if (targets.size() > 1) {
        std::cout << (all_ok ? "ALL PASS" : "SOME FAILED") << " (" << targets.size()
                  << " fixtures)\n";
    }

    return all_ok ? kExitOk : kExitFail;
}
