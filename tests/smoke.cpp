#include <parcae/core/core.hpp>
#include <parcae/core/version.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

#include <nlohmann/json.hpp>

#if defined(PARCAE_HAS_CLI_GOLDENS)
#include "parcae_cli_paths.h"
#if !defined(PARCAE_CLI_SEARCH_CYCLE)
#error "PARCAE_CLI_SEARCH_CYCLE required"
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#endif

TEST_CASE("parcae_core version macros are wired", "[smoke][version]") {
    REQUIRE(Version::major == PARCAE_VERSION_MAJOR);
    REQUIRE(Version::minor == PARCAE_VERSION_MINOR);
    REQUIRE(Version::patch == PARCAE_VERSION_PATCH);
}

TEST_CASE("toolkit Version is 0.7.0 (search-engine exit)", "[smoke][version][search]") {
    REQUIRE(Version::major == 0);
    REQUIRE(Version::minor == 7);
    REQUIRE(Version::patch == 0);
    REQUIRE(std::string(PARCAE_VERSION_STRING) == "0.7.0");
}

TEST_CASE("nlohmann_json is available through parcae::core", "[smoke]") {
    const nlohmann::json value = {{"modulus", 29}};
    REQUIRE(value.at("modulus") == 29);
}

#if defined(PARCAE_HAS_CLI_GOLDENS)

namespace {

[[nodiscard]] std::string quote_arg(const std::string& arg) {
    return std::string("\"") + arg + '"';
}

[[nodiscard]] std::pair<int, std::string> run_cli(
    const std::filesystem::path& exe,
    const std::vector<std::string>& args) {
    const auto tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path out_path = tmp / "parcae_m51_status_out.json";
    const std::filesystem::path err_path = tmp / "parcae_m51_status_err.txt";
    const std::filesystem::path script_path = tmp / "parcae_m51_status_run.cmd";

    {
        std::ofstream script(script_path, std::ios::binary);
        REQUIRE(script);
        script << "@echo off\r\n";
        script << quote_arg(exe.string());
        for (const std::string& arg : args) {
            script << ' ' << quote_arg(arg);
        }
        script << " >" << quote_arg(out_path.string()) << " 2>"
               << quote_arg(err_path.string()) << "\r\n";
        script << "exit /B %ERRORLEVEL%\r\n";
    }

    const int exit_code =
        std::system((std::string("cmd /C ") + quote_arg(script_path.string())).c_str());

    std::string stdout_text;
    if (std::filesystem::exists(out_path)) {
        std::ifstream in(out_path, std::ios::binary);
        std::ostringstream buf;
        buf << in.rdbuf();
        stdout_text = buf.str();
    }

    std::error_code ec;
    std::filesystem::remove(out_path, ec);
    std::filesystem::remove(err_path, ec);
    std::filesystem::remove(script_path, ec);
    return {exit_code, stdout_text};
}

}  // namespace

TEST_CASE(
    "parcae-search-cycle --status reports toolkit_version 0.7.0",
    "[smoke][version][search][tool][search_cycle][status]") {
#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif
    REQUIRE(std::string(PARCAE_VERSION_STRING) == "0.7.0");

    const auto [code, out] = run_cli(
        PARCAE_CLI_SEARCH_CYCLE,
        {"--status", "--json", "--data-dir", std::string(PARCAE_TEST_DATA_DIR)});
    REQUIRE(code == 0);
    const nlohmann::json envelope = nlohmann::json::parse(out);
    REQUIRE(envelope.at("ok").get<bool>());
    REQUIRE(envelope.at("tool").get<std::string>() == "search_cycle");
    REQUIRE(envelope.at("result").at("toolkit_version").get<std::string>() == "0.7.0");
    REQUIRE(envelope.at("result").at("toolkit_version").get<std::string>() ==
            PARCAE_VERSION_STRING);
    REQUIRE(envelope.at("result").at("run_ready").get<bool>());
    REQUIRE(envelope.at("result").at("scheduler_ready").get<bool>());
}

#endif  // PARCAE_HAS_CLI_GOLDENS
