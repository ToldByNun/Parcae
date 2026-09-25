#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <parcae/core/core.hpp>
#include <parcae/core/version.hpp>
#include <string>

#if defined(PARCAE_HAS_CLI_GOLDENS)
#include "cli_spawn.hpp"
#include "parcae_cli_paths.h"
#if !defined(PARCAE_CLI_SEARCH_CYCLE) || !defined(PARCAE_CLI_COMPILE)
#error "PARCAE_CLI_SEARCH_CYCLE and PARCAE_CLI_COMPILE required"
#endif
#endif

TEST_CASE("parcae_core version macros are wired", "[smoke][version]") {
    REQUIRE(Version::major == PARCAE_VERSION_MAJOR);
    REQUIRE(Version::minor == PARCAE_VERSION_MINOR);
    REQUIRE(Version::patch == PARCAE_VERSION_PATCH);
}

TEST_CASE("toolkit Version is 0.8.0 (dsl-console exit)", "[smoke][version][search]") {
    REQUIRE(Version::major == 0);
    REQUIRE(Version::minor == 8);
    REQUIRE(Version::patch == 0);
    REQUIRE(std::string(PARCAE_VERSION_STRING) == "0.8.0");
}

TEST_CASE("nlohmann_json is available through parcae::core", "[smoke]") {
    const nlohmann::json value = {{"modulus", 29}};
    REQUIRE(value.at("modulus") == 29);
}

#if defined(PARCAE_HAS_CLI_GOLDENS)

TEST_CASE("parcae-search-cycle --status reports toolkit_version 0.8.0",
          "[smoke][version][search][tool][search_cycle][status]") {
#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif
    REQUIRE(std::string(PARCAE_VERSION_STRING) == "0.8.0");

    const CliSpawnResult run =
        run_cli_capture(PARCAE_CLI_SEARCH_CYCLE,
                        {"--status", "--json", "--data-dir", std::string(PARCAE_TEST_DATA_DIR)},
                        "search_cycle_status");
    REQUIRE(run.exit_code == 0);
    const nlohmann::json envelope = nlohmann::json::parse(run.stdout_text);
    REQUIRE(envelope.at("ok").get<bool>());
    REQUIRE(envelope.at("tool").get<std::string>() == "search_cycle");
    REQUIRE(envelope.at("result").at("toolkit_version").get<std::string>() == "0.8.0");
    REQUIRE(envelope.at("result").at("toolkit_version").get<std::string>() ==
            PARCAE_VERSION_STRING);
    REQUIRE(envelope.at("result").at("run_ready").get<bool>());
    REQUIRE(envelope.at("result").at("scheduler_ready").get<bool>());
}

TEST_CASE("parcae-compile --status reports toolkit_version 0.8.0",
          "[smoke][version][dsl][tool][compile][status]") {
#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif
    REQUIRE(std::string(PARCAE_VERSION_STRING) == "0.8.0");

    const CliSpawnResult run = run_cli_capture(
        PARCAE_CLI_COMPILE, {"--status", "--json", "--data-dir", std::string(PARCAE_TEST_DATA_DIR)},
        "compile_status");
    REQUIRE(run.exit_code == 0);
    const nlohmann::json envelope = nlohmann::json::parse(run.stdout_text);
    REQUIRE(envelope.at("ok").get<bool>());
    REQUIRE(envelope.at("tool").get<std::string>() == "compile");
    REQUIRE(envelope.at("result").at("toolkit_version").get<std::string>() == "0.8.0");
    REQUIRE(envelope.at("result").at("toolkit_version").get<std::string>() ==
            PARCAE_VERSION_STRING);
    REQUIRE(envelope.at("result").contains("pipeline_ready"));
    REQUIRE(envelope.at("result").at("dsl_spec_version").get<std::string>().size() > 0);
}

#endif // PARCAE_HAS_CLI_GOLDENS
