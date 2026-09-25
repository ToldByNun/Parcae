#include "cli_spawn.hpp"
#include "parcae_cli_paths.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <parcae/tool/tool_response.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef PARCAE_GOLDENS_TOOL_JSON_DIR
#error "PARCAE_GOLDENS_TOOL_JSON_DIR must be defined"
#endif

#if !defined(PARCAE_CLI_TOKENIZE) || !defined(PARCAE_CLI_DECODE) || !defined(PARCAE_CLI_SCORE) ||  \
    !defined(PARCAE_CLI_VALIDATE) || !defined(PARCAE_CLI_SEARCH_RUN) ||                            \
    !defined(PARCAE_CLI_SEARCH_CYCLE) || !defined(PARCAE_CLI_GENERATE) ||                          \
    !defined(PARCAE_CLI_RANK) || !defined(PARCAE_CLI_HYPOTHESIS) || !defined(PARCAE_TEST_DATA_DIR)
#error "CLI golden tests require PARCAE_CLI_* and PARCAE_TEST_DATA_DIR"
#endif

namespace {

[[nodiscard]] std::filesystem::path goldens_dir() {
    return std::filesystem::path(PARCAE_GOLDENS_TOOL_JSON_DIR);
}

[[nodiscard]] std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

[[nodiscard]] nlohmann::json load_golden(std::string_view name) {
    const std::filesystem::path path = goldens_dir() / name;
    StatusOr<nlohmann::json> parsed = ToolResponse::parse(read_file(path));
    INFO(path.string());
    REQUIRE(parsed.ok());
    return parsed.value();
}

[[nodiscard]] CliSpawnResult run_cli(const std::filesystem::path& exe,
                                     const std::vector<std::string>& args) {
    const CliSpawnResult run = run_cli_capture(exe, args, "tool_json_golden");
    INFO(run.stderr_text);
    INFO(run.stdout_text);
    return run;
}

void expect_cli_matches_golden(const std::filesystem::path& exe,
                               const std::vector<std::string>& args, std::string_view golden_name,
                               int expected_exit) {
    const CliSpawnResult run = run_cli(exe, args);
    REQUIRE(run.exit_code == expected_exit);

    StatusOr<nlohmann::json> actual = ToolResponse::parse(run.stdout_text);
    REQUIRE(actual.ok());
    const nlohmann::json expected = load_golden(golden_name);
    REQUIRE(actual.value() == expected);
}

/// Status JSON embeds absolute `data_dir` and build-dependent `cuda_built`.
void expect_search_cycle_status_matches_golden(const std::vector<std::string>& args) {
    const CliSpawnResult run = run_cli(PARCAE_CLI_SEARCH_CYCLE, args);
    REQUIRE(run.exit_code == 0);

    StatusOr<nlohmann::json> actual = ToolResponse::parse(run.stdout_text);
    REQUIRE(actual.ok());
    nlohmann::json expected = load_golden("search_cycle_status.json");

    REQUIRE(actual.value().at("ok").get<bool>());
    REQUIRE(actual.value().at("tool").get<std::string>() == "search_cycle");
    REQUIRE(actual.value().at("schema").get<std::string>() == "parcae.tool_response.v0");

    nlohmann::json& actual_result = actual.value().at("result");
    nlohmann::json& expected_result = expected.at("result");
    actual_result.erase("data_dir");
    expected_result.erase("data_dir");
    actual_result.erase("cuda_built");
    expected_result.erase("cuda_built");
    REQUIRE(actual.value() == expected);
}

[[nodiscard]] std::vector<std::string> with_data_dir(std::vector<std::string> args) {
    args.insert(args.begin(), {std::string("--data-dir"), std::string(PARCAE_TEST_DATA_DIR)});
    return args;
}

} // namespace

TEST_CASE("CLI JSON golden: tokenize ok", "[tool][golden][cli]") {
    const std::filesystem::path input = goldens_dir() / "inputs" / "mini-runes.txt";
    expect_cli_matches_golden(PARCAE_CLI_TOKENIZE, with_data_dir({"--json", input.string()}),
                              "tokenize_ok.json", 0);
}

TEST_CASE("CLI JSON golden: tokenize usage failure", "[tool][golden][cli]") {
    expect_cli_matches_golden(PARCAE_CLI_TOKENIZE, with_data_dir({"--json"}), "tokenize_usage.json",
                              2);
}

TEST_CASE("CLI JSON golden: score list catalog", "[tool][golden][cli]") {
    expect_cli_matches_golden(PARCAE_CLI_SCORE, with_data_dir({"--list", "--json"}),
                              "score_list.json", 0);
}

TEST_CASE("CLI JSON golden: score ic_mod29 indices", "[tool][golden][cli]") {
    const std::filesystem::path input =
        std::filesystem::path(PARCAE_TEST_DATA_DIR) / "fixtures" / "cli" / "score-indices.txt";
    expect_cli_matches_golden(
        PARCAE_CLI_SCORE,
        with_data_dir({"--score-id", "ic_mod29", "--indices", "--input", input.string(), "--json"}),
        "score_ic_indices.json", 0);
}

TEST_CASE("CLI JSON golden: decode caesar", "[tool][golden][cli]") {
    const std::filesystem::path input = goldens_dir() / "inputs" / "mini-runes.txt";
    expect_cli_matches_golden(PARCAE_CLI_DECODE,
                              with_data_dir({"--json", "--input", input.string(), "--transform-id",
                                             "caesar", "--shift", "3", "--direction", "decrypt"}),
                              "decode_caesar.json", 0);
}

TEST_CASE("CLI JSON golden: decode caesar rebuild-text", "[tool][golden][cli]") {
    const std::filesystem::path input = goldens_dir() / "inputs" / "mini-runes.txt";
    expect_cli_matches_golden(
        PARCAE_CLI_DECODE,
        with_data_dir({"--json", "--rebuild-text", "--input", input.string(), "--transform-id",
                       "caesar", "--shift", "3", "--direction", "decrypt"}),
        "decode_caesar_rebuild.json", 0);
}

TEST_CASE("CLI JSON golden: validate a-warning", "[tool][golden][cli]") {
    expect_cli_matches_golden(PARCAE_CLI_VALIDATE,
                              with_data_dir({"--id", "a-warning", "--require-locked", "--json"}),
                              "validate_a_warning.json", 0);
}

TEST_CASE("CLI JSON golden: search-run omit-timing", "[tool][golden][cli]") {
    expect_cli_matches_golden(
        PARCAE_CLI_SEARCH_RUN,
        with_data_dir({"--backend", "cpu", "--seed", "2109016688", "--stream-length", "256",
                       "--repeats", "2", "--json", "--omit-timing"}),
        "search_run_omit_timing.json", 0);
}

TEST_CASE("CLI JSON golden: search-cycle usage failure", "[tool][golden][cli][search_cycle]") {
    expect_cli_matches_golden(PARCAE_CLI_SEARCH_CYCLE, with_data_dir({"--json"}),
                              "search_cycle_usage.json", 2);
}

TEST_CASE("CLI JSON golden: search-cycle status", "[tool][golden][cli][search_cycle]") {
    expect_search_cycle_status_matches_golden(with_data_dir({"--status", "--json"}));
}

TEST_CASE("CLI generate+rank a-warning atbash via chi2 (no plaintext)",
          "[tool][generate][rank][a-warning][cli]") {
    const std::filesystem::path cipher = std::filesystem::path(PARCAE_TEST_DATA_DIR) / "fixtures" /
                                         "solved" / "a-warning" / "ciphertext.txt";

    const CliSpawnResult generated_run =
        run_cli(PARCAE_CLI_GENERATE, with_data_dir({"--generator-id", "gen_atbash", "--runes",
                                                    "--input", cipher.string(), "--json"}));
    REQUIRE(generated_run.exit_code == 0);
    StatusOr<nlohmann::json> generated = ToolResponse::parse(generated_run.stdout_text);
    REQUIRE(generated.ok());
    REQUIRE(generated.value().at("ok").get<bool>());
    REQUIRE(generated.value().at("tool").get<std::string>() == "generate");
    REQUIRE(generated.value().at("result").at("count").get<std::size_t>() == 1);
    REQUIRE(generated.value()
                .at("result")
                .at("candidates")
                .at(0)
                .at("candidate_id")
                .get<std::string>() == "atbash");

    const auto tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path candidates_path = tmp / "parcae_a_warning_candidates.json";
    {
        std::ofstream out(candidates_path, std::ios::binary);
        REQUIRE(out);
        out << generated_run.stdout_text;
    }

    const CliSpawnResult ranked_run = run_cli(
        PARCAE_CLI_RANK, with_data_dir({"--candidates", candidates_path.string(), "--score-id",
                                        "chi2_english_gp_v0", "--k", "1", "--json"}));
    std::error_code ec;
    std::filesystem::remove(candidates_path, ec);

    REQUIRE(ranked_run.exit_code == 0);
    StatusOr<nlohmann::json> ranked = ToolResponse::parse(ranked_run.stdout_text);
    REQUIRE(ranked.ok());
    REQUIRE(ranked.value().at("ok").get<bool>());
    REQUIRE(ranked.value().at("tool").get<std::string>() == "rank");
    REQUIRE(ranked.value().at("result").at("order").get<std::string>() == "asc");
    REQUIRE(ranked.value().at("result").at("hits").size() == 1);
    REQUIRE(ranked.value().at("result").at("hits").at(0).at("candidate_id").get<std::string>() ==
            "atbash");
    const std::string latin =
        ranked.value().at("result").at("hits").at(0).at("latin").get<std::string>();
    REQUIRE(latin.rfind("AWARNING", 0) == 0);
    // Contract: unary chi2 path — no reference leaked into the rank payload.
    REQUIRE_FALSE(ranked.value().at("result").contains("reference"));
}
