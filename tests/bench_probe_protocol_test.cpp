#include <parcae/bench/bench_config.hpp>
#include <parcae/bench/bench_formatter.hpp>
#include <parcae/bench/bench_probe_protocol.hpp>
#include <parcae/bench/bench_probe_runner.hpp>
#include <parcae/bench/bench_tier_spec.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

[[nodiscard]] nlohmann::json valid_probe_json(std::string_view tier) {
    const BenchTierSpec::Tier* spec = BenchTierSpec::find_primary(tier);
    REQUIRE(spec != nullptr);
    return nlohmann::json{
        {"probe_schema_version", "1.0.0"},
        {"tool_id", "mock-probe"},
        {"tier", std::string(tier)},
        {"runes_per_sec", 1.25e9},
        {"keys_per_sec", 4.3e7},
        {"wall_seconds", 0.412},
        {"backend", "cpu"},
        {"accuracy",
         {{"oracle_cracked", true}, {"top_rank", 1}, {"notes", ""}}},
        {"config",
         {{"C", spec->candidates}, {"T", spec->tokens}, {"reps", spec->repeats}}},
    };
}

}  // namespace

TEST_CASE("BenchProbeProtocol accepts valid 1.0.0 payload", "[bench][probe]") {
    const std::string text = valid_probe_json("T1").dump();
    StatusOr<BenchProbeProtocol::Result> parsed = BenchProbeProtocol::parse(text);
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value().tool_id == "mock-probe");
    REQUIRE(parsed.value().tier == "T1");
    REQUIRE(parsed.value().backend == BenchReport::Backend::Cpu);
    REQUIRE(parsed.value().config.candidates == BenchTierSpec::t1.candidates);
    REQUIRE(parsed.value().accuracy.oracle_cracked);
    REQUIRE(parsed.value().accuracy.top_rank == 1);
}

TEST_CASE("BenchProbeProtocol rejects wrong version / missing / non-finite", "[bench][probe]") {
    {
        nlohmann::json j = valid_probe_json("T1");
        j["probe_schema_version"] = "0.9.0";
        REQUIRE_FALSE(BenchProbeProtocol::parse(j.dump()).ok());
    }
    {
        nlohmann::json j = valid_probe_json("T1");
        j.erase("tool_id");
        REQUIRE_FALSE(BenchProbeProtocol::parse(j.dump()).ok());
    }
    {
        nlohmann::json j = valid_probe_json("T1");
        j["runes_per_sec"] = std::numeric_limits<double>::infinity();
        REQUIRE_FALSE(BenchProbeProtocol::parse(j.dump()).ok());
    }
    {
        REQUIRE_FALSE(BenchProbeProtocol::parse("not-json").ok());
    }
    {
        nlohmann::json j = valid_probe_json("T1");
        j["backend"] = "gpu";
        REQUIRE_FALSE(BenchProbeProtocol::parse(j.dump()).ok());
    }
}

TEST_CASE("BenchConfig parse_probe_tiers and timeout", "[bench][probe]") {
    StatusOr<std::vector<std::string>> tiers =
        BenchConfig::parse_probe_tiers(" T1 , T3 ");
    REQUIRE(tiers.ok());
    REQUIRE(tiers.value().size() == 2u);
    REQUIRE(tiers.value()[0] == "T1");
    REQUIRE(tiers.value()[1] == "T3");
    REQUIRE_FALSE(BenchConfig::parse_probe_tiers("T9").ok());

    StatusOr<std::uint32_t> ms = BenchConfig::parse_timeout_ms("120000");
    REQUIRE(ms.ok());
    REQUIRE(ms.value() == 120000u);
    REQUIRE_FALSE(BenchConfig::parse_timeout_ms("0").ok());
    REQUIRE_FALSE(BenchConfig::parse_timeout_ms("abc").ok());

    const BenchConfig defaults;
    REQUIRE(defaults.probe_timeout_ms() == BenchConfig::default_probe_timeout_ms);
    REQUIRE(defaults.probe_tiers().size() == 3u);
}

TEST_CASE("BenchProbeRunner substitute_tier", "[bench][probe]") {
    REQUIRE(
        BenchProbeRunner::substitute_tier("tool --tier {tier} --json", "T2") ==
        "tool --tier T2 --json");
    REQUIRE(
        BenchProbeRunner::substitute_tier("{tier}-{tier}", "T1") == "T1-T1");
}

TEST_CASE("BenchProbeRunner spawn type/cat JSON file", "[bench][probe]") {
    const auto tmp = std::filesystem::temp_directory_path();
    const auto stamp = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const std::filesystem::path t1 = tmp / ("parcae_probe_" + stamp + "_T1.json");
    {
        std::ofstream out(t1, std::ios::binary);
        REQUIRE(out);
        out << valid_probe_json("T1").dump();
    }

#ifdef _WIN32
    const std::string cmd = "type \"" + t1.string() + "\"";
#else
    const std::string cmd = "cat \"" + t1.string() + "\"";
#endif

    // Single-tier template without {tier} — file is fixed T1.
    BenchConfig cfg;
    cfg.set_probe_cmd(cmd);
    cfg.set_probe_tiers({"T1"});
    cfg.set_probe_timeout_ms(30000);
    cfg.set_compare_builtin(true);

    StatusOr<BenchReport::Document> doc = BenchProbeRunner::run(cfg);
    std::error_code ec;
    std::filesystem::remove(t1, ec);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().suite() == BenchReport::Suite::Probe);
    REQUIRE(doc.value().rows().size() == 1u);
    REQUIRE(doc.value().rows()[0].status() == BenchReport::RowStatus::Pass);
    REQUIRE(doc.value().rows()[0].name() == "T1");
    REQUIRE(doc.value().rows()[0].detail().find("compare_builtin: config_ok") !=
            std::string::npos);
    REQUIRE(doc.value().all_pass());

    const std::string human = BenchFormatter::format(doc.value());
    REQUIRE(human.find("PROBE") != std::string::npos);
}

TEST_CASE("BenchProbeRunner timeout kills long command", "[bench][probe]") {
#ifdef _WIN32
    const std::string cmd = "ping -n 8 127.0.0.1 >NUL";
#else
    const std::string cmd = "sleep 8";
#endif
    StatusOr<BenchProbeRunner::Capture> cap =
        BenchProbeRunner::spawn_with_timeout(cmd, 200);
    REQUIRE(cap.ok());
    REQUIRE(cap.value().timed_out);
}
