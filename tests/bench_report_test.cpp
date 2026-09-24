#include <catch2/catch_test_macros.hpp>
#include <parcae/bench/bench_formatter.hpp>
#include <parcae/bench/bench_report.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/core/version.hpp>
#include <string>

[[nodiscard]] static BenchReport::Document make_fixture_doc() {
    BenchReport::Document doc(BenchReport::Suite::Slo, "0.9.0-test");
    const bool t1_pass =
        BenchTierSpec::pass_tier(0.95 * BenchTierSpec::t1.estimated_peak, BenchTierSpec::t1.slo_min,
                                 BenchTierSpec::t1.estimated_peak);
    doc.add_row(BenchReport::Row::from_tier_spec(BenchTierSpec::t1, BenchReport::Backend::Cuda,
                                                 0.95 * BenchTierSpec::t1.estimated_peak, 1.0e6,
                                                 0.42, t1_pass));
    doc.add_row(BenchReport::Row::make("A.score_parity", "CPU vs CUDA chi2 digest",
                                       BenchReport::Suite::Accuracy, BenchReport::Backend::Both,
                                       BenchReport::RowStatus::Pass, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0, 0, 0, "epsilon ok"));
    return doc;
}
TEST_CASE("BenchReport suite backend status strings", "[bench][report]") {
    REQUIRE(std::string(BenchReport::suite_str(BenchReport::Suite::Slo)) == "slo");
    REQUIRE(std::string(BenchReport::suite_str(BenchReport::Suite::Accuracy)) == "accuracy");
    REQUIRE(std::string(BenchReport::backend_str(BenchReport::Backend::Cuda)) == "cuda");
    REQUIRE(std::string(BenchReport::status_str(BenchReport::RowStatus::Skipped)) == "skipped");
}

TEST_CASE("BenchReport JSON includes timing fields by default", "[bench][report]") {
    const BenchReport::Document doc = make_fixture_doc();
    const nlohmann::json j = doc.to_json(false);
    REQUIRE(j.at("omit_timing").get<bool>() == false);
    REQUIRE(j.at("suite").get<std::string>() == "slo");
    REQUIRE(j.at("toolkit_version").get<std::string>() == "0.9.0-test");
    REQUIRE(j.at("row_count").get<std::size_t>() == 2u);
    REQUIRE(j.at("rows").at(0).contains("runes_per_sec"));
    REQUIRE(j.at("rows").at(0).contains("keys_per_sec"));
    REQUIRE(j.at("rows").at(0).contains("wall_seconds"));
    REQUIRE(j.at("rows").at(0).contains("percent_peak"));
    REQUIRE(j.at("rows").at(0).at("name").get<std::string>() == "T1");
    REQUIRE(j.at("rows").at(0).at("candidates").get<std::size_t>() == 29u);
    REQUIRE(j.at("rows").at(0).at("repeats").get<std::size_t>() == 64u);
}

TEST_CASE("BenchReport omit_timing drops non-deterministic rate fields", "[bench][report]") {
    const BenchReport::Document doc = make_fixture_doc();
    const nlohmann::json j = doc.to_json(true);
    REQUIRE(j.at("omit_timing").get<bool>() == true);
    REQUIRE_FALSE(j.at("rows").at(0).contains("runes_per_sec"));
    REQUIRE_FALSE(j.at("rows").at(0).contains("keys_per_sec"));
    REQUIRE_FALSE(j.at("rows").at(0).contains("wall_seconds"));
    REQUIRE_FALSE(j.at("rows").at(0).contains("percent_peak"));
    // Structural / SLO fields remain for digest replay.
    REQUIRE(j.at("rows").at(0).contains("target_min"));
    REQUIRE(j.at("rows").at(0).contains("estimated_peak"));
    REQUIRE(j.at("rows").at(0).contains("candidates"));
    REQUIRE(j.at("rows").at(0).contains("pass"));
}

TEST_CASE("BenchReport omit_timing dump is stable across calls", "[bench][report]") {
    const BenchReport::Document doc = make_fixture_doc();
    const std::string a = doc.to_json(true).dump();
    const std::string b = doc.to_json(true).dump();
    REQUIRE(a == b);
    REQUIRE(a.find("runes_per_sec") == std::string::npos);
}

TEST_CASE("BenchReport fail row clears all_pass", "[bench][report]") {
    BenchReport::Document doc(BenchReport::Suite::Slo);
    doc.add_row(BenchReport::Row::from_tier_spec(BenchTierSpec::t3, BenchReport::Backend::Cuda,
                                                 0.1e9, // below SLO
                                                 0.0, 1.0, false));
    REQUIRE_FALSE(doc.all_pass());
    REQUIRE(doc.to_json(true).at("ok").get<bool>() == false);
}

TEST_CASE("BenchReport skipped rows do not fail all_pass", "[bench][report]") {
    BenchReport::Document doc(BenchReport::Suite::Hardware);
    doc.add_row(BenchReport::Row::make(
        "T1", "cpu scaled", BenchReport::Suite::Hardware, BenchReport::Backend::Cpu,
        BenchReport::RowStatus::Skipped, 0.0, 0.0, 0.0, BenchTierSpec::t1.slo_min,
        BenchTierSpec::t1.slo_max, BenchTierSpec::t1.estimated_peak, BenchTierSpec::t1.candidates,
        BenchTierSpec::t1.tokens, BenchTierSpec::t1.repeats, "cuda not built"));
    REQUIRE(doc.all_pass());
}

TEST_CASE("BenchFormatter human table includes tier and PASS", "[bench][report]") {
    const BenchReport::Document doc = make_fixture_doc();
    const std::string human = BenchFormatter::format(doc);
    REQUIRE(human.find("PARCAE — BENCH (slo)") != std::string::npos);
    REQUIRE(human.find("T1") != std::string::npos);
    REQUIRE(human.find("SLO TIERS") != std::string::npos);
    REQUIRE(human.find("ACCURACY") != std::string::npos);
    REQUIRE(human.find("pass") != std::string::npos);
    REQUIRE(human.find("ALL ROWS PASS") != std::string::npos);
    REQUIRE(BenchFormatter::format_rps(15.0e9) == "15.00B");
    REQUIRE(BenchFormatter::format_target(15.0e9, 35.0e9) == "15.00B-35.00B");
    REQUIRE(BenchFormatter::format_target(1.0e9, 0.0) == ">=1.00B");
    REQUIRE(BenchFormatter::format_throughput(3.25e9).find("B runes/s") != std::string::npos);
}

TEST_CASE("BenchReport Document defaults toolkit_version from Version macros", "[bench][report]") {
    BenchReport::Document doc(BenchReport::Suite::All);
    REQUIRE(doc.toolkit_version() == std::string(PARCAE_VERSION_STRING));
}
