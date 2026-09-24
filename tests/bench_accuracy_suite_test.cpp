#include <catch2/catch_test_macros.hpp>
#include <parcae/bench/bench_accuracy_suite.hpp>
#include <parcae/bench/bench_formatter.hpp>
#include <parcae/tool/context.hpp>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#define PARCAE_TEST_DATA_DIR ""
#endif

TEST_CASE("BenchAccuracySuite Options defaults", "[bench][accuracy]") {
    const BenchAccuracySuite::Options opts;
    REQUIRE_FALSE(opts.allow_cuda());
    REQUIRE(opts.seed() == 2109016688u);
}

TEST_CASE("BenchAccuracySuite CPU checks pass on locked fixtures", "[bench][accuracy]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    BenchAccuracySuite::Options opts;
    StatusOr<BenchReport::Document> doc = BenchAccuracySuite::run(ctx, opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().suite() == BenchReport::Suite::Accuracy);
    REQUIRE(doc.value().rows().size() == 3u);
    REQUIRE(doc.value().all_pass());

    REQUIRE(doc.value().rows()[0].name() == "A.fixture_eval");
    REQUIRE(doc.value().rows()[1].name() == "A.chi2_sanity");
    REQUIRE(doc.value().rows()[2].name() == "A.oracle_rank");
    REQUIRE(doc.value().rows()[0].backend() == BenchReport::Backend::Cpu);

    const nlohmann::json j = doc.value().to_json(true);
    REQUIRE(j.at("omit_timing").get<bool>());
    REQUIRE(j.at("suite").get<std::string>() == "accuracy");
    REQUIRE(j.at("all_pass").get<bool>());
    REQUIRE_FALSE(j.at("rows").at(0).contains("runes_per_sec"));

    const std::string human = BenchFormatter::format(doc.value());
    REQUIRE(human.find("ACCURACY") != std::string::npos);
    REQUIRE(human.find("A.fixture_eval") != std::string::npos);
}

TEST_CASE("BenchAccuracySuite CUDA skipped rows when allow_cuda without device",
          "[bench][accuracy]") {
    const Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    BenchAccuracySuite::Options opts;
    opts.set_allow_cuda(true);
    StatusOr<BenchReport::Document> doc = BenchAccuracySuite::run(ctx, opts);
    REQUIRE(doc.ok());
#if defined(PARCAE_HAS_CUDA)
    // With CUDA build + available GPU, expect fused/planted rows present.
    REQUIRE(doc.value().rows().size() >= 3u);
#else
    REQUIRE(doc.value().rows().size() == 6u);
    REQUIRE(doc.value().rows()[3].name() == "A.fused_parity");
    REQUIRE(doc.value().rows()[3].status() == BenchReport::RowStatus::Skipped);
    REQUIRE(doc.value().rows()[4].name() == "A.planted_caesar");
    REQUIRE(doc.value().rows()[5].name() == "A.planted_bigram");
    // Skipped does not fail all_pass.
    REQUIRE(doc.value().all_pass());
#endif
}
