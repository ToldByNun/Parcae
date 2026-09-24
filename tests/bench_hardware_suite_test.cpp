#include <parcae/bench/bench_formatter.hpp>
#include <parcae/bench/bench_hardware_suite.hpp>
#include <parcae/tool/context.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#define PARCAE_TEST_DATA_DIR ""
#endif

TEST_CASE("BenchHardwareSuite Options defaults", "[bench][hardware]") {
    const BenchHardwareSuite::Options opts;
    REQUIRE_FALSE(opts.allow_cuda());
    REQUIRE_FALSE(opts.require_cuda());
    REQUIRE_FALSE(opts.allow_skip());
    REQUIRE_FALSE(opts.cpu_full());
    REQUIRE(opts.backend() == BenchHardwareSuite::BackendSelect::Both);
    REQUIRE(opts.seed() == 0x48415244u);

    REQUIRE(BenchHardwareSuite::parse_backend("cpu").ok());
    REQUIRE(
        BenchHardwareSuite::parse_backend("cpu").value() ==
        BenchHardwareSuite::BackendSelect::Cpu);
    REQUIRE(
        BenchHardwareSuite::parse_backend("both").value() ==
        BenchHardwareSuite::BackendSelect::Both);
    REQUIRE_FALSE(BenchHardwareSuite::parse_backend("gpu").ok());
}

TEST_CASE("BenchHardwareSuite CPU smoke rows for T1–T3", "[bench][hardware]") {
    const parcae::tool::Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    BenchHardwareSuite::Options opts;
    opts.set_backend(BenchHardwareSuite::BackendSelect::Cpu);
    StatusOr<BenchReport::Document> doc = BenchHardwareSuite::run(ctx, opts);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().suite() == BenchReport::Suite::Hardware);
    REQUIRE(doc.value().rows().size() == 3u);
    REQUIRE(doc.value().all_pass());

    REQUIRE(doc.value().rows()[0].name() == "T1");
    REQUIRE(doc.value().rows()[1].name() == "T2");
    REQUIRE(doc.value().rows()[2].name() == "T3");
    REQUIRE(doc.value().rows()[0].backend() == BenchReport::Backend::Cpu);
    REQUIRE(doc.value().rows()[0].status() == BenchReport::RowStatus::Pass);
    REQUIRE(doc.value().rows()[0].detail().find("scale_factor=") != std::string::npos);
    REQUIRE(doc.value().rows()[0].detail().find("smoke") != std::string::npos);
    REQUIRE(doc.value().rows()[1].detail().find("cpu_partial") != std::string::npos);

    const nlohmann::json j = doc.value().to_json(true);
    REQUIRE(j.at("suite").get<std::string>() == "hardware");
    REQUIRE(j.at("omit_timing").get<bool>());
    REQUIRE_FALSE(j.at("rows").at(0).contains("runes_per_sec"));

    const std::string human = BenchFormatter::format(doc.value());
    REQUIRE(human.find("HARDWARE") != std::string::npos);
    REQUIRE(human.find("T1") != std::string::npos);
}

TEST_CASE(
    "BenchHardwareSuite CUDA skipped_not_built when allow_cuda without device",
    "[bench][hardware]") {
    const parcae::tool::Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    BenchHardwareSuite::Options opts;
    opts.set_backend(BenchHardwareSuite::BackendSelect::Both);
    opts.set_allow_cuda(true);
    StatusOr<BenchReport::Document> doc = BenchHardwareSuite::run(ctx, opts);
    REQUIRE(doc.ok());
#if defined(PARCAE_HAS_CUDA)
    // With CUDA build + available GPU, expect measured CUDA rows (or skips if no device).
    REQUIRE(doc.value().rows().size() >= 3u);
#else
    REQUIRE(doc.value().rows().size() == 6u);
    REQUIRE(doc.value().rows()[3].name() == "T1");
    REQUIRE(doc.value().rows()[3].backend() == BenchReport::Backend::Cuda);
    REQUIRE(doc.value().rows()[3].status() == BenchReport::RowStatus::Skipped);
    REQUIRE(doc.value().rows()[3].detail() == "skipped_not_built");
    REQUIRE(doc.value().rows()[4].name() == "T2");
    REQUIRE(doc.value().rows()[5].name() == "T3");
    REQUIRE(doc.value().all_pass());
#endif
}

TEST_CASE("BenchHardwareSuite require_cuda fails without device", "[bench][hardware]") {
    const parcae::tool::Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    BenchHardwareSuite::Options opts;
    opts.set_backend(BenchHardwareSuite::BackendSelect::Cuda);
    opts.set_require_cuda(true);
    StatusOr<BenchReport::Document> doc = BenchHardwareSuite::run(ctx, opts);
#if defined(PARCAE_HAS_CUDA)
    // May pass if a device is present; only assert error path on CPU-only builds.
    (void)doc;
#else
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("require-cuda") != std::string::npos);
#endif
}

TEST_CASE(
    "BenchHardwareSuite cuda-only allow_skip yields skipped rows", "[bench][hardware]") {
    const parcae::tool::Context ctx{std::string(PARCAE_TEST_DATA_DIR)};
    BenchHardwareSuite::Options opts;
    opts.set_backend(BenchHardwareSuite::BackendSelect::Cuda);
    opts.set_allow_cuda(true);
    opts.set_allow_skip(true);
    StatusOr<BenchReport::Document> doc = BenchHardwareSuite::run(ctx, opts);
#if defined(PARCAE_HAS_CUDA)
    REQUIRE(doc.ok());
#else
    REQUIRE(doc.ok());
    REQUIRE(doc.value().rows().size() == 3u);
    REQUIRE(doc.value().rows()[0].status() == BenchReport::RowStatus::Skipped);
    REQUIRE(doc.value().rows()[0].detail() == "skipped_not_built");
    REQUIRE(doc.value().all_pass());
#endif
}
