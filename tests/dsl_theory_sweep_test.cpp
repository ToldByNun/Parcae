#include <parcae/core/status_or.hpp>
#include <parcae/core/version.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_sweep.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace {

constexpr const char* kSha =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

[[nodiscard]] TheoryArtifact::Verification ok_exhaustive() {
    return TheoryArtifact::Verification{
        DslVerifier::Mode::Exhaustive,
        true,
        std::nullopt,
        "2026-09-21T00:00:00Z",
    };
}

[[nodiscard]] StatusOr<TheoryArtifact> make_with_sweep(
    std::string name,
    TheoryIr::Tier tier,
    nlohmann::json sweep,
    std::optional<std::string> claim = std::nullopt) {
    return TheoryArtifact::make(
        std::move(name),
        1,
        tier,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {TheoryArtifact::Param{"c0", 0, 5}, TheoryArtifact::Param{"c1", 0, 3}},
        {"poly2_mod29"},
        std::move(claim),
        std::nullopt,
        {},
        std::move(sweep));
}

[[nodiscard]] std::filesystem::path make_temp_root(std::string_view suffix) {
    const auto root =
        std::filesystem::temp_directory_path() / ("parcae_theory_sweep_" + std::string(suffix));
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    return root;
}

}  // namespace

TEST_CASE("TheorySweep expands param_grid cartesian product", "[dsl][sweep]") {
    nlohmann::json sweep{
        {"theory", "grid_theory"},
        {"corpus", "synthetic_noise_v0"},
        {"param_grid", {{"c0", nlohmann::json::array({0, 1})}, {"c1", {{"min", 0}, {"max", 1}}}}},
        {"record_metrics", nlohmann::json::array({"ic_mod29"})},
    };
    StatusOr<TheoryArtifact> a =
        make_with_sweep("grid_theory", TheoryIr::Tier::A, std::move(sweep));
    REQUIRE(a.ok());

    StatusOr<TheorySweep::Plan> plan = TheorySweep::plan(a.value());
    REQUIRE(plan.ok());
    REQUIRE(plan.value().candidates().size() == 4);  // 2 * 2
    REQUIRE(plan.value().total_before_limit() == 4);
    REQUIRE_FALSE(plan.value().truncated());
    REQUIRE(plan.value().corpus() == "synthetic_noise_v0");
}

TEST_CASE("TheorySweep respects --limit truncation", "[dsl][sweep]") {
    nlohmann::json sweep{
        {"theory", "lim_theory"},
        {"corpus", "lab_corpus"},
        {"param_grid", {{"c0", "full"}, {"c1", nlohmann::json::array({0})}}},
        {"record_metrics", nlohmann::json::array({"ic_mod29"})},
    };
    StatusOr<TheoryArtifact> a =
        make_with_sweep("lim_theory", TheoryIr::Tier::A, std::move(sweep));
    REQUIRE(a.ok());

    TheorySweep::Options opt;
    opt.set_limit(3);
    StatusOr<TheorySweep::Plan> plan = TheorySweep::plan(a.value(), opt);
    REQUIRE(plan.ok());
    REQUIRE(plan.value().candidates().size() == 3);
    REQUIRE(plan.value().total_before_limit() == 6);  // c0 0..5
    REQUIRE(plan.value().truncated());
}

TEST_CASE("TheorySweep rejects null sweep and solved corpus for tier B", "[dsl][sweep]") {
    StatusOr<TheoryArtifact> no_sweep = TheoryArtifact::make(
        "nosweep",
        1,
        TheoryIr::Tier::A,
        TheoryIr::Family::Elementwise,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::ElementwiseDefault);
    REQUIRE(no_sweep.ok());
    REQUIRE_FALSE(TheorySweep::plan(no_sweep.value()).ok());

    nlohmann::json bad{
        {"theory", "b_theory"},
        {"corpus", "a-warning"},
        {"param_grid", {{"c0", nlohmann::json::array({0})}, {"c1", nlohmann::json::array({0})}}},
        {"record_metrics", nlohmann::json::array({"ic_mod29"})},
        {"compare_against", "random_baseline"},
    };
    StatusOr<TheoryArtifact> b = make_with_sweep(
        "b_theory",
        TheoryIr::Tier::B,
        std::move(bad),
        std::string("Spekulativ. sweep corpus gate."));
    REQUIRE(b.ok());
    StatusOr<TheorySweep::Plan> plan = TheorySweep::plan(b.value());
    REQUIRE_FALSE(plan.ok());
    REQUIRE(plan.status().message().find("solved-oracle") != std::string::npos);
}

TEST_CASE("TheorySweep plan_uri rejects stale dsl_spec", "[dsl][sweep]") {
    const std::filesystem::path root = make_temp_root("stale");
    nlohmann::json sweep{
        {"theory", "stale_sw"},
        {"corpus", "lab"},
        {"param_grid", {{"c0", nlohmann::json::array({0})}, {"c1", nlohmann::json::array({0})}}},
        {"record_metrics", nlohmann::json::array({"ic_mod29"})},
    };
    StatusOr<TheoryArtifact> a =
        make_with_sweep("stale_sw", TheoryIr::Tier::A, std::move(sweep));
    REQUIRE(a.ok());
    REQUIRE(a.value().store(root).ok());

    nlohmann::json j = a.value().to_json();
    j["dsl_spec_version"] = "0.9.0";
    {
        std::ofstream out(a.value().manifest_path(root), std::ios::binary | std::ios::trunc);
        out << j.dump(2) << '\n';
    }

    StatusOr<TheorySweep::Plan> plan = TheorySweep::plan_uri(root, "stale_sw@1");
    REQUIRE_FALSE(plan.ok());
    REQUIRE(plan.status().message().find("re-run parcae-compile") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("TheorySweep plan_uri happy path", "[dsl][sweep]") {
    const std::filesystem::path root = make_temp_root("ok");
    nlohmann::json sweep{
        {"theory", "ok_sw"},
        {"corpus", "lab"},
        {"param_grid",
         {{"c0", nlohmann::json{{"values", nlohmann::json::array({1, 2})}}},
          {"c1", nlohmann::json::array({0})}}},
        {"record_metrics", nlohmann::json::array({"ic_mod29", "chi2"})},
    };
    StatusOr<TheoryArtifact> a =
        make_with_sweep("ok_sw", TheoryIr::Tier::A, std::move(sweep));
    REQUIRE(a.ok());
    REQUIRE(a.value().store(root).ok());

    StatusOr<TheorySweep::Plan> plan =
        TheorySweep::plan_uri(root, "parcae://theories/ok_sw@1");
    REQUIRE(plan.ok());
    REQUIRE(plan.value().candidates().size() == 2);
    REQUIRE(plan.value().to_json().at("execute").get<bool>() == false);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}
