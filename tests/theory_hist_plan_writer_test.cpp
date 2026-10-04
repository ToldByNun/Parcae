#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_hist_plan_writer.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <string>

namespace {

[[nodiscard]] TheoryIr make_progressive() {
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr s =
        Z29Expr::add(Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), i));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "plan_progressive", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, s),
        Z29Expr::sub(x, s), std::string("Speculative. hist plan writer."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_autokey() {
    const StatusOr<ParamIr> lag = ParamIr::make("lag", 1, 28);
    REQUIRE(lag.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr prior =
        Z29Expr::call("z29_autokey_shift", {x, Z29Expr::var("lag")});
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "plan_autokey", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {lag.value()}, Z29Expr::add(x, prior),
        Z29Expr::sub(x, prior), std::string("Speculative. autokey soft-fall plan."));
    REQUIRE(theory.ok());
    return theory.value();
}

} // namespace

TEST_CASE("TheoryHistPlanWriter prepares S2 specialized plan + sources",
          "[dsl][hist][plan]") {
    const TheoryIr theory = make_progressive();
    StatusOr<TheoryHistPlanWriter::Bundle> bundle =
        TheoryHistPlanWriter::prepare(theory, "parcae://theories/plan_progressive@1");
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().write_sources());
    REQUIRE(bundle.value().summary().emitted_strategy() == "S2_uchar4_inline");
    REQUIRE(TheoryHistPlanWriter::validate_plan_json(bundle.value().plan_json(),
                                                     "parcae://theories/plan_progressive@1")
                .ok());
    REQUIRE(bundle.value().plan_json().at("s2_linear").is_object());
    REQUIRE(bundle.value().plan_json().at("s2_linear").at("cipher_minus_ks") == true);
}

TEST_CASE("TheoryHistPlanWriter soft-fall still writes safe S0 plan", "[dsl][hist][plan]") {
    const TheoryIr theory = make_autokey();
    StatusOr<TheoryHistPlanWriter::Bundle> bundle =
        TheoryHistPlanWriter::prepare(theory, "parcae://theories/plan_autokey@1");
    REQUIRE(bundle.ok());
    REQUIRE_FALSE(bundle.value().specialized());
    REQUIRE_FALSE(bundle.value().write_sources());
    REQUIRE(bundle.value().summary().emitted_strategy() == "S0_bytecode");
    REQUIRE(bundle.value().summary().specialized() == false);
    REQUIRE(TheoryHistPlanWriter::validate_plan_json(bundle.value().plan_json()).ok());
}

TEST_CASE("TheoryHistPlanWriter write + reload validates schema", "[dsl][hist][plan]") {
    const TheoryIr theory = make_progressive();
    StatusOr<TheoryHistPlanWriter::Bundle> bundle =
        TheoryHistPlanWriter::prepare(theory, "parcae://theories/plan_progressive@1");
    REQUIRE(bundle.ok());

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_hist_plan_writer";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    REQUIRE(TheoryHistPlanWriter::write(root, bundle.value()).ok());
    REQUIRE(std::filesystem::is_regular_file(root / "hist" / "hist_plan.json"));
    REQUIRE(std::filesystem::is_regular_file(root / bundle.value().header_rel()));
    REQUIRE(std::filesystem::is_regular_file(root / bundle.value().source_rel()));

    StatusOr<nlohmann::json> loaded = TheoryHistPlanWriter::load_plan(
        root / "hist" / "hist_plan.json", "parcae://theories/plan_progressive@1");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().at("specialized") == true);
}

TEST_CASE("TheoryHistPlanWriter rejects specialized=false with non-S0 emit",
          "[dsl][hist][plan]") {
    nlohmann::json bad{{"schema", "parcae.theory_hist_plan.v0"},
                       {"theory_uri", "parcae://theories/x@1"},
                       {"intended_strategy", "S3_scalar_inline"},
                       {"emitted_strategy", "S3_scalar_inline"},
                       {"specialized", false},
                       {"cipher_var", "x"},
                       {"reason", "broken"},
                       {"s1_lut", nullptr},
                       {"s2_linear", nullptr},
                       {"s3", nullptr},
                       {"s4_autokey", nullptr}};
    REQUIRE_FALSE(TheoryHistPlanWriter::validate_plan_json(bad).ok());
}
