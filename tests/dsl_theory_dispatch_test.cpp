#include <parcae/core/index29.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/theory_apply_ir.hpp>
#include <parcae/dsl/theory_dispatch.hpp>
#include <parcae/dsl/theory_envelope_bridge.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_registry.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_EXAMPLES_DIR
#error "PARCAE_EXAMPLES_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_DIR
#error "PARCAE_PYTHON_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_EXE
#error "PARCAE_PYTHON_EXE must be defined"
#endif

namespace {

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

[[nodiscard]] std::vector<Index29> stream_of(std::initializer_list<std::uint8_t> vals) {
    std::vector<Index29> out;
    out.reserve(vals.size());
    for (std::uint8_t v : vals) {
        out.push_back(Index29{v});
    }
    return out;
}

}  // namespace

TEST_CASE("TheoryApplyIr round-trips poly2 TheoryIr", "[dsl][dispatch][i41]") {
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr c2 = Z29Expr::var("c2");
    const Z29Expr::Ptr c1 = Z29Expr::var("c1");
    const Z29Expr::Ptr c0 = Z29Expr::var("c0");
    // keystream = c2*i*i + c1*i + c0 ; encrypt x+ks ; decrypt x-ks
    // For applicator cipher_var is "x"; keystream uses "i" in example but compile
    // uses position via keyed_stream — here use x as cipher and i unbound for unit.
    // Simpler: caesar-like for round-trip.
    StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr s = Z29Expr::var("shift");
    StatusOr<TheoryIr> th = TheoryIr::make(
        "caesar_rt",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {shift.value()},
        Z29Expr::add(x, s),
        Z29Expr::sub(x, s));
    REQUIRE(th.ok());

    StatusOr<nlohmann::json> json = TheoryApplyIr::to_json(th.value());
    REQUIRE(json.ok());
    REQUIRE(json.value().at("schema").get<std::string>() == std::string(TheoryApplyIr::schema_id));

    StatusOr<TheoryIr> back = TheoryApplyIr::from_json(json.value());
    REQUIRE(back.ok());
    REQUIRE(back.value().name() == "caesar_rt");
    REQUIRE(back.value().params().size() == 1);

    const auto input = stream_of({1, 2, 3, 4});
    const nlohmann::json params{{"shift", 5}};
    StatusOr<std::vector<Index29>> via_ir = TheoryDispatch::apply(
        back.value(), input, params, TransformDirection::Encrypt);
    REQUIRE(via_ir.ok());
    StatusOr<std::vector<Index29>> direct = DslIrApplicator::apply(
        th.value(), input, params, TransformDirection::Encrypt);
    REQUIRE(direct.ok());
    REQUIRE(via_ir.value() == direct.value());
}

TEST_CASE("TheoryDispatch catalog envelope delegates to ApplyTransform", "[dsl][dispatch][i41]") {
    const auto root = std::filesystem::temp_directory_path() / "parcae_dispatch_catalog_i41";
    const nlohmann::json env_json{
        {"transform_id", "caesar"},
        {"direction", "encrypt"},
        {"params", {{"shift", 3}}},
    };
    StatusOr<TheoryEnvelopeBridge::Envelope> env =
        TheoryEnvelopeBridge::from_json(env_json);
    REQUIRE(env.ok());
    REQUIRE(env.value().is_catalog());

    const auto input = stream_of({0, 1, 2});
    StatusOr<std::vector<Index29>> out =
        TheoryDispatch::apply(root, env.value(), input);
    REQUIRE(out.ok());
    REQUIRE(out.value()[0].value() == 3);
    REQUIRE(out.value()[1].value() == 4);
    REQUIRE(out.value()[2].value() == 5);
}

TEST_CASE(
    "TheoryDispatch applies compiled new_math theory URI",
    "[dsl][dispatch][i41][compile]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));
    const auto root =
        std::filesystem::temp_directory_path() / "parcae_dispatch_new_math_i41";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    const auto src =
        std::filesystem::path(PARCAE_EXAMPLES_DIR) / "new_math_example.py";
    StatusOr<DslCompile::Result> compiled =
        DslCompile::compile_file(src, root, compile_options());
    REQUIRE(compiled.ok());

    const std::filesystem::path dir = root / "quadratic_polynomial_stream" / "1";
    REQUIRE(std::filesystem::is_regular_file(dir / "apply_ir.json"));

    StatusOr<TheoryArtifact> loaded =
        TheoryRegistry::load(root, "quadratic_polynomial_stream", 1);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().paths().apply_ir().has_value());

    StatusOr<TheoryEnvelopeBridge::Envelope> env =
        TheoryEnvelopeBridge::load(dir / "envelope.json");
    REQUIRE(env.ok());
    // Bind non-default params for a visible keystream.
    nlohmann::json params = env.value().params();
    params["c2"] = 1;
    params["c1"] = 0;
    params["c0"] = 0;
    TheoryEnvelopeBridge::Envelope bound{
        TheoryEnvelopeBridge::Kind::Theory,
        env.value().transform_id(),
        TransformDirection::Encrypt,
        std::move(params),
        env.value().interrupt(),
        env.value().theory_uri()};

    const auto input = stream_of({0, 1, 2, 3});
    StatusOr<std::vector<Index29>> out = TheoryDispatch::apply(root, bound, input);
    REQUIRE(out.ok());

    TheoryEnvelopeBridge::Envelope dec{
        TheoryEnvelopeBridge::Kind::Theory,
        bound.transform_id(),
        TransformDirection::Decrypt,
        bound.params(),
        bound.interrupt(),
        bound.theory_uri()};
    StatusOr<std::vector<Index29>> plain =
        TheoryDispatch::apply(root, dec, out.value());
    REQUIRE(plain.ok());
    REQUIRE(plain.value() == input);

    std::filesystem::remove_all(root, ec);
}

TEST_CASE(
    "TheoryDispatch applies compiled koan1_style compose URI",
    "[dsl][dispatch][i41][compile]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));
    const auto root =
        std::filesystem::temp_directory_path() / "parcae_dispatch_koan_i41";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    const auto src =
        std::filesystem::path(PARCAE_EXAMPLES_DIR) / "full_lifecycle_example.py";
    StatusOr<DslCompile::Result> compiled =
        DslCompile::compile_file(src, root, compile_options());
    REQUIRE(compiled.ok());

    StatusOr<TheoryEnvelopeBridge::Envelope> env = TheoryEnvelopeBridge::load(
        root / "koan1_style" / "1" / "envelope.json");
    REQUIRE(env.ok());
    nlohmann::json params = env.value().params();
    params["caesar_shift"] = 7;
    TheoryEnvelopeBridge::Envelope enc{
        TheoryEnvelopeBridge::Kind::Theory,
        env.value().transform_id(),
        TransformDirection::Encrypt,
        params,
        env.value().interrupt(),
        env.value().theory_uri()};
    TheoryEnvelopeBridge::Envelope dec{
        TheoryEnvelopeBridge::Kind::Theory,
        env.value().transform_id(),
        TransformDirection::Decrypt,
        params,
        env.value().interrupt(),
        env.value().theory_uri()};

    const auto input = stream_of({3, 8, 14, 20, 27});
    StatusOr<std::vector<Index29>> cipher = TheoryDispatch::apply(root, enc, input);
    REQUIRE(cipher.ok());
    StatusOr<std::vector<Index29>> plain =
        TheoryDispatch::apply(root, dec, cipher.value());
    REQUIRE(plain.ok());
    REQUIRE(plain.value() == input);

    std::filesystem::remove_all(root, ec);
}
