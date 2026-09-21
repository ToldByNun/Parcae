#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/primitive_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

TEST_CASE("DslIrApplicator expr caesar matches CaesarTransform", "[dsl][applicator]") {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr shift = Z29Expr::var("shift");
    const Z29Expr::Ptr encrypt = Z29Expr::add(x, shift);
    const Z29Expr::Ptr decrypt = Z29Expr::sub(x, shift);

    const std::vector<Index29> input{Index29{3}, Index29{10}, Index29{28}, Index29{0}};
    const Index29 shift_v{5};
    Z29Expr::Env env{{"shift", shift_v}};

    std::vector<Index29> out(input.size());
    REQUIRE(DslIrApplicator::apply_into(
                encrypt, "x", env, input, out, InterruptPolicy::none())
                .ok());

    std::vector<Index29> expect(input.size());
    REQUIRE(CaesarTransform::kernel(input, expect, shift_v, TransformDirection::Encrypt).ok());
    REQUIRE(out == expect);

    REQUIRE(DslIrApplicator::apply_into(
                decrypt, "x", env, out, out, InterruptPolicy::none())
                .ok());
    REQUIRE(out == input);
}

TEST_CASE("DslIrApplicator respects interrupt skip pass-through", "[dsl][applicator]") {
    const Z29Expr::Ptr step = Z29Expr::add(Z29Expr::var("x"), Z29Expr::var("shift"));
    Z29Expr::Env env{{"shift", Index29{1}}};
    const std::vector<Index29> input{Index29{0}, Index29{1}, Index29{2}};
    const StatusOr<InterruptPolicy> interrupt =
        InterruptPolicy::from_skip_indices({1});
    REQUIRE(interrupt.ok());

    std::vector<Index29> out(input.size());
    REQUIRE(DslIrApplicator::apply_into(step, "x", env, input, out, interrupt.value()).ok());
    REQUIRE(out[0].value() == 1);
    REQUIRE(out[1].value() == 1); // skipped
    REQUIRE(out[2].value() == 3);
}

TEST_CASE("DslIrApplicator PrimitiveIr poly2 stream", "[dsl][applicator]") {
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr c2 = Z29Expr::var("c2");
    const Z29Expr::Ptr c1 = Z29Expr::var("c1");
    const Z29Expr::Ptr c0 = Z29Expr::var("c0");
    const Z29Expr::Ptr body = Z29Expr::add(
        Z29Expr::add(Z29Expr::mul(Z29Expr::mul(c2, i), i), Z29Expr::mul(c1, i)), c0);
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "poly2_mod29", "(i: Z29, c2: Z29, c1: Z29, c0: Z29) -> Z29", body);
    REQUIRE(prim.ok());

    const std::vector<Index29> input{Index29{2}, Index29{7}};
    const std::vector<Index29> tail{Index29{3}, Index29{5}, Index29{1}};
    const StatusOr<std::vector<Index29>> out =
        DslIrApplicator::apply(prim.value(), tail, input);
    REQUIRE(out.ok());
    REQUIRE(out.value().size() == 2);

    const Index29 e0 = Z29::add(
        Z29::add(Z29::mul(Z29::mul(Index29{3}, Index29{2}), Index29{2}), Z29::mul(Index29{5}, Index29{2})),
        Index29{1});
    const Index29 e1 = Z29::add(
        Z29::add(Z29::mul(Z29::mul(Index29{3}, Index29{7}), Index29{7}), Z29::mul(Index29{5}, Index29{7})),
        Index29{1});
    REQUIRE(out.value()[0] == e0);
    REQUIRE(out.value()[1] == e1);
}

TEST_CASE("DslIrApplicator TheoryIr encrypt/decrypt round-trip", "[dsl][applicator]") {
    const StatusOr<ParamIr> shift_p = ParamIr::make("shift", 0, 28);
    REQUIRE(shift_p.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr shift = Z29Expr::var("shift");
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "dsl_caesar",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {shift_p.value()},
        Z29Expr::add(x, shift),
        Z29Expr::sub(x, shift));
    REQUIRE(theory.ok());

    const std::vector<Index29> plain{Index29{4}, Index29{15}, Index29{27}};
    const nlohmann::json params{{"shift", 9}};

    const StatusOr<std::vector<Index29>> cipher = DslIrApplicator::apply(
        theory.value(), plain, params, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());
    const StatusOr<std::vector<Index29>> back = DslIrApplicator::apply(
        theory.value(), cipher.value(), params, TransformDirection::Decrypt);
    REQUIRE(back.ok());
    REQUIRE(back.value() == plain);
}

TEST_CASE("DslIrApplicator rejects missing theory step", "[dsl][applicator]") {
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "empty_steps",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {});
    REQUIRE(theory.ok());
    std::vector<Index29> in{Index29{1}};
    std::vector<Index29> out(1);
    const Status st = DslIrApplicator::apply_into(
        theory.value(),
        in,
        out,
        nlohmann::json::object(),
        TransformDirection::Encrypt);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("encrypt_step") != std::string::npos);
}

TEST_CASE("DslIrApplicator rejects interrupt under none_by_design", "[dsl][applicator]") {
    const StatusOr<ParamIr> c0 = ParamIr::make("c0", 0, 28);
    REQUIRE(c0.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "no_irq",
        TheoryIr::Family::KeyedStream,
        TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign,
        {c0.value()},
        x,
        x,
        std::string("Speculative. interrupt parity fixture."));
    REQUIRE(theory.ok());

    const std::vector<Index29> plain{Index29{1}, Index29{2}, Index29{3}};
    const nlohmann::json params{{"c0", 0}};
    const StatusOr<InterruptPolicy> irq = InterruptPolicy::from_skip_indices({1});
    REQUIRE(irq.ok());

    REQUIRE(DslIrApplicator::apply(
                theory.value(), plain, params, TransformDirection::Encrypt, InterruptPolicy::none())
                .ok());

    const StatusOr<std::vector<Index29>> rejected = DslIrApplicator::apply(
        theory.value(), plain, params, TransformDirection::Encrypt, irq.value());
    REQUIRE_FALSE(rejected.ok());
    REQUIRE(rejected.status().message().find("none_by_design") != std::string::npos);
    REQUIRE(rejected.status().message().find("E030") != std::string::npos);
}

TEST_CASE("DslIrApplicator rejects param outside declared domain", "[dsl][applicator]") {
    const StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
    REQUIRE(a.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "need_a",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {a.value()},
        x,
        x);
    REQUIRE(theory.ok());
    std::vector<Index29> in{Index29{1}};
    const StatusOr<std::vector<Index29>> out = DslIrApplicator::apply(
        theory.value(), in, nlohmann::json{{"a", 0}}, TransformDirection::Encrypt);
    REQUIRE_FALSE(out.ok());
    REQUIRE(out.status().message().find("E040") != std::string::npos);
}
