#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/compose_ir.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/primitive_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

TEST_CASE("ParamIr accepts valid domain", "[dsl][ir]") {
    const StatusOr<ParamIr> p = ParamIr::make("a", 1, 28);
    REQUIRE(p.ok());
    REQUIRE(p.value().cardinality() == 28);
    REQUIRE(p.value().contains(Index29{1}));
    REQUIRE_FALSE(p.value().contains(Index29{0}));
}

TEST_CASE("ParamIr rejects min > max with E040", "[dsl][ir]") {
    const StatusOr<ParamIr> p = ParamIr::make("x", 5, 3, "t.py", 2, 0);
    REQUIRE_FALSE(p.ok());
    REQUIRE(p.status().message().find("E040") != std::string::npos);
}

TEST_CASE("PrimitiveIr parses signature and evals poly2", "[dsl][ir]") {
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr c2 = Z29Expr::var("c2");
    const Z29Expr::Ptr c1 = Z29Expr::var("c1");
    const Z29Expr::Ptr c0 = Z29Expr::var("c0");
    const Z29Expr::Ptr body = Z29Expr::add(
        Z29Expr::add(Z29Expr::mul(Z29Expr::mul(c2, i), i), Z29Expr::mul(c1, i)), c0);

    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "poly2_mod29",
        "(i: Z29, c2: Z29, c1: Z29, c0: Z29) -> Z29",
        body);
    REQUIRE(prim.ok());
    REQUIRE(prim.value().arity() == 4);
    REQUIRE(prim.value().param_names()[0] == "i");

    const std::vector<Index29> args{
        Index29{7}, Index29{3}, Index29{5}, Index29{2}};
    const Index29 expect = Z29::add(
        Z29::add(
            Z29::mul(Z29::mul(Index29{3}, Index29{7}), Index29{7}),
            Z29::mul(Index29{5}, Index29{7})),
        Index29{2});
    REQUIRE(prim.value().eval(args).value() == expect);
}

TEST_CASE("PrimitiveIr rejects bad signature", "[dsl][ir]") {
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "bad", "(i: int) -> Z29", Z29Expr::var("i"));
    REQUIRE_FALSE(prim.ok());
    REQUIRE(prim.status().message().find("E032") != std::string::npos);
}

TEST_CASE("TheoryIr elementwise tier A ok without claim", "[dsl][ir]") {
    const StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
    const StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "my_affine",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {a.value(), b.value()});
    REQUIRE(th.ok());
}

TEST_CASE("TheoryIr tier B requires structural_claim E013", "[dsl][ir]") {
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "speculative",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::B,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {},
        {},
        {},
        std::nullopt,
        "theories/x.py",
        42,
        4);
    REQUIRE_FALSE(th.ok());
    REQUIRE(th.status().message().find("E013") != std::string::npos);
    REQUIRE(th.status().message().find("theories/x.py:42:4:") != std::string::npos);
}

TEST_CASE("TheoryIr keyed_stream without interrupt is E030", "[dsl][ir]") {
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "streamy",
        TheoryIr::Family::KeyedStream,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {},
        {},
        {},
        std::nullopt,
        "theories/x.py",
        88,
        4);
    REQUIRE_FALSE(th.ok());
    REQUIRE(th.status().message().find("E030") != std::string::npos);
}

TEST_CASE("TheoryIr keyed_stream with none_by_design ok", "[dsl][ir]") {
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "streamy",
        TheoryIr::Family::KeyedStream,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::NoneByDesign,
        {});
    REQUIRE(th.ok());
    REQUIRE(
        TheoryIr::interrupt_mode_str(th.value().interrupt_mode()) ==
        std::string("none_by_design"));
}

TEST_CASE("ComposeIr koan1-style steps", "[dsl][ir]") {
    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> comp = ComposeIr::make(
        "my_koan1_style",
        TheoryIr::Tier::A,
        {"atbash", "caesar"},
        {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(comp.ok());
    REQUIRE(comp.value().steps().size() == 2);
    REQUIRE(comp.value().step_params().size() == 1);
}

TEST_CASE("ComposeIr rejects unknown step in step_params", "[dsl][ir]") {
    const StatusOr<ComposeIr> comp = ComposeIr::make(
        "bad",
        TheoryIr::Tier::A,
        {"atbash"},
        {},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE_FALSE(comp.ok());
    REQUIRE(comp.status().message().find("E032") != std::string::npos);
}

TEST_CASE("ComposeIr tier C requires claim", "[dsl][ir]") {
    const StatusOr<ComposeIr> comp = ComposeIr::make(
        "c_compose",
        TheoryIr::Tier::C,
        {"atbash", "caesar"},
        {},
        {},
        std::nullopt,
        "t.py",
        1,
        0);
    REQUIRE_FALSE(comp.ok());
    REQUIRE(comp.status().message().find("E013") != std::string::npos);
}

TEST_CASE("TheoryIr parse_family and parse_tier", "[dsl][ir]") {
    REQUIRE(TheoryIr::parse_family("keyed_stream").value() == TheoryIr::Family::KeyedStream);
    REQUIRE(TheoryIr::parse_tier("C").value() == TheoryIr::Tier::C);
    REQUIRE_FALSE(TheoryIr::parse_family("nope").ok());
}
