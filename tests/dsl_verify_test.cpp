#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/primitive_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

[[nodiscard]] PrimitiveIr make_poly2() {
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr c2 = Z29Expr::var("c2");
    const Z29Expr::Ptr c1 = Z29Expr::var("c1");
    const Z29Expr::Ptr c0 = Z29Expr::var("c0");
    const Z29Expr::Ptr body = Z29Expr::add(
        Z29Expr::add(Z29Expr::mul(Z29Expr::mul(c2, i), i), Z29Expr::mul(c1, i)), c0);
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "poly2_mod29", "(i: Z29, c2: Z29, c1: Z29, c0: Z29) -> Z29", body);
    REQUIRE(prim.ok());
    return prim.value();
}

}  // namespace

TEST_CASE("DslVerifier exhaustive poly2 arity 4 passes", "[dsl][verify]") {
    const PrimitiveIr prim = make_poly2();
    const StatusOr<DslVerifier::Report> report =
        DslVerifier::verify_primitive_exhaustive(prim);
    REQUIRE(report.ok());
    REQUIRE(report.value().passed());
    REQUIRE(report.value().mode() == DslVerifier::Mode::Exhaustive);
    REQUIRE(report.value().samples_checked() == DslVerifier::max_exhaustive_samples);
    REQUIRE(report.value().primitive_name() == "poly2_mod29");
}

TEST_CASE("DslVerifier exhaustive identity arity 1 passes", "[dsl][verify]") {
    const StatusOr<PrimitiveIr> prim =
        PrimitiveIr::make("id29", "(x: Z29) -> Z29", Z29Expr::var("x"));
    REQUIRE(prim.ok());
    const StatusOr<DslVerifier::Report> report =
        DslVerifier::verify_primitive_exhaustive(prim.value());
    REQUIRE(report.ok());
    REQUIRE(report.value().passed());
    REQUIRE(report.value().samples_checked() == 29);
}

TEST_CASE("DslVerifier exhaustive arity 0 passes", "[dsl][verify]") {
    const StatusOr<PrimitiveIr> prim =
        PrimitiveIr::make("const7", "() -> Z29", Z29Expr::constant(7).value());
    REQUIRE(prim.ok());
    const StatusOr<DslVerifier::Report> report =
        DslVerifier::verify_primitive_exhaustive(prim.value());
    REQUIRE(report.ok());
    REQUIRE(report.value().passed());
    REQUIRE(report.value().samples_checked() == 1);
}

TEST_CASE("DslVerifier exhaustive inv fails on domain 0 with E050", "[dsl][verify]") {
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "unsafe_inv", "(x: Z29) -> Z29", Z29Expr::inv(Z29Expr::var("x")));
    REQUIRE(prim.ok());
    const StatusOr<DslVerifier::Report> report =
        DslVerifier::verify_primitive_exhaustive(prim.value());
    REQUIRE_FALSE(report.ok());
    REQUIRE(report.status().message().find(DslRuleId::E050_verify_failed) != std::string::npos);
    REQUIRE(report.status().message().find("totality") != std::string::npos);
    REQUIRE(report.status().message().find("x=0") != std::string::npos);
}

TEST_CASE("DslVerifier rejects arity greater than 4 for exhaustive", "[dsl][verify]") {
    // Five Z29 params — exhaustive not allowed.
    std::string sig = "(a: Z29, b: Z29, c: Z29, d: Z29, e: Z29) -> Z29";
    const Z29Expr::Ptr body = Z29Expr::var("a");
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make("too_wide", sig, body);
    REQUIRE(prim.ok());
    REQUIRE(prim.value().arity() == 5);
    const StatusOr<DslVerifier::Report> report =
        DslVerifier::verify_primitive_exhaustive(prim.value());
    REQUIRE_FALSE(report.ok());
    REQUIRE(report.status().message().find("E050") != std::string::npos);
    REQUIRE(report.status().message().find("fuzz") != std::string::npos);
}

TEST_CASE("DslVerifier fuzz arity 5 with seed 0xC1CADA passes", "[dsl][verify][fuzz]") {
    std::string sig = "(a: Z29, b: Z29, c: Z29, d: Z29, e: Z29) -> Z29";
    // (a*b) + (c*d) + e — total over full domain
    const Z29Expr::Ptr body = Z29Expr::add(
        Z29Expr::add(
            Z29Expr::mul(Z29Expr::var("a"), Z29Expr::var("b")),
            Z29Expr::mul(Z29Expr::var("c"), Z29Expr::var("d"))),
        Z29Expr::var("e"));
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make("wide5", sig, body);
    REQUIRE(prim.ok());

    const StatusOr<DslVerifier::Report> report = DslVerifier::verify_primitive_fuzz(prim.value());
    REQUIRE(report.ok());
    REQUIRE(report.value().passed());
    REQUIRE(report.value().mode() == DslVerifier::Mode::Fuzz);
    REQUIRE(report.value().seed().has_value());
    REQUIRE(*report.value().seed() == DslVerifier::default_fuzz_seed);
    REQUIRE(report.value().samples_checked() == DslVerifier::default_fuzz_samples);
    REQUIRE(report.value().detail().find("C1CADA") != std::string::npos);
}

TEST_CASE("DslVerifier fuzz finds inv(0) with E050", "[dsl][verify][fuzz]") {
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "unsafe_inv", "(x: Z29) -> Z29", Z29Expr::inv(Z29Expr::var("x")));
    REQUIRE(prim.ok());
    // Enough samples that x=0 appears with high probability; seed is fixed.
    const StatusOr<DslVerifier::Report> report =
        DslVerifier::verify_primitive_fuzz(
            prim.value(), DslVerifier::default_fuzz_seed, DslVerifier::default_fuzz_samples);
    REQUIRE_FALSE(report.ok());
    REQUIRE(report.status().message().find("E050") != std::string::npos);
    REQUIRE(report.status().message().find("totality") != std::string::npos);
}

TEST_CASE("DslVerifier verify_primitive auto-selects fuzz for arity 5", "[dsl][verify][fuzz]") {
    std::string sig = "(a: Z29, b: Z29, c: Z29, d: Z29, e: Z29) -> Z29";
    const StatusOr<PrimitiveIr> prim =
        PrimitiveIr::make("pick_fuzz", sig, Z29Expr::var("a"));
    REQUIRE(prim.ok());
    const StatusOr<DslVerifier::Report> report = DslVerifier::verify_primitive(prim.value());
    REQUIRE(report.ok());
    REQUIRE(report.value().mode() == DslVerifier::Mode::Fuzz);
    REQUIRE(*report.value().seed() == 0xC1CADAu);
}

TEST_CASE("DslVerifier verify_primitive auto-selects exhaustive for arity 2", "[dsl][verify]") {
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "add2",
        "(x: Z29, y: Z29) -> Z29",
        Z29Expr::add(Z29Expr::var("x"), Z29Expr::var("y")));
    REQUIRE(prim.ok());
    const StatusOr<DslVerifier::Report> report = DslVerifier::verify_primitive(prim.value());
    REQUIRE(report.ok());
    REQUIRE(report.value().mode() == DslVerifier::Mode::Exhaustive);
    REQUIRE_FALSE(report.value().seed().has_value());
    REQUIRE(report.value().samples_checked() == 29u * 29u);
}

TEST_CASE("DslVerifier fuzz is deterministic for fixed seed", "[dsl][verify][fuzz]") {
    std::string sig = "(a: Z29, b: Z29, c: Z29, d: Z29, e: Z29) -> Z29";
    const StatusOr<PrimitiveIr> prim =
        PrimitiveIr::make("det_fuzz", sig, Z29Expr::var("a"));
    REQUIRE(prim.ok());
    const StatusOr<DslVerifier::Report> a =
        DslVerifier::verify_primitive_fuzz(prim.value(), 0xC1CADAu, 200);
    const StatusOr<DslVerifier::Report> b =
        DslVerifier::verify_primitive_fuzz(prim.value(), 0xC1CADAu, 200);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    REQUIRE(a.value().detail() == b.value().detail());
}

TEST_CASE("DslVerifier cuda mirror gate covers atbash+add+mul", "[dsl][verify][mirror]") {
    // Body uses ops that emit via Z29Device (atbash → sub(28,x)).
    const Z29Expr::Ptr body = Z29Expr::add(
        Z29Expr::atbash(Z29Expr::var("x")),
        Z29Expr::mul(Z29Expr::var("y"), Z29Expr::constant(3).value()));
    const StatusOr<PrimitiveIr> prim =
        PrimitiveIr::make("mirror_mix", "(x: Z29, y: Z29) -> Z29", body);
    REQUIRE(prim.ok());
    const StatusOr<DslVerifier::Report> report =
        DslVerifier::verify_primitive_exhaustive(prim.value());
    REQUIRE(report.ok());
    REQUIRE(report.value().passed());
    REQUIRE(report.value().samples_checked() == 29u * 29u);
    REQUIRE(report.value().detail().find("cuda_mirror") != std::string::npos);
}

TEST_CASE("Z29Expr eval_cuda_mirror matches CPU eval for poly2 cell", "[dsl][verify][mirror]") {
    const PrimitiveIr prim = make_poly2();
    Z29Expr::Env env;
    env.emplace("i", Index29{7});
    env.emplace("c2", Index29{3});
    env.emplace("c1", Index29{11});
    env.emplace("c0", Index29{5});
    const StatusOr<Index29> cpu = prim.body()->eval(env);
    const StatusOr<Index29> cuda = prim.body()->eval_cuda_mirror(env);
    REQUIRE(cpu.ok());
    REQUIRE(cuda.ok());
    REQUIRE(cpu.value() == cuda.value());
}

TEST_CASE("Z29Expr eval_cuda_mirror atbash uses device sub(28,x)", "[dsl][verify][mirror]") {
    const Z29Expr::Ptr expr = Z29Expr::atbash(Z29Expr::var("x"));
    for (std::uint8_t x = 0; x < 29; ++x) {
        Z29Expr::Env env;
        env.emplace("x", Index29{x});
        const StatusOr<Index29> cpu = expr->eval(env);
        const StatusOr<Index29> cuda = expr->eval_cuda_mirror(env);
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        REQUIRE(cpu.value() == cuda.value());
        REQUIRE(cuda.value().value() == static_cast<std::uint8_t>(28 - x));
    }
}

TEST_CASE("DslVerifier max_exhaustive_samples is 29^4", "[dsl][verify]") {
    REQUIRE(DslVerifier::max_exhaustive_samples == 707281u);
    REQUIRE(DslVerifier::max_exhaustive_arity == 4u);
    REQUIRE(DslVerifier::default_fuzz_seed == 0xC1CADAu);
}
