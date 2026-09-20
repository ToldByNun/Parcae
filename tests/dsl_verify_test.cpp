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

TEST_CASE("DslVerifier rejects arity greater than 4", "[dsl][verify]") {
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

TEST_CASE("DslVerifier max_exhaustive_samples is 29^4", "[dsl][verify]") {
    REQUIRE(DslVerifier::max_exhaustive_samples == 707281u);
    REQUIRE(DslVerifier::max_exhaustive_arity == 4u);
}
