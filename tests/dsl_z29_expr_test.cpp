#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

TEST_CASE("Z29Expr constant and var eval", "[dsl][z29expr]") {
    const StatusOr<Z29Expr::Ptr> c7 = Z29Expr::constant(7);
    REQUIRE(c7.ok());
    REQUIRE(c7.value()->kind() == Z29Expr::Kind::Const);
    REQUIRE(c7.value()->eval({}).value().value() == 7);

    const Z29Expr::Ptr x = Z29Expr::var("x");
    Z29Expr::Env env{{"x", Index29{11}}};
    REQUIRE(x->eval(env).value().value() == 11);
}

TEST_CASE("Z29Expr rejects out-of-domain constant with E040", "[dsl][z29expr]") {
    const StatusOr<Z29Expr::Ptr> bad = Z29Expr::constant(29);
    REQUIRE_FALSE(bad.ok());
    REQUIRE(bad.status().message().find("E040") != std::string::npos);
}

TEST_CASE("Z29Expr add/sub/mul match Z29", "[dsl][z29expr]") {
    const auto a = Z29Expr::constant(28).value();
    const auto b = Z29Expr::constant(1).value();
    REQUIRE(Z29Expr::add(a, b)->eval({}).value().value() == Z29::add(Index29{28}, Index29{1}).value());
    REQUIRE(Z29Expr::sub(b, a)->eval({}).value().value() == Z29::sub(Index29{1}, Index29{28}).value());
    REQUIRE(Z29Expr::mul(a, b)->eval({}).value().value() == Z29::mul(Index29{28}, Index29{1}).value());
}

TEST_CASE("Z29Expr inv and inv(0) domain error", "[dsl][z29expr]") {
    for (std::uint8_t v = 1; v < Index29::modulus; ++v) {
        const auto c = Z29Expr::constant(v).value();
        REQUIRE(Z29Expr::inv(c)->eval({}).value() == Z29::inv(Index29{v}));
    }
    const auto zero = Z29Expr::constant(0).value();
    const StatusOr<Index29> bad = Z29Expr::inv(zero)->eval({});
    REQUIRE_FALSE(bad.ok());
    REQUIRE(bad.status().message().find("E040") != std::string::npos);
}

TEST_CASE("Z29Expr neg and atbash", "[dsl][z29expr]") {
    const auto x = Z29Expr::constant(3).value();
    REQUIRE(Z29Expr::neg(x)->eval({}).value() == Z29::neg(Index29{3}));
    REQUIRE(Z29Expr::atbash(x)->eval({}).value() == Z29::atbash(Index29{3}));
}

TEST_CASE("Z29Expr unbound var is E032", "[dsl][z29expr]") {
    const StatusOr<Index29> r = Z29Expr::var("missing")->eval({});
    REQUIRE_FALSE(r.ok());
    REQUIRE(r.status().message().find("E032") != std::string::npos);
}

TEST_CASE("Z29Expr builtin call z29_mul", "[dsl][z29expr]") {
    const auto expr = Z29Expr::call(
        "z29_mul", {Z29Expr::constant(3).value(), Z29Expr::constant(5).value()});
    REQUIRE(expr->eval({}).value().value() == 15);
}

TEST_CASE("Z29Expr unknown call fails E032", "[dsl][z29expr]") {
    const auto expr = Z29Expr::call("not_a_prim", {Z29Expr::constant(1).value()});
    const StatusOr<Index29> r = expr->eval({});
    REQUIRE_FALSE(r.ok());
    REQUIRE(r.status().message().find("E032") != std::string::npos);
}

TEST_CASE("Z29Expr poly2 matches Z29 oracle", "[dsl][z29expr]") {
    // (c2 * i * i) + (c1 * i) + c0
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr c2 = Z29Expr::var("c2");
    const Z29Expr::Ptr c1 = Z29Expr::var("c1");
    const Z29Expr::Ptr c0 = Z29Expr::var("c0");
    const Z29Expr::Ptr body = Z29Expr::add(
        Z29Expr::add(Z29Expr::mul(Z29Expr::mul(c2, i), i), Z29Expr::mul(c1, i)), c0);

    const std::uint8_t i_v = 7;
    const std::uint8_t c2_v = 3;
    const std::uint8_t c1_v = 5;
    const std::uint8_t c0_v = 2;
    Z29Expr::Env env{
        {"i", Index29{i_v}},
        {"c2", Index29{c2_v}},
        {"c1", Index29{c1_v}},
        {"c0", Index29{c0_v}},
    };

    const Index29 expect = Z29::add(
        Z29::add(
            Z29::mul(Z29::mul(Index29{c2_v}, Index29{i_v}), Index29{i_v}),
            Z29::mul(Index29{c1_v}, Index29{i_v})),
        Index29{c0_v});
    REQUIRE(body->eval(env).value() == expect);
}

TEST_CASE("Z29Expr bitwise shift compare bool match Z29 and cuda mirror", "[dsl][z29expr]") {
    const auto a = Z29Expr::constant(28).value();
    const auto b = Z29Expr::constant(7).value();
    const auto c = Z29Expr::constant(3).value();

    REQUIRE(Z29Expr::bit_xor(a, b)->eval({}).value() == Z29::bit_xor(Index29{28}, Index29{7}));
    REQUIRE(
        Z29Expr::bit_xor(a, b)->eval_cuda_mirror({}).value() ==
        Z29Expr::bit_xor(a, b)->eval({}).value());

    REQUIRE(Z29Expr::bit_and(a, b)->eval({}).value() == Z29::bit_and(Index29{28}, Index29{7}));
    REQUIRE(Z29Expr::bit_or(a, b)->eval({}).value() == Z29::bit_or(Index29{28}, Index29{7}));
    REQUIRE(Z29Expr::bit_not(c)->eval({}).value() == Z29::bit_not(Index29{3}));
    REQUIRE(Z29Expr::bit_not(c)->eval({}).value() == Z29::atbash(Index29{3}));

    REQUIRE(Z29Expr::lshift(c, b)->eval({}).value() == Z29::lshift(Index29{3}, Index29{7}));
    REQUIRE(Z29Expr::rshift(a, c)->eval({}).value() == Z29::rshift(Index29{28}, Index29{3}));

    REQUIRE(Z29Expr::pow(c, b)->eval({}).value() == Z29::pow(Index29{3}, Index29{7}));
    REQUIRE(Z29Expr::floor_div(a, c)->eval({}).value() == Z29::floor_div(Index29{28}, Index29{3}));

    // Modular / : 3 / 7 == 3 * inv(7)
    REQUIRE(
        Z29Expr::div(c, b)->eval({}).value() ==
        Z29::mul(Index29{3}, Z29::inv(Index29{7})));
    REQUIRE_FALSE(Z29Expr::div(c, Z29Expr::constant(0).value())->eval({}).ok());

    REQUIRE(Z29Expr::lt(c, b)->eval({}).value().value() == 1);
    REQUIRE(Z29Expr::gt(c, b)->eval({}).value().value() == 0);
    REQUIRE(Z29Expr::eq(c, c)->eval({}).value().value() == 1);
    REQUIRE(Z29Expr::bool_and(c, b)->eval({}).value().value() == 1);
    REQUIRE(Z29Expr::bool_or(Z29Expr::constant(0).value(), b)->eval({}).value().value() == 1);
    REQUIRE(Z29Expr::bool_not(Z29Expr::constant(0).value())->eval({}).value().value() == 1);

    for (const auto& expr : {
             Z29Expr::mod(a, b),
             Z29Expr::lshift(c, Z29Expr::constant(2).value()),
             Z29Expr::ge(a, b),
             Z29Expr::bool_and(a, Z29Expr::constant(0).value()),
         }) {
        REQUIRE(expr->eval({}).value() == expr->eval_cuda_mirror({}).value());
    }
}

