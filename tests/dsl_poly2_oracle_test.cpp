#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/primitive_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

/// Hand oracle: (c2·i·i + c1·i + c0) mod 29 via host Z29 (same as dsl.md poly2_mod29).
[[nodiscard]] Index29 poly2_oracle(Index29 i, Index29 c2, Index29 c1, Index29 c0) {
    return Z29::add(
        Z29::add(Z29::mul(Z29::mul(c2, i), i), Z29::mul(c1, i)), c0);
}

[[nodiscard]] Z29Expr::Ptr poly2_binop_body() {
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr c2 = Z29Expr::var("c2");
    const Z29Expr::Ptr c1 = Z29Expr::var("c1");
    const Z29Expr::Ptr c0 = Z29Expr::var("c0");
    return Z29Expr::add(
        Z29Expr::add(Z29Expr::mul(Z29Expr::mul(c2, i), i), Z29Expr::mul(c1, i)), c0);
}

[[nodiscard]] Z29Expr::Ptr poly2_call_body() {
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr c2 = Z29Expr::var("c2");
    const Z29Expr::Ptr c1 = Z29Expr::var("c1");
    const Z29Expr::Ptr c0 = Z29Expr::var("c0");
    const Z29Expr::Ptr i2 = Z29Expr::call("z29_mul", {i, i});
    const Z29Expr::Ptr t2 = Z29Expr::call("z29_mul", {c2, i2});
    const Z29Expr::Ptr t1 = Z29Expr::call("z29_mul", {c1, i});
    const Z29Expr::Ptr s = Z29Expr::call("z29_add", {t2, t1});
    return Z29Expr::call("z29_add", {s, c0});
}

[[nodiscard]] PrimitiveIr make_poly2(const Z29Expr::Ptr& body) {
    const StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
        "poly2_mod29",
        "(i: Z29, c2: Z29, c1: Z29, c0: Z29) -> Z29",
        body);
    REQUIRE(prim.ok());
    return prim.value();
}

}  // namespace

TEST_CASE("poly2 Z29Expr BinOp matches Z29 oracle exhaustively", "[dsl][oracle][poly2]") {
    const Z29Expr::Ptr body = poly2_binop_body();
    std::uint64_t mismatches = 0;
    Index29 first_bad_got{};
    Index29 first_bad_expect{};
    std::uint8_t bad_i = 0;
    std::uint8_t bad_c2 = 0;
    std::uint8_t bad_c1 = 0;
    std::uint8_t bad_c0 = 0;
    for (std::uint8_t i = 0; i < Index29::modulus; ++i) {
        for (std::uint8_t c2 = 0; c2 < Index29::modulus; ++c2) {
            for (std::uint8_t c1 = 0; c1 < Index29::modulus; ++c1) {
                for (std::uint8_t c0 = 0; c0 < Index29::modulus; ++c0) {
                    const Z29Expr::Env env{
                        {"i", Index29{i}},
                        {"c2", Index29{c2}},
                        {"c1", Index29{c1}},
                        {"c0", Index29{c0}},
                    };
                    const StatusOr<Index29> got = body->eval(env);
                    const Index29 expect =
                        poly2_oracle(Index29{i}, Index29{c2}, Index29{c1}, Index29{c0});
                    if (!got.ok() || got.value() != expect) {
                        if (mismatches == 0) {
                            bad_i = i;
                            bad_c2 = c2;
                            bad_c1 = c1;
                            bad_c0 = c0;
                            first_bad_expect = expect;
                            first_bad_got = got.ok() ? got.value() : Index29{0};
                        }
                        ++mismatches;
                    }
                }
            }
        }
    }
    INFO(
        "first mismatch i=" << static_cast<int>(bad_i) << " c2=" << static_cast<int>(bad_c2)
                            << " c1=" << static_cast<int>(bad_c1) << " c0="
                            << static_cast<int>(bad_c0) << " got="
                            << static_cast<int>(first_bad_got.value()) << " expect="
                            << static_cast<int>(first_bad_expect.value()));
    REQUIRE(mismatches == 0);
}

TEST_CASE("poly2 PrimitiveIr eval matches Z29 oracle exhaustively", "[dsl][oracle][poly2]") {
    const PrimitiveIr prim = make_poly2(poly2_binop_body());
    REQUIRE(prim.arity() == 4);

    std::uint64_t mismatches = 0;
    for (std::uint8_t i = 0; i < Index29::modulus; ++i) {
        for (std::uint8_t c2 = 0; c2 < Index29::modulus; ++c2) {
            for (std::uint8_t c1 = 0; c1 < Index29::modulus; ++c1) {
                for (std::uint8_t c0 = 0; c0 < Index29::modulus; ++c0) {
                    const Index29 args[4]{Index29{i}, Index29{c2}, Index29{c1}, Index29{c0}};
                    const StatusOr<Index29> got = prim.eval(args);
                    const Index29 expect =
                        poly2_oracle(Index29{i}, Index29{c2}, Index29{c1}, Index29{c0});
                    if (!got.ok() || got.value() != expect) {
                        ++mismatches;
                    }
                }
            }
        }
    }
    REQUIRE(mismatches == 0);
}

TEST_CASE("poly2 Call-shaped IR matches BinOp IR and oracle", "[dsl][oracle][poly2]") {
    const PrimitiveIr binop = make_poly2(poly2_binop_body());
    const PrimitiveIr calls = make_poly2(poly2_call_body());

    std::mt19937 rng(0xC1CADAu);
    std::uniform_int_distribution<int> dist(0, 28);
    for (int trial = 0; trial < 2048; ++trial) {
        const Index29 args[4]{
            Index29{static_cast<std::uint8_t>(dist(rng))},
            Index29{static_cast<std::uint8_t>(dist(rng))},
            Index29{static_cast<std::uint8_t>(dist(rng))},
            Index29{static_cast<std::uint8_t>(dist(rng))},
        };
        const StatusOr<Index29> a = binop.eval(args);
        const StatusOr<Index29> b = calls.eval(args);
        REQUIRE(a.ok());
        REQUIRE(b.ok());
        REQUIRE(a.value() == b.value());
        REQUIRE(a.value() == poly2_oracle(args[0], args[1], args[2], args[3]));
    }
}

TEST_CASE("poly2 DslIrApplicator stream matches oracle", "[dsl][oracle][poly2]") {
    const PrimitiveIr prim = make_poly2(poly2_binop_body());
    const std::vector<Index29> stream{
        Index29{0},
        Index29{1},
        Index29{7},
        Index29{14},
        Index29{28},
    };
    const std::vector<Index29> tail{Index29{3}, Index29{5}, Index29{2}}; // c2,c1,c0

    const StatusOr<std::vector<Index29>> out =
        DslIrApplicator::apply(prim, tail, stream, InterruptPolicy::none());
    REQUIRE(out.ok());
    REQUIRE(out.value().size() == stream.size());
    for (std::size_t k = 0; k < stream.size(); ++k) {
        REQUIRE(
            out.value()[k] ==
            poly2_oracle(stream[k], Index29{3}, Index29{5}, Index29{2}));
    }
}

TEST_CASE("poly2 eval is deterministic (double eval)", "[dsl][oracle][poly2]") {
    const PrimitiveIr prim = make_poly2(poly2_binop_body());
    const Index29 args[4]{Index29{11}, Index29{4}, Index29{9}, Index29{17}};
    const StatusOr<Index29> a = prim.eval(args);
    const StatusOr<Index29> b = prim.eval(args);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    REQUIRE(a.value() == b.value());
    REQUIRE(a.value() == poly2_oracle(args[0], args[1], args[2], args[3]));
}
