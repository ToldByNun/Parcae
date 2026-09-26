#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>

TEST_CASE("Index29 accepts 0..28", "[index29]") {
    for (std::uint8_t v = 0; v < Index29::modulus; ++v) {
        const Index29 idx{v};
        REQUIRE(idx.value() == v);
    }
}

TEST_CASE("Index29 try_make accepts 0..28 and rejects 29+", "[index29][try_make]") {
    for (std::uint8_t v = 0; v < Index29::modulus; ++v) {
        const StatusOr<Index29> got = Index29::try_make(v);
        REQUIRE(got.ok());
        REQUIRE(got.value().value() == v);
    }
    const StatusOr<Index29> bad = Index29::try_make(29);
    REQUIRE_FALSE(bad.ok());
    REQUIRE(bad.status().message().find("out of range") != std::string::npos);
    REQUIRE_FALSE(Index29::try_make(255).ok());
}

TEST_CASE("Z29 add/sub wraparound", "[z29]") {
    REQUIRE(Z29::add(Index29{28}, Index29{1}).value() == 0);
    REQUIRE(Z29::sub(Index29{0}, Index29{1}).value() == 28);
    REQUIRE(Z29::add(Index29{0}, Index29{0}).value() == 0);
    REQUIRE(Z29::add(Index29{14}, Index29{15}).value() == 0);
    REQUIRE(Z29::sub(Index29{5}, Index29{5}).value() == 0);
    REQUIRE(Z29::neg(Index29{0}).value() == 0);
    REQUIRE(Z29::neg(Index29{1}).value() == 28);
}

TEST_CASE("Z29 mul and inv table for 1..28", "[z29]") {
    for (std::uint8_t a = 1; a < Index29::modulus; ++a) {
        const Index29 inv_a = Z29::inv(Index29{a});
        REQUIRE(Z29::mul(Index29{a}, inv_a).value() == 1);
    }

    REQUIRE(Z29::mul(Index29{2}, Z29::inv(Index29{2})).value() == 1);
    REQUIRE(Z29::mul(Index29{15}, Z29::inv(Index29{15})).value() == 1);
    REQUIRE(Z29::mul(Index29{0}, Index29{7}).value() == 0);
}

TEST_CASE("Z29 atbash is an involution", "[z29]") {
    REQUIRE(Z29::atbash(Index29{0}).value() == 28);
    REQUIRE(Z29::atbash(Index29{28}).value() == 0);

    for (std::uint8_t x = 0; x < Index29::modulus; ++x) {
        const Index29 idx{x};
        REQUIRE(Z29::atbash(Z29::atbash(idx)) == idx);
    }
}

TEST_CASE("Z29 ops are usable in constant expressions", "[z29]") {
    constexpr Index29 a{28};
    constexpr Index29 b{1};
    constexpr auto sum = Z29::add(a, b);
    constexpr auto diff = Z29::sub(Index29{0}, Index29{1});
    constexpr auto reflected = Z29::atbash(Index29{3});
    constexpr auto inv2 = Z29::inv(Index29{2});

    STATIC_REQUIRE(sum.value() == 0);
    STATIC_REQUIRE(diff.value() == 28);
    STATIC_REQUIRE(reflected.value() == 25);
    STATIC_REQUIRE(Z29::mul(Index29{2}, inv2).value() == 1);
}

TEST_CASE("Z29 select mux", "[z29][select]") {
    REQUIRE(Z29::select(Index29{0}, Index29{5}, Index29{9}).value() == 9);
    REQUIRE(Z29::select(Index29{1}, Index29{5}, Index29{9}).value() == 5);
    REQUIRE(Z29::select(Index29{7}, Index29{3}, Index29{4}).value() == 3);
}

TEST_CASE("Z29 floor_div of representatives", "[z29][floor_div]") {
    REQUIRE(Z29::floor_div(Index29{28}, Index29{3}).value() == 9);
    REQUIRE(Z29::floor_div(Index29{5}, Index29{2}).value() == 2);
    REQUIRE(Z29::floor_div(Index29{0}, Index29{7}).value() == 0);
    REQUIRE(Z29::floor_div(Index29{28}, Index29{1}).value() == 28);

    constexpr auto q = Z29::floor_div(Index29{28}, Index29{3});
    STATIC_REQUIRE(q.value() == 9);
}

TEST_CASE("Z29 try_inv matches inv for 1..28 and rejects 0", "[z29][try_inv]") {
    for (std::uint8_t a = 1; a < Index29::modulus; ++a) {
        const StatusOr<Index29> soft = Z29::try_inv(Index29{a});
        REQUIRE(soft.ok());
        REQUIRE(soft.value() == Z29::inv(Index29{a}));
        REQUIRE(Z29::mul(Index29{a}, soft.value()).value() == 1);
    }
    const StatusOr<Index29> zero = Z29::try_inv(Index29{0});
    REQUIRE_FALSE(zero.ok());
    REQUIRE(zero.status().message().find("inverse of 0") != std::string::npos);
}

TEST_CASE("Z29 try_floor_div matches floor_div and rejects divisor 0",
          "[z29][try_floor_div]") {
    REQUIRE(Z29::try_floor_div(Index29{28}, Index29{3}).value().value() == 9);
    REQUIRE(Z29::try_floor_div(Index29{5}, Index29{2}).value().value() == 2);
    REQUIRE(Z29::try_floor_div(Index29{0}, Index29{7}).value().value() == 0);
    const StatusOr<Index29> bad = Z29::try_floor_div(Index29{3}, Index29{0});
    REQUIRE_FALSE(bad.ok());
    REQUIRE(bad.status().message().find("divisor is 0") != std::string::npos);
}
