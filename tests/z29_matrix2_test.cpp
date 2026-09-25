#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/math/z29_matrix2.hpp>
#include <string>

TEST_CASE("Z29Matrix2 det for identity and singular", "[math][matrix][z29_matrix2]") {
    const Z29Matrix2 identity{Index29{1}, Index29{0}, Index29{0}, Index29{1}};
    REQUIRE(identity.det().value() == 1);

    const Z29Matrix2 singular{Index29{2}, Index29{4}, Index29{3}, Index29{6}};
    REQUIRE(singular.det().value() == 0);

    // [[2, 3], [5, 7]] → det = 14 - 15 = -1 ≡ 28 (mod 29)
    const Z29Matrix2 m{Index29{2}, Index29{3}, Index29{5}, Index29{7}};
    REQUIRE(m.det().value() == 28);
}

TEST_CASE("Z29Matrix2 try_inverse rejects singular", "[math][matrix][z29_matrix2]") {
    const Z29Matrix2 singular{Index29{1}, Index29{2}, Index29{2}, Index29{4}};
    StatusOr<Z29Matrix2> inv = singular.try_inverse();
    REQUIRE_FALSE(inv.ok());
    REQUIRE(inv.status().message().find("singular") != std::string::npos);
}

TEST_CASE("Z29Matrix2 try_inverse then mul recovers identity", "[math][matrix][z29_matrix2]") {
    const Z29Matrix2 m{Index29{2}, Index29{3}, Index29{5}, Index29{7}};
    StatusOr<Z29Matrix2> inv = m.try_inverse();
    REQUIRE(inv.ok());

    // M · M^{-1} should be I on basis vectors (and thus as a matrix product).
    const auto e0 = inv.value().mul_vec(m.mul_vec(Index29{1}, Index29{0}));
    const auto e1 = inv.value().mul_vec(m.mul_vec(Index29{0}, Index29{1}));
    REQUIRE(e0[0].value() == 1);
    REQUIRE(e0[1].value() == 0);
    REQUIRE(e1[0].value() == 0);
    REQUIRE(e1[1].value() == 1);

    // Direct adjugate check: inv = 28 * [[7, -3], [-5, 2]] since det=28, inv(28)=28
    // (28^{-1} ≡ 28 because 28*28=784≡1 mod 29).
    REQUIRE(inv.value().a().value() == Z29::mul(Index29{28}, Index29{7}).value());
    REQUIRE(inv.value().b().value() == Z29::mul(Index29{28}, Z29::neg(Index29{3})).value());
    REQUIRE(inv.value().c().value() == Z29::mul(Index29{28}, Z29::neg(Index29{5})).value());
    REQUIRE(inv.value().d().value() == Z29::mul(Index29{28}, Index29{2}).value());
}

TEST_CASE("Z29Matrix2 mul_vec matches hand calculation", "[math][matrix][z29_matrix2]") {
    const Z29Matrix2 m{Index29{2}, Index29{3}, Index29{5}, Index29{7}};
    const auto out = m.mul_vec(Index29{4}, Index29{6});
    // [2*4+3*6, 5*4+7*6] = [8+18, 20+42] = [26, 62] → [26, 4] mod 29
    REQUIRE(out[0].value() == 26);
    REQUIRE(out[1].value() == 4);

    const auto via_array = m.mul_vec(std::array<Index29, 2>{Index29{4}, Index29{6}});
    REQUIRE(via_array == out);
}

TEST_CASE("Z29Matrix2 encrypt-decrypt roundtrip for all invertible keys in a grid",
          "[math][matrix][z29_matrix2]") {
    // Bounded exhaustive: a,b,c in {0..6}, d chosen so we still cover many invertible matrices.
    for (std::uint8_t a = 0; a < 7; ++a) {
        for (std::uint8_t b = 0; b < 7; ++b) {
            for (std::uint8_t c = 0; c < 7; ++c) {
                for (std::uint8_t d = 0; d < 7; ++d) {
                    const Z29Matrix2 m{Index29{a}, Index29{b}, Index29{c}, Index29{d}};
                    if (m.det().value() == 0) {
                        continue;
                    }
                    StatusOr<Z29Matrix2> inv = m.try_inverse();
                    REQUIRE(inv.ok());
                    for (std::uint8_t x = 0; x < Index29::modulus; ++x) {
                        for (std::uint8_t y = 0; y < Index29::modulus; ++y) {
                            const auto mid = m.mul_vec(Index29{x}, Index29{y});
                            const auto back = inv.value().mul_vec(mid);
                            REQUIRE(back[0].value() == x);
                            REQUIRE(back[1].value() == y);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Z29Matrix2 from_row_major and accessors", "[math][matrix][z29_matrix2]") {
    const auto m = Z29Matrix2::from_row_major(
        {Index29{1}, Index29{2}, Index29{3}, Index29{4}});
    REQUIRE(m.a().value() == 1);
    REQUIRE(m.b().value() == 2);
    REQUIRE(m.c().value() == 3);
    REQUIRE(m.d().value() == 4);
    REQUIRE(m.at(0, 0).value() == 1);
    REQUIRE(m.at(0, 1).value() == 2);
    REQUIRE(m.at(1, 0).value() == 3);
    REQUIRE(m.at(1, 1).value() == 4);
    REQUIRE(m.row_major()[2].value() == 3);
}
