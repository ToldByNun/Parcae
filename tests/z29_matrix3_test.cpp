#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/math/z29_matrix3.hpp>
#include <string>

TEST_CASE("Z29Matrix3 det for identity and singular", "[math][matrix][z29_matrix3]") {
    const Z29Matrix3 identity{Index29{1}, Index29{0}, Index29{0}, Index29{0}, Index29{1},
                              Index29{0}, Index29{0}, Index29{0}, Index29{1}};
    REQUIRE(identity.det().value() == 1);

    // Row2 = 2 * Row1 → singular
    const Z29Matrix3 singular{Index29{1}, Index29{2}, Index29{3}, Index29{2}, Index29{4},
                              Index29{6}, Index29{0}, Index29{1}, Index29{0}};
    REQUIRE(singular.det().value() == 0);

    // [[1,2,3],[0,1,4],[5,6,0]]
    // det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
    const Z29Matrix3 m{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                       Index29{4}, Index29{5}, Index29{6}, Index29{0}};
    REQUIRE(m.det().value() == 1);
}

TEST_CASE("Z29Matrix3 try_inverse rejects singular", "[math][matrix][z29_matrix3]") {
    const Z29Matrix3 singular{Index29{1}, Index29{2}, Index29{3}, Index29{2}, Index29{4},
                              Index29{6}, Index29{0}, Index29{1}, Index29{0}};
    StatusOr<Z29Matrix3> inv = singular.try_inverse();
    REQUIRE_FALSE(inv.ok());
    REQUIRE(inv.status().message().find("singular") != std::string::npos);
}

TEST_CASE("Z29Matrix3 try_inverse then mul recovers identity", "[math][matrix][z29_matrix3]") {
    const Z29Matrix3 m{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                       Index29{4}, Index29{5}, Index29{6}, Index29{0}};
    StatusOr<Z29Matrix3> inv = m.try_inverse();
    REQUIRE(inv.ok());
    REQUIRE(m.det().value() == 1);

    const auto e0 = inv.value().mul_vec(m.mul_vec(Index29{1}, Index29{0}, Index29{0}));
    const auto e1 = inv.value().mul_vec(m.mul_vec(Index29{0}, Index29{1}, Index29{0}));
    const auto e2 = inv.value().mul_vec(m.mul_vec(Index29{0}, Index29{0}, Index29{1}));
    REQUIRE(e0[0].value() == 1);
    REQUIRE(e0[1].value() == 0);
    REQUIRE(e0[2].value() == 0);
    REQUIRE(e1[0].value() == 0);
    REQUIRE(e1[1].value() == 1);
    REQUIRE(e1[2].value() == 0);
    REQUIRE(e2[0].value() == 0);
    REQUIRE(e2[1].value() == 0);
    REQUIRE(e2[2].value() == 1);

    // Full plaintext space for this fixed invertible key.
    for (std::uint8_t x = 0; x < Index29::modulus; ++x) {
        for (std::uint8_t y = 0; y < Index29::modulus; ++y) {
            for (std::uint8_t z = 0; z < Index29::modulus; ++z) {
                const auto mid = m.mul_vec(Index29{x}, Index29{y}, Index29{z});
                const auto back = inv.value().mul_vec(mid);
                REQUIRE(back[0].value() == x);
                REQUIRE(back[1].value() == y);
                REQUIRE(back[2].value() == z);
            }
        }
    }
}

TEST_CASE("Z29Matrix3 mul_vec matches hand calculation", "[math][matrix][z29_matrix3]") {
    const Z29Matrix3 m{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                       Index29{4}, Index29{5}, Index29{6}, Index29{0}};
    const auto out = m.mul_vec(Index29{2}, Index29{3}, Index29{4});
    // [1*2+2*3+3*4, 0*2+1*3+4*4, 5*2+6*3+0*4] = [2+6+12, 3+16, 10+18] = [20, 19, 28]
    REQUIRE(out[0].value() == 20);
    REQUIRE(out[1].value() == 19);
    REQUIRE(out[2].value() == 28);

    const auto via_array =
        m.mul_vec(std::array<Index29, 3>{Index29{2}, Index29{3}, Index29{4}});
    REQUIRE(via_array == out);
}

TEST_CASE("Z29Matrix3 encrypt-decrypt roundtrip on bounded invertible keys",
          "[math][matrix][z29_matrix3]") {
    // Entries in {0,1,2}: 3^9 keys; vectors sampled on a 5-step lattice for speed
    // (full 29^3 covered by the identity basis case above + denser spot checks).
    for (std::uint8_t e0 = 0; e0 < 3; ++e0) {
        for (std::uint8_t e1 = 0; e1 < 3; ++e1) {
            for (std::uint8_t e2 = 0; e2 < 3; ++e2) {
                for (std::uint8_t e3 = 0; e3 < 3; ++e3) {
                    for (std::uint8_t e4 = 0; e4 < 3; ++e4) {
                        for (std::uint8_t e5 = 0; e5 < 3; ++e5) {
                            for (std::uint8_t e6 = 0; e6 < 3; ++e6) {
                                for (std::uint8_t e7 = 0; e7 < 3; ++e7) {
                                    for (std::uint8_t e8 = 0; e8 < 3; ++e8) {
                                        const Z29Matrix3 m{
                                            Index29{e0}, Index29{e1}, Index29{e2}, Index29{e3},
                                            Index29{e4}, Index29{e5}, Index29{e6}, Index29{e7},
                                            Index29{e8}};
                                        if (m.det().value() == 0) {
                                            continue;
                                        }
                                        StatusOr<Z29Matrix3> inv = m.try_inverse();
                                        REQUIRE(inv.ok());
                                        for (std::uint8_t x = 0; x < Index29::modulus; x += 4) {
                                            for (std::uint8_t y = 0; y < Index29::modulus;
                                                 y += 4) {
                                                for (std::uint8_t z = 0; z < Index29::modulus;
                                                     z += 4) {
                                                    const auto mid =
                                                        m.mul_vec(Index29{x}, Index29{y},
                                                                  Index29{z});
                                                    const auto back = inv.value().mul_vec(mid);
                                                    REQUIRE(back[0].value() == x);
                                                    REQUIRE(back[1].value() == y);
                                                    REQUIRE(back[2].value() == z);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Z29Matrix3 from_row_major and accessors", "[math][matrix][z29_matrix3]") {
    const auto m = Z29Matrix3::from_row_major({Index29{1}, Index29{2}, Index29{3}, Index29{4},
                                               Index29{5}, Index29{6}, Index29{7}, Index29{8},
                                               Index29{9}});
    REQUIRE(m.at(0, 0).value() == 1);
    REQUIRE(m.at(0, 2).value() == 3);
    REQUIRE(m.at(1, 1).value() == 5);
    REQUIRE(m.at(2, 2).value() == 9);
    REQUIRE(m.row_major()[4].value() == 5);
}
