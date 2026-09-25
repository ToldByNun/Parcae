#include <catch2/catch_test_macros.hpp>

#include "parcae/core/index29.hpp"
#include "parcae/math/z29_matrix2.hpp"
#include "parcae/math/z29_matrix3.hpp"

#include "z29_matrix2_device.hpp"
#include "z29_matrix3_device.hpp"

#include <array>
#include <cstdint>
#include <vector>

#if defined(PARCAE_HAS_CUDA)
#include "parcae_cuda.hpp"
#include "z29_matrix_device_ops.hpp"
#endif

namespace {

[[nodiscard]] std::array<std::uint8_t, 4> to_bytes2(const Z29Matrix2& m) {
    return {m.a().value(), m.b().value(), m.c().value(), m.d().value()};
}

[[nodiscard]] std::array<std::uint8_t, 9> to_bytes3(const Z29Matrix3& m) {
    std::array<std::uint8_t, 9> out{};
    for (std::size_t i = 0; i < 9; ++i) {
        out[i] = m.row_major()[i].value();
    }
    return out;
}

} // namespace

TEST_CASE("Z29Matrix2Device det/mul_vec/inverse match CPU Z29Matrix2",
          "[cuda][parity][matrix][z29_matrix2]") {
    for (std::uint8_t a = 0; a < 7; ++a) {
        for (std::uint8_t b = 0; b < 7; ++b) {
            for (std::uint8_t c = 0; c < 7; ++c) {
                for (std::uint8_t d = 0; d < 7; ++d) {
                    const Z29Matrix2 cpu{Index29{a}, Index29{b}, Index29{c}, Index29{d}};
                    const auto bytes = to_bytes2(cpu);
                    REQUIRE(Z29Matrix2Device::det(bytes.data()) == cpu.det().value());

                    std::uint8_t vec_out[2]{};
                    Z29Matrix2Device::mul_vec(bytes.data(), 4, 6, vec_out);
                    const auto cpu_vec = cpu.mul_vec(Index29{4}, Index29{6});
                    REQUIRE(vec_out[0] == cpu_vec[0].value());
                    REQUIRE(vec_out[1] == cpu_vec[1].value());

                    std::uint8_t inv_out[4]{};
                    std::uint8_t singular = 0xFFu;
                    Z29Matrix2Device::try_inverse(bytes.data(), inv_out, &singular);
                    StatusOr<Z29Matrix2> cpu_inv = cpu.try_inverse();
                    if (!cpu_inv.ok()) {
                        REQUIRE(singular == 1);
                    } else {
                        REQUIRE(singular == 0);
                        REQUIRE(inv_out[0] == cpu_inv.value().a().value());
                        REQUIRE(inv_out[1] == cpu_inv.value().b().value());
                        REQUIRE(inv_out[2] == cpu_inv.value().c().value());
                        REQUIRE(inv_out[3] == cpu_inv.value().d().value());
                    }
                }
            }
        }
    }
}

TEST_CASE("Z29Matrix3Device det/mul_vec/inverse match CPU Z29Matrix3",
          "[cuda][parity][matrix][z29_matrix3]") {
    const Z29Matrix3 known{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                           Index29{4}, Index29{5}, Index29{6}, Index29{0}};
    const auto known_bytes = to_bytes3(known);
    REQUIRE(Z29Matrix3Device::det(known_bytes.data()) == known.det().value());

    std::uint8_t vec_out[3]{};
    Z29Matrix3Device::mul_vec(known_bytes.data(), 2, 3, 4, vec_out);
    const auto cpu_vec = known.mul_vec(Index29{2}, Index29{3}, Index29{4});
    REQUIRE(vec_out[0] == cpu_vec[0].value());
    REQUIRE(vec_out[1] == cpu_vec[1].value());
    REQUIRE(vec_out[2] == cpu_vec[2].value());

    std::uint8_t inv_out[9]{};
    std::uint8_t singular = 0xFFu;
    Z29Matrix3Device::try_inverse(known_bytes.data(), inv_out, &singular);
    StatusOr<Z29Matrix3> cpu_inv = known.try_inverse();
    REQUIRE(cpu_inv.ok());
    REQUIRE(singular == 0);
    for (std::size_t i = 0; i < 9; ++i) {
        REQUIRE(inv_out[i] == cpu_inv.value().row_major()[i].value());
    }

    const Z29Matrix3 singular_m{Index29{1}, Index29{2}, Index29{3}, Index29{2}, Index29{4},
                                Index29{6}, Index29{0}, Index29{1}, Index29{0}};
    const auto sing_bytes = to_bytes3(singular_m);
    REQUIRE(Z29Matrix3Device::det(sing_bytes.data()) == 0);
    singular = 0;
    Z29Matrix3Device::try_inverse(sing_bytes.data(), inv_out, &singular);
    REQUIRE(singular == 1);

    // Bounded entry grid parity (entries in {0,1,2}).
    for (std::uint8_t e0 = 0; e0 < 3; ++e0) {
        for (std::uint8_t e1 = 0; e1 < 3; ++e1) {
            for (std::uint8_t e2 = 0; e2 < 3; ++e2) {
                for (std::uint8_t e3 = 0; e3 < 3; ++e3) {
                    for (std::uint8_t e4 = 0; e4 < 3; ++e4) {
                        for (std::uint8_t e5 = 0; e5 < 3; ++e5) {
                            for (std::uint8_t e6 = 0; e6 < 3; ++e6) {
                                for (std::uint8_t e7 = 0; e7 < 3; ++e7) {
                                    for (std::uint8_t e8 = 0; e8 < 3; ++e8) {
                                        const Z29Matrix3 cpu{
                                            Index29{e0}, Index29{e1}, Index29{e2}, Index29{e3},
                                            Index29{e4}, Index29{e5}, Index29{e6}, Index29{e7},
                                            Index29{e8}};
                                        const auto bytes = to_bytes3(cpu);
                                        REQUIRE(Z29Matrix3Device::det(bytes.data()) ==
                                                cpu.det().value());
                                        singular = 0xFFu;
                                        Z29Matrix3Device::try_inverse(bytes.data(), inv_out,
                                                                      &singular);
                                        StatusOr<Z29Matrix3> inv = cpu.try_inverse();
                                        if (!inv.ok()) {
                                            REQUIRE(singular == 1);
                                        } else {
                                            REQUIRE(singular == 0);
                                            for (std::size_t i = 0; i < 9; ++i) {
                                                REQUIRE(inv_out[i] ==
                                                        inv.value().row_major()[i].value());
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

#if defined(PARCAE_HAS_CUDA)

TEST_CASE("CUDA Z29MatrixDeviceOps inverse2 singularity flags match CPU",
          "[cuda][parity][matrix][z29_matrix2]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint8_t> matrices;
    std::vector<std::uint8_t> expect_singular;
    std::vector<std::uint8_t> expect_inv;

    for (std::uint8_t a = 0; a < 5; ++a) {
        for (std::uint8_t b = 0; b < 5; ++b) {
            for (std::uint8_t c = 0; c < 5; ++c) {
                for (std::uint8_t d = 0; d < 5; ++d) {
                    const Z29Matrix2 cpu{Index29{a}, Index29{b}, Index29{c}, Index29{d}};
                    const auto bytes = to_bytes2(cpu);
                    matrices.insert(matrices.end(), bytes.begin(), bytes.end());
                    StatusOr<Z29Matrix2> inv = cpu.try_inverse();
                    if (!inv.ok()) {
                        expect_singular.push_back(1);
                        expect_inv.insert(expect_inv.end(), {0, 0, 0, 0});
                    } else {
                        expect_singular.push_back(0);
                        expect_inv.push_back(inv.value().a().value());
                        expect_inv.push_back(inv.value().b().value());
                        expect_inv.push_back(inv.value().c().value());
                        expect_inv.push_back(inv.value().d().value());
                    }
                }
            }
        }
    }

    std::vector<std::uint8_t> out(matrices.size(), 0xFFu);
    std::vector<std::uint8_t> singular(expect_singular.size(), 0xFFu);
    REQUIRE(Z29MatrixDeviceOps::apply_host_inverse2(matrices, out, singular).ok());
    REQUIRE(singular == expect_singular);
    for (std::size_t i = 0; i < singular.size(); ++i) {
        if (singular[i] == 0) {
            for (std::size_t k = 0; k < 4; ++k) {
                REQUIRE(out[i * 4 + k] == expect_inv[i * 4 + k]);
            }
        }
    }
}

TEST_CASE("CUDA Z29MatrixDeviceOps mul_vec2 matches CPU",
          "[cuda][parity][matrix][z29_matrix2]") {
    REQUIRE(ParcaeCuda::available());

    const Z29Matrix2 m{Index29{2}, Index29{3}, Index29{5}, Index29{7}};
    const auto mb = to_bytes2(m);
    std::vector<std::uint8_t> matrices = {mb[0], mb[1], mb[2], mb[3], mb[0], mb[1], mb[2], mb[3]};
    std::vector<std::uint8_t> vecs_in = {4, 6, 1, 0};
    std::vector<std::uint8_t> vecs_out(4, 0xFFu);
    REQUIRE(Z29MatrixDeviceOps::apply_host_mul_vec2(matrices, vecs_in, vecs_out).ok());

    const auto v0 = m.mul_vec(Index29{4}, Index29{6});
    const auto v1 = m.mul_vec(Index29{1}, Index29{0});
    REQUIRE(vecs_out[0] == v0[0].value());
    REQUIRE(vecs_out[1] == v0[1].value());
    REQUIRE(vecs_out[2] == v1[0].value());
    REQUIRE(vecs_out[3] == v1[1].value());
}

TEST_CASE("CUDA Z29MatrixDeviceOps inverse3 singularity flags match CPU",
          "[cuda][parity][matrix][z29_matrix3]") {
    REQUIRE(ParcaeCuda::available());

    const Z29Matrix3 invertible{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                                Index29{4}, Index29{5}, Index29{6}, Index29{0}};
    const Z29Matrix3 singular_m{Index29{1}, Index29{2}, Index29{3}, Index29{2}, Index29{4},
                                Index29{6}, Index29{0}, Index29{1}, Index29{0}};
    const auto ib = to_bytes3(invertible);
    const auto sb = to_bytes3(singular_m);

    std::vector<std::uint8_t> matrices;
    matrices.insert(matrices.end(), ib.begin(), ib.end());
    matrices.insert(matrices.end(), sb.begin(), sb.end());

    std::vector<std::uint8_t> out(18, 0xFFu);
    std::vector<std::uint8_t> singular(2, 0xFFu);
    REQUIRE(Z29MatrixDeviceOps::apply_host_inverse3(matrices, out, singular).ok());
    REQUIRE(singular[0] == 0);
    REQUIRE(singular[1] == 1);

    StatusOr<Z29Matrix3> inv = invertible.try_inverse();
    REQUIRE(inv.ok());
    for (std::size_t i = 0; i < 9; ++i) {
        REQUIRE(out[i] == inv.value().row_major()[i].value());
    }
}

TEST_CASE("CUDA Z29MatrixDeviceOps mul_vec3 matches CPU",
          "[cuda][parity][matrix][z29_matrix3]") {
    REQUIRE(ParcaeCuda::available());

    const Z29Matrix3 m{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                       Index29{4}, Index29{5}, Index29{6}, Index29{0}};
    const auto mb = to_bytes3(m);
    std::vector<std::uint8_t> matrices(mb.begin(), mb.end());
    std::vector<std::uint8_t> vecs_in = {2, 3, 4};
    std::vector<std::uint8_t> vecs_out(3, 0xFFu);
    REQUIRE(Z29MatrixDeviceOps::apply_host_mul_vec3(matrices, vecs_in, vecs_out).ok());
    const auto v = m.mul_vec(Index29{2}, Index29{3}, Index29{4});
    REQUIRE(vecs_out[0] == v[0].value());
    REQUIRE(vecs_out[1] == v[1].value());
    REQUIRE(vecs_out[2] == v[2].value());
}

#else

TEST_CASE("CUDA Z29MatrixDeviceOps skipped (PARCAE_HAS_CUDA unset)",
          "[cuda][parity][matrix]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise Z29MatrixDeviceOps kernels");
}

#endif
