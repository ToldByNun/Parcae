#include "parcae/core/index29.hpp"
#include "parcae/core/z29.hpp"

#include "z29_device.hpp"

#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "affine_kernel.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace {

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

[[nodiscard]] std::vector<Index29> from_bytes(const std::vector<std::uint8_t>& bytes) {
    std::vector<Index29> out;
    out.reserve(bytes.size());
    for (std::uint8_t value : bytes) {
        out.push_back(Index29{value});
    }
    return out;
}

} // namespace

#endif

TEST_CASE("Z29Device inv matches Z29 for all a in 1..28", "[cuda][parity][affine][z29]") {
    for (std::uint8_t a = 1; a <= 28; ++a) {
        REQUIRE(Z29Device::inv(a) == Z29::inv(Index29{a}).value());
        REQUIRE(Z29Device::mul(a, Z29Device::inv(a)) == 1);
    }
}

#if defined(PARCAE_HAS_CUDA)

TEST_CASE("CUDA affine device inv table via kernel", "[cuda][parity][affine][z29]") {
    REQUIRE(ParcaeCuda::available());

    // Decrypt of 1 with (a, b=0) is inv(a) — exercises Z29Device::inv on device.
    for (std::uint8_t a = 1; a <= 28; ++a) {
        std::vector<std::uint8_t> in{1};
        std::vector<std::uint8_t> out{0xFFu};
        REQUIRE(AffineKernel::apply_host(in, out, a, 0, CudaDir::Decrypt).ok());
        REQUIRE(out[0] == Z29::inv(Index29{a}).value());
    }
}

TEST_CASE("CUDA affine parity vs CPU fixed encrypt decrypt", "[cuda][parity][affine]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> input{
        Index29{0}, Index29{1}, Index29{14}, Index29{27}, Index29{28},
    };
    const Index29 a{2};
    const Index29 b{5};

    std::vector<Index29> cpu_enc(input.size());
    REQUIRE(AffineTransform::kernel(input, cpu_enc, a, b, TransformDirection::Encrypt).ok());

    const std::vector<std::uint8_t> host_in = to_bytes(input);
    std::vector<std::uint8_t> host_enc(host_in.size(), 0xFFu);
    REQUIRE(AffineKernel::apply_host(host_in, host_enc, 2, 5, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host_enc) == cpu_enc);

    std::vector<Index29> cpu_dec(input.size());
    REQUIRE(AffineTransform::kernel(cpu_enc, cpu_dec, a, b, TransformDirection::Decrypt).ok());

    std::vector<std::uint8_t> host_dec(host_enc.size());
    REQUIRE(AffineKernel::apply_host(host_enc, host_dec, 2, 5, CudaDir::Decrypt).ok());
    REQUIRE(from_bytes(host_dec) == cpu_dec);
    REQUIRE(host_dec == host_in);
}

TEST_CASE("CUDA affine parity vs CPU random round-trip", "[cuda][parity][affine]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xAFF1A7u);
    std::uniform_int_distribution<int> index_dist(0, 28);
    std::uniform_int_distribution<int> a_dist(1, 28);

    std::vector<Index29> input;
    input.reserve(257);
    for (std::size_t i = 0; i < 257; ++i) {
        input.push_back(Index29{static_cast<std::uint8_t>(index_dist(rng))});
    }
    const std::uint8_t a = static_cast<std::uint8_t>(a_dist(rng));
    const std::uint8_t b = static_cast<std::uint8_t>(index_dist(rng));

    std::vector<Index29> cpu_out(input.size());
    REQUIRE(
        AffineTransform::kernel(input, cpu_out, Index29{a}, Index29{b}, TransformDirection::Encrypt)
            .ok());

    std::vector<std::uint8_t> host = to_bytes(input);
    std::vector<std::uint8_t> host_out(host.size());
    REQUIRE(AffineKernel::apply_host(host, host_out, a, b, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host_out) == cpu_out);

    std::vector<std::uint8_t> round_trip(host_out.size());
    REQUIRE(AffineKernel::apply_host(host_out, round_trip, a, b, CudaDir::Decrypt).ok());
    REQUIRE(round_trip == host);
}

TEST_CASE("CUDA affine in-place and validation", "[cuda][parity][affine]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint8_t> host{0, 1, 2, 28};
    std::vector<Index29> cpu_in = from_bytes(host);
    std::vector<Index29> cpu_out(cpu_in.size());
    REQUIRE(AffineTransform::kernel(cpu_in, cpu_out, Index29{3}, Index29{7},
                                    TransformDirection::Encrypt)
                .ok());

    REQUIRE(AffineKernel::apply_host(host, host, 3, 7, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host) == cpu_out);

    REQUIRE(AffineKernel::launch_device(nullptr, nullptr, 0, 1, 0, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(AffineKernel::launch_device(nullptr, nullptr, 0, 0, 0, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(AffineKernel::launch_device(nullptr, nullptr, 0, 29, 0, CudaDir::Encrypt).ok());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> out(1);
    REQUIRE_FALSE(AffineKernel::apply_host(in, out, 2, 1, CudaDir::Encrypt).ok());
}

#else

TEST_CASE("CUDA affine kernel skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][affine]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise AffineKernel");
}

#endif
