#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "atbash_kernel.hpp"
#include "parcae_cuda.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/transform/atbash_transform.hpp"

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

}  // namespace

TEST_CASE("CUDA atbash parity vs CPU fixed vector", "[cuda][parity][atbash]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> input{
        Index29{0},
        Index29{1},
        Index29{14},
        Index29{27},
        Index29{28},
    };
    std::vector<Index29> cpu_out(input.size());
    REQUIRE(AtbashTransform::kernel(input, cpu_out).ok());

    const std::vector<std::uint8_t> host_in = to_bytes(input);
    std::vector<std::uint8_t> host_out(host_in.size(), 0xFFu);
    REQUIRE(AtbashKernel::apply_host(host_in, host_out).ok());
    REQUIRE(from_bytes(host_out) == cpu_out);
}

TEST_CASE("CUDA atbash parity vs CPU random + involution", "[cuda][parity][atbash]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xA7BA54u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> input;
    input.reserve(257);
    for (std::size_t i = 0; i < 257; ++i) {
        input.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }

    std::vector<Index29> cpu_out(input.size());
    REQUIRE(AtbashTransform::kernel(input, cpu_out).ok());

    std::vector<std::uint8_t> host = to_bytes(input);
    std::vector<std::uint8_t> host_out(host.size());
    REQUIRE(AtbashKernel::apply_host(host, host_out).ok());
    REQUIRE(from_bytes(host_out) == cpu_out);

    // Involution: atbash(atbash(x)) == x
    std::vector<std::uint8_t> round_trip(host_out.size());
    REQUIRE(AtbashKernel::apply_host(host_out, round_trip).ok());
    REQUIRE(round_trip == host);
}

TEST_CASE("CUDA atbash in-place on device", "[cuda][parity][atbash]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint8_t> host{0, 5, 10, 15, 20, 28};
    const std::vector<std::uint8_t> expected{28, 23, 18, 13, 8, 0};

    REQUIRE(AtbashKernel::apply_host(host, host).ok());
    REQUIRE(host == expected);
}

TEST_CASE("CUDA atbash empty and size mismatch", "[cuda][parity][atbash]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(AtbashKernel::launch_device(nullptr, nullptr, 0).ok());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> out(1);
    REQUIRE_FALSE(AtbashKernel::apply_host(in, out).ok());
}

#else

TEST_CASE("CUDA atbash skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][atbash]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise AtbashKernel");
}

#endif
