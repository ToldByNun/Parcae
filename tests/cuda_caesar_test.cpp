#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "caesar_kernel.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"

#include <cuda_runtime_api.h>

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

TEST_CASE("CUDA caesar parity vs CPU fixed encrypt decrypt", "[cuda][parity][caesar]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> input{
        Index29{0}, Index29{1}, Index29{14}, Index29{27}, Index29{28},
    };
    const Index29 shift{3};

    std::vector<Index29> cpu_enc(input.size());
    REQUIRE(CaesarTransform::kernel(input, cpu_enc, shift, TransformDirection::Encrypt).ok());

    const std::vector<std::uint8_t> host_in = to_bytes(input);
    std::vector<std::uint8_t> host_enc(host_in.size(), 0xFFu);
    REQUIRE(CaesarKernel::apply_host(host_in, host_enc, 3, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host_enc) == cpu_enc);

    std::vector<Index29> cpu_dec(input.size());
    REQUIRE(CaesarTransform::kernel(cpu_enc, cpu_dec, shift, TransformDirection::Decrypt).ok());

    std::vector<std::uint8_t> host_dec(host_enc.size());
    REQUIRE(CaesarKernel::apply_host(host_enc, host_dec, 3, CudaDir::Decrypt).ok());
    REQUIRE(from_bytes(host_dec) == cpu_dec);
    REQUIRE(host_dec == host_in);
}

TEST_CASE("CUDA caesar parity vs CPU random round-trip", "[cuda][parity][caesar]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xCAE5A7u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> input;
    input.reserve(257);
    for (std::size_t i = 0; i < 257; ++i) {
        input.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }
    const std::uint8_t shift = static_cast<std::uint8_t>(dist(rng));

    std::vector<Index29> cpu_out(input.size());
    REQUIRE(
        CaesarTransform::kernel(input, cpu_out, Index29{shift}, TransformDirection::Encrypt).ok());

    std::vector<std::uint8_t> host = to_bytes(input);
    std::vector<std::uint8_t> host_out(host.size());
    REQUIRE(CaesarKernel::apply_host(host, host_out, shift, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host_out) == cpu_out);

    std::vector<std::uint8_t> round_trip(host_out.size());
    REQUIRE(CaesarKernel::apply_host(host_out, round_trip, shift, CudaDir::Decrypt).ok());
    REQUIRE(round_trip == host);
}

TEST_CASE("CUDA caesar in-place on device", "[cuda][parity][caesar]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint8_t> host{0, 5, 10, 28};
    const std::vector<std::uint8_t> expected{27, 3, 8, 26}; // decrypt shift 2
    REQUIRE(CaesarKernel::apply_host(host, host, 2, CudaDir::Decrypt).ok());
    REQUIRE(host == expected);
}

TEST_CASE("CUDA caesar empty mismatch and bad shift", "[cuda][parity][caesar]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(CaesarKernel::launch_device(nullptr, nullptr, 0, 0, CudaDir::Encrypt).ok());
    REQUIRE(CaesarKernel::launch_device_async(nullptr, nullptr, 0, 0, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(CaesarKernel::launch_device(nullptr, nullptr, 0, 29, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(
        CaesarKernel::launch_device_async(nullptr, nullptr, 0, 29, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(
        CaesarKernel::launch_device_async(nullptr, nullptr, 1, 0, CudaDir::Encrypt).ok());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> out(1);
    REQUIRE_FALSE(CaesarKernel::apply_host(in, out, 1, CudaDir::Encrypt).ok());
}

TEST_CASE("CUDA caesar uchar4 lengths and async", "[cuda][parity][caesar]") {
    REQUIRE(ParcaeCuda::available());

    const std::uint8_t shift = 7;
    const std::size_t lengths[] = {1u, 3u, 4u, 5u, 29u, 257u};
    for (std::size_t n : lengths) {
        std::vector<Index29> input;
        input.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            input.push_back(Index29{static_cast<std::uint8_t>(i % 29u)});
        }
        std::vector<Index29> cpu(n);
        REQUIRE(CaesarTransform::kernel(input, cpu, Index29{shift}, TransformDirection::Encrypt)
                    .ok());

        const std::vector<std::uint8_t> host_in = to_bytes(input);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        REQUIRE(device_in.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_out =
            DeviceBuffer<std::uint8_t>::allocate(n);
        REQUIRE(device_out.ok());

        REQUIRE(CaesarKernel::launch_device_async(device_in.value().data(),
                                                  device_out.value().data(), n, shift,
                                                  CudaDir::Encrypt)
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "caesar async sync").ok());

        std::vector<std::uint8_t> host_out(n);
        REQUIRE(device_out.value().copy_to_host(host_out).ok());
        REQUIRE(from_bytes(host_out) == cpu);
    }
}

#else

TEST_CASE("CUDA caesar skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][caesar]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CaesarKernel");
}

#endif
