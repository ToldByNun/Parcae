#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <vector>

TEST_CASE("CUDA DeviceBuffer host round-trip", "[cuda][buffer]") {
    REQUIRE(parcae::cuda::available());

    const std::vector<std::uint8_t> host_in{0, 1, 2, 28, 7, 13};
    StatusOr<parcae::cuda::DeviceBuffer<std::uint8_t>> device =
        parcae::cuda::DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device.ok());
    REQUIRE(device.value().size() == host_in.size());

    std::vector<std::uint8_t> host_out(host_in.size());
    REQUIRE(device.value().copy_to_host(host_out).ok());
    REQUIRE(host_out == host_in);
}

TEST_CASE("CUDA DeviceBuffer size mismatch is an error", "[cuda][buffer]") {
    StatusOr<parcae::cuda::DeviceBuffer<std::uint8_t>> device =
        parcae::cuda::DeviceBuffer<std::uint8_t>::allocate(4);
    REQUIRE(device.ok());

    std::vector<std::uint8_t> wrong(2, 0);
    REQUIRE_FALSE(device.value().copy_from_host(wrong).ok());
    REQUIRE_FALSE(device.value().copy_to_host(wrong).ok());
}

TEST_CASE("CUDA DeviceBuffer empty allocate", "[cuda][buffer]") {
    StatusOr<parcae::cuda::DeviceBuffer<std::uint8_t>> device =
        parcae::cuda::DeviceBuffer<std::uint8_t>::allocate(0);
    REQUIRE(device.ok());
    REQUIRE(device.value().empty());
    REQUIRE(device.value().copy_from_host({}).ok());
}

#else

TEST_CASE("CUDA DeviceBuffer skipped (PARCAE_HAS_CUDA unset)", "[cuda][buffer]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise DeviceBuffer");
}

#endif
