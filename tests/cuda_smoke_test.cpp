#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "device_buffer.hpp"
#include "identity_copy.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <vector>

TEST_CASE("CUDA smoke identity copy H2D kernel D2H", "[cuda][smoke]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> host_in{0, 1, 2, 28, 7, 13, 27, 3};
    std::vector<std::uint8_t> host_out(host_in.size(), 0xFFu);

    REQUIRE(IdentityCopy::apply_host(host_in, host_out).ok());
    REQUIRE(host_out == host_in);
}

TEST_CASE("CUDA smoke identity copy via DeviceBuffer launch_device", "[cuda][smoke]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> host_in{5, 4, 3, 2, 1, 0, 28};
    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());

    StatusOr<DeviceBuffer<std::uint8_t>> device_out =
        DeviceBuffer<std::uint8_t>::allocate(host_in.size());
    REQUIRE(device_out.ok());

    REQUIRE(IdentityCopy::launch_device(device_in.value().data(), device_out.value().data(),
                                        host_in.size())
                .ok());

    std::vector<std::uint8_t> host_out(host_in.size());
    REQUIRE(device_out.value().copy_to_host(host_out).ok());
    REQUIRE(host_out == host_in);
}

TEST_CASE("CUDA smoke identity copy empty and size mismatch", "[cuda][smoke]") {
    REQUIRE(ParcaeCuda::available());

    REQUIRE(IdentityCopy::launch_device(nullptr, nullptr, 0).ok());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> out(1);
    REQUIRE_FALSE(IdentityCopy::apply_host(in, out).ok());
}

#else

TEST_CASE("CUDA smoke skipped (PARCAE_HAS_CUDA unset)", "[cuda][smoke]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise identity copy kernel");
}

#endif
