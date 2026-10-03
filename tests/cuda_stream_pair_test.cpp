#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "cuda_stream_pair.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <utility>
#include <vector>

TEST_CASE("CUDA CudaStreamPair create dedicated streams", "[cuda][streams]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<CudaStreamPair> pair = CudaStreamPair::create();
    REQUIRE(pair.ok());
    REQUIRE(pair.value().uses_dedicated_streams());
    REQUIRE(pair.value().copy() != nullptr);
    REQUIRE(pair.value().compute() != nullptr);
    REQUIRE(pair.value().copy() != pair.value().compute());
}

TEST_CASE("CUDA CudaStreamPair legacy fallback", "[cuda][streams]") {
    REQUIRE(ParcaeCuda::available());

    CudaStreamPair legacy = CudaStreamPair::legacy_default();
    REQUIRE_FALSE(legacy.uses_dedicated_streams());
    REQUIRE(legacy.copy() == nullptr);
    REQUIRE(legacy.compute() == nullptr);
    REQUIRE(legacy.record_h2d_done().ok());
    REQUIRE(legacy.wait_h2d_on_compute().ok());
    REQUIRE(legacy.record_compute_done().ok());
    REQUIRE(legacy.wait_compute_on_copy().ok());
    REQUIRE(legacy.synchronize_copy().ok());
}

TEST_CASE("CUDA CudaStreamPair create_or_legacy is dedicated when available",
          "[cuda][streams]") {
    REQUIRE(ParcaeCuda::available());

    CudaStreamPair pair = CudaStreamPair::create_or_legacy();
    REQUIRE(pair.uses_dedicated_streams());
}

TEST_CASE("CUDA CudaStreamPair orders H2D then D2H via events", "[cuda][streams]") {
    REQUIRE(ParcaeCuda::available());

    CudaStreamPair streams = CudaStreamPair::create_or_legacy();
    REQUIRE(streams.uses_dedicated_streams());

    const std::vector<std::uint8_t> host_in{1, 2, 3, 4, 5, 6, 7, 8};
    StatusOr<DeviceBuffer<std::uint8_t>> device = DeviceBuffer<std::uint8_t>::allocate(host_in.size());
    REQUIRE(device.ok());

    REQUIRE(CudaError::to_status(
                cudaMemcpyAsync(device.value().data(), host_in.data(), host_in.size(),
                                cudaMemcpyHostToDevice, streams.copy()),
                "streams H2D")
                .ok());
    REQUIRE(streams.record_h2d_done().ok());
    REQUIRE(streams.wait_h2d_on_compute().ok());

    // No kernel — just prove copy→compute→copy event chain + D2H.
    REQUIRE(streams.record_compute_done().ok());
    REQUIRE(streams.wait_compute_on_copy().ok());

    std::vector<std::uint8_t> host_out(host_in.size(), 0);
    REQUIRE(CudaError::to_status(
                cudaMemcpyAsync(host_out.data(), device.value().data(), host_in.size(),
                                cudaMemcpyDeviceToHost, streams.copy()),
                "streams D2H")
                .ok());
    REQUIRE(streams.synchronize_copy().ok());
    REQUIRE(host_out == host_in);
}

TEST_CASE("CUDA CudaStreamPair move leaves source empty", "[cuda][streams]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<CudaStreamPair> pair = CudaStreamPair::create();
    REQUIRE(pair.ok());
    CudaStreamPair moved = std::move(pair.value());
    REQUIRE(moved.uses_dedicated_streams());
    REQUIRE_FALSE(pair.value().uses_dedicated_streams());
}

#else

TEST_CASE("CUDA CudaStreamPair skipped (PARCAE_HAS_CUDA unset)", "[cuda][streams]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CudaStreamPair");
}

#endif
