#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/hist_alphabet_map.hpp"

#include "column_hist_once.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

TEST_CASE("ColumnHistOnce empty stream is zero cols", "[cuda][hist][column]") {
    REQUIRE(ParcaeCuda::available());

    constexpr std::uint32_t L = 3;
    std::vector<std::uint32_t> cols(static_cast<std::size_t>(L) * ColumnHistOnce::alphabet, 99u);
    REQUIRE(ColumnHistOnce::count_host({}, L, cols).ok());
    REQUIRE(std::accumulate(cols.begin(), cols.end(), 0u) == 0u);
    for (std::uint32_t c : cols) {
        REQUIRE(c == 0u);
    }
}

TEST_CASE("ColumnHistOnce matches HistAlphabetMap count_column_hist", "[cuda][hist][column]") {
    REQUIRE(ParcaeCuda::available());

    SECTION("short T with mixed remainders") {
        const std::vector<std::uint8_t> cipher = {0, 1, 2, 3, 4, 5, 28, 7, 11};
        constexpr std::uint32_t L = 4;
        std::vector<std::uint32_t> gpu(static_cast<std::size_t>(L) * ColumnHistOnce::alphabet, 0u);
        REQUIRE(ColumnHistOnce::count_host(cipher, L, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == cipher.size());

        StatusOr<std::vector<std::uint32_t>> cpu = HistAlphabetMap::count_column_hist(cipher, L);
        REQUIRE(cpu.ok());
        REQUIRE(gpu == cpu.value());
    }

    SECTION("multi-tile column under fat-tile clamp") {
        constexpr std::size_t T = 8192;
        constexpr std::uint32_t L = 5;
        std::vector<std::uint8_t> cipher(T);
        for (std::size_t i = 0; i < T; ++i) {
            cipher[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
        }

        REQUIRE(ColumnHistOnce::tiles_for_public(T, L) >= 1);

        std::vector<std::uint32_t> gpu(static_cast<std::size_t>(L) * ColumnHistOnce::alphabet, 0u);
        REQUIRE(ColumnHistOnce::count_host(cipher, L, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == T);

        StatusOr<std::vector<std::uint32_t>> cpu = HistAlphabetMap::count_column_hist(cipher, L);
        REQUIRE(cpu.ok());
        REQUIRE(gpu == cpu.value());
    }
}

TEST_CASE("ColumnHistOnce launch_async device path", "[cuda][hist][column]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {0, 0, 1, 28, 7, 3};
    constexpr std::uint32_t L = 3;
    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(cipher);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_cols =
        DeviceBuffer<std::uint32_t>::allocate(static_cast<std::size_t>(L) * ColumnHistOnce::alphabet);
    REQUIRE(device_cols.ok());

    REQUIRE(ColumnHistOnce::launch_async(device_in.value().data(), device_cols.value().data(),
                                         cipher.size(), L)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "ColumnHistOnce test sync").ok());

    std::vector<std::uint32_t> host(static_cast<std::size_t>(L) * ColumnHistOnce::alphabet, 0u);
    REQUIRE(device_cols.value().copy_to_host(host).ok());
    StatusOr<std::vector<std::uint32_t>> cpu = HistAlphabetMap::count_column_hist(cipher, L);
    REQUIRE(cpu.ok());
    REQUIRE(host == cpu.value());
}

TEST_CASE("ColumnHistOnce rejects bad period", "[cuda][hist][column][edge]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {0, 1, 2};
    std::vector<std::uint32_t> cols(29, 0u);
    REQUIRE_FALSE(ColumnHistOnce::count_host(cipher, 0u, cols).ok());
    REQUIRE_FALSE(
        ColumnHistOnce::count_host(cipher, ColumnHistOnce::kMaxPeriod + 1u, cols).ok());
}

#else

TEST_CASE("ColumnHistOnce skipped without CUDA", "[cuda][hist][column]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
