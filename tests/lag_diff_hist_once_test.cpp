#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/hist_alphabet_map.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "lag_diff_hist_once.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

TEST_CASE("LagDiffHistOnce matches HistAlphabetMap and sum == max(0,T-L)",
          "[cuda][hist][lag]") {
    REQUIRE(ParcaeCuda::available());

    SECTION("short tail") {
        const std::vector<std::uint8_t> cipher = {0, 1, 2, 3, 4, 5, 28, 7, 11};
        constexpr std::uint32_t L = 3;
        std::vector<std::uint32_t> gpu(LagDiffHistOnce::alphabet, 0u);
        REQUIRE(LagDiffHistOnce::count_host(cipher, L, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == cipher.size() - L);

        StatusOr<HistAlphabetMap::Hist> cpu = HistAlphabetMap::count_lag_diff_hist(cipher, L);
        REQUIRE(cpu.ok());
        for (std::size_t b = 0; b < LagDiffHistOnce::alphabet; ++b) {
            REQUIRE(gpu[b] == cpu.value()[b]);
        }
    }

    SECTION("multi-tile tail") {
        constexpr std::size_t T = 8192;
        constexpr std::uint32_t L = 7;
        std::vector<std::uint8_t> cipher(T);
        for (std::size_t i = 0; i < T; ++i) {
            cipher[i] = static_cast<std::uint8_t>((i * 3u + 11u) % 29u);
        }
        REQUIRE(LagDiffHistOnce::tiles_for_public(T, L) >= 1);

        std::vector<std::uint32_t> gpu(LagDiffHistOnce::alphabet, 0u);
        REQUIRE(LagDiffHistOnce::count_host(cipher, L, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == T - L);

        StatusOr<HistAlphabetMap::Hist> cpu = HistAlphabetMap::count_lag_diff_hist(cipher, L);
        REQUIRE(cpu.ok());
        for (std::size_t b = 0; b < LagDiffHistOnce::alphabet; ++b) {
            REQUIRE(gpu[b] == cpu.value()[b]);
        }
    }
}

TEST_CASE("LagDiffHistOnce L>=T is zero hist", "[cuda][hist][lag][edge]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {1, 2, 3, 4};
    std::vector<std::uint32_t> gpu(LagDiffHistOnce::alphabet, 99u);

    REQUIRE(LagDiffHistOnce::count_host(cipher, 4u, gpu).ok());
    REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == 0u);

    REQUIRE(LagDiffHistOnce::count_host(cipher, 10u, gpu).ok());
    REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == 0u);
}

TEST_CASE("LagDiffHistOnce rejects lag 0", "[cuda][hist][lag][edge]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {0, 1, 2};
    std::vector<std::uint32_t> counts(29, 0u);
    REQUIRE_FALSE(LagDiffHistOnce::count_host(cipher, 0u, counts).ok());
    REQUIRE_FALSE(LagDiffHistOnce::count_sum_host(cipher, 0u, counts).ok());
}

TEST_CASE("LagDiffHistOnce sum matches HistAlphabetMap count_lag_sum_hist",
          "[cuda][hist][lag][sum]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {0, 1, 2, 3, 4, 5, 28, 7, 11, 13};
    constexpr std::uint32_t L = 4;
    std::vector<std::uint32_t> gpu(LagDiffHistOnce::alphabet, 0u);
    REQUIRE(LagDiffHistOnce::count_sum_host(cipher, L, gpu).ok());
    REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == cipher.size() - L);

    StatusOr<HistAlphabetMap::Hist> cpu = HistAlphabetMap::count_lag_sum_hist(cipher, L);
    REQUIRE(cpu.ok());
    for (std::size_t b = 0; b < LagDiffHistOnce::alphabet; ++b) {
        REQUIRE(gpu[b] == cpu.value()[b]);
    }
}

TEST_CASE("LagDiffHistOnce launch_async device path", "[cuda][hist][lag]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {0, 5, 10, 15, 20, 25, 28};
    constexpr std::uint32_t L = 2;
    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(cipher);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(LagDiffHistOnce::alphabet);
    REQUIRE(device_counts.ok());

    REQUIRE(LagDiffHistOnce::launch_async(device_in.value().data(), device_counts.value().data(),
                                          cipher.size(), L)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "LagDiffHistOnce sync").ok());

    std::vector<std::uint32_t> host(LagDiffHistOnce::alphabet, 0u);
    REQUIRE(device_counts.value().copy_to_host(host).ok());
    StatusOr<HistAlphabetMap::Hist> cpu = HistAlphabetMap::count_lag_diff_hist(cipher, L);
    REQUIRE(cpu.ok());
    for (std::size_t b = 0; b < LagDiffHistOnce::alphabet; ++b) {
        REQUIRE(host[b] == cpu.value()[b]);
    }
}

#else

TEST_CASE("LagDiffHistOnce skipped without CUDA", "[cuda][hist][lag]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
