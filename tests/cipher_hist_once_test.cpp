#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/hist_alphabet_map.hpp"

#include "cipher_hist_once.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast_parity.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

TEST_CASE("CipherHistOnce empty stream is zero hist", "[cuda][hist][once]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint32_t> counts(CipherHistOnce::alphabet, 99u);
    REQUIRE(CipherHistOnce::count_host({}, counts).ok());
    REQUIRE(std::accumulate(counts.begin(), counts.end(), 0u) == 0u);
    for (std::uint32_t c : counts) {
        REQUIRE(c == 0u);
    }
}

TEST_CASE("CipherHistOnce sum equals T and matches HistAlphabetMap", "[cuda][hist][once]") {
    REQUIRE(ParcaeCuda::available());

    SECTION("short remainder not multiple of 4") {
        const std::vector<std::uint8_t> cipher = {0, 1, 2, 3, 4, 5, 28};
        std::vector<std::uint32_t> gpu(CipherHistOnce::alphabet, 0u);
        REQUIRE(CipherHistOnce::count_host(cipher, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == cipher.size());

        StatusOr<HistAlphabetMap::Hist> cpu = HistAlphabetMap::count_stream_hist(cipher);
        REQUIRE(cpu.ok());
        for (std::size_t b = 0; b < CipherHistOnce::alphabet; ++b) {
            REQUIRE(gpu[b] == cpu.value()[b]);
        }
    }

    SECTION("multi-tile length under fat-tile clamp") {
        constexpr std::size_t T = 4096;
        std::vector<std::uint8_t> cipher(T);
        for (std::size_t i = 0; i < T; ++i) {
            cipher[i] = static_cast<std::uint8_t>(i % 29u);
        }

        std::vector<std::uint32_t> gpu(CipherHistOnce::alphabet, 0u);
        REQUIRE(CipherHistOnce::count_host(cipher, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == T);
        REQUIRE(CipherHistOnce::tiles_for_public(T) >= 1);

        StatusOr<HistAlphabetMap::Hist> cpu = HistAlphabetMap::count_stream_hist(cipher);
        REQUIRE(cpu.ok());
        for (std::size_t b = 0; b < CipherHistOnce::alphabet; ++b) {
            REQUIRE(gpu[b] == cpu.value()[b]);
        }
    }

    SECTION("fair-sized T with grid-stride beyond tile cap") {
        // packs = T/4 = 262144, work tiles = 1024, clamped to 64 → must stride.
        constexpr std::size_t T = 1048576;
        std::vector<std::uint8_t> cipher(T);
        for (std::size_t i = 0; i < T; ++i) {
            cipher[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
        }

        REQUIRE(CipherHistOnce::tiles_for_public(T) == CipherHistOnce::kProductionTileCap);

        std::vector<std::uint32_t> gpu(CipherHistOnce::alphabet, 0u);
        REQUIRE(CipherHistOnce::count_host(cipher, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == T);

        StatusOr<HistAlphabetMap::Hist> cpu = HistAlphabetMap::count_stream_hist(cipher);
        REQUIRE(cpu.ok());
        for (std::size_t b = 0; b < CipherHistOnce::alphabet; ++b) {
            REQUIRE(gpu[b] == cpu.value()[b]);
        }
    }
}

TEST_CASE("CipherHistOnce matches HistFastParity warp identity hist", "[cuda][hist][once][parity]") {
    REQUIRE(ParcaeCuda::available());

    constexpr std::size_t T = 2048;
    std::vector<std::uint8_t> cipher(T);
    for (std::size_t i = 0; i < T; ++i) {
        cipher[i] = static_cast<std::uint8_t>((i * 5u + 11u) % 29u);
    }

    std::vector<std::uint32_t> once(CipherHistOnce::alphabet, 0u);
    REQUIRE(CipherHistOnce::count_host(cipher, once).ok());

    std::vector<std::uint32_t> warp;
    std::vector<std::uint32_t> local;
    REQUIRE(HistFastParity::compare_identity_hist(cipher.data(), cipher.size(), warp, local).ok());
    REQUIRE(local == warp);
    REQUIRE(once == warp);
    REQUIRE(std::accumulate(once.begin(), once.end(), 0u) == T);
}

TEST_CASE("CipherHistOnce launch_async device path", "[cuda][hist][once]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {0, 0, 1, 28, 7};
    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(cipher);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(CipherHistOnce::alphabet);
    REQUIRE(device_counts.ok());

    REQUIRE(CipherHistOnce::launch_async(device_in.value().data(), device_counts.value().data(),
                                         cipher.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "CipherHistOnce test sync").ok());

    std::vector<std::uint32_t> host(CipherHistOnce::alphabet, 0u);
    REQUIRE(device_counts.value().copy_to_host(host).ok());
    REQUIRE(std::accumulate(host.begin(), host.end(), 0u) == cipher.size());
    REQUIRE(host[0] == 2u);
    REQUIRE(host[1] == 1u);
    REQUIRE(host[7] == 1u);
    REQUIRE(host[28] == 1u);
}

TEST_CASE("CipherHistOnce rejects bad args", "[cuda][hist][once][edge]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint32_t> bad_size(10, 0u);
    const std::vector<std::uint8_t> cipher = {1, 2, 3};
    REQUIRE_FALSE(CipherHistOnce::count_host(cipher, bad_size).ok());

    StatusOr<DeviceBuffer<std::uint32_t>> counts =
        DeviceBuffer<std::uint32_t>::allocate(CipherHistOnce::alphabet);
    REQUIRE(counts.ok());
    REQUIRE_FALSE(CipherHistOnce::launch_async(nullptr, counts.value().data(), 4u).ok());
    REQUIRE_FALSE(CipherHistOnce::launch_async(reinterpret_cast<const std::uint8_t*>(1), nullptr, 4u)
                      .ok());
}

#endif
