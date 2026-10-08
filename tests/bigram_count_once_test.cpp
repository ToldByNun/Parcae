#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/hist_alphabet_map.hpp"

#include "bigram_count_once.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

TEST_CASE("BigramCountOnce T<2 is zero matrix", "[cuda][hist][bigram]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint32_t> counts(BigramCountOnce::bins, 99u);
    REQUIRE(BigramCountOnce::count_host({}, counts).ok());
    REQUIRE(std::accumulate(counts.begin(), counts.end(), 0u) == 0u);

    REQUIRE(BigramCountOnce::count_host(std::vector<std::uint8_t>{7}, counts).ok());
    REQUIRE(std::accumulate(counts.begin(), counts.end(), 0u) == 0u);
}

TEST_CASE("BigramCountOnce matches HistAlphabetMap count_bigrams", "[cuda][hist][bigram]") {
    REQUIRE(ParcaeCuda::available());

    SECTION("short stream") {
        const std::vector<std::uint8_t> cipher = {0, 1, 2, 3, 4, 5, 28, 7};
        std::vector<std::uint32_t> gpu(BigramCountOnce::bins, 0u);
        REQUIRE(BigramCountOnce::count_host(cipher, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == cipher.size() - 1u);

        StatusOr<HistAlphabetMap::BigramHist> cpu = HistAlphabetMap::count_bigrams(cipher);
        REQUIRE(cpu.ok());
        for (std::size_t i = 0; i < BigramCountOnce::bins; ++i) {
            REQUIRE(gpu[i] == cpu.value()[i]);
        }
    }

    SECTION("multi-tile pairs") {
        constexpr std::size_t T = 4096;
        std::vector<std::uint8_t> cipher(T);
        for (std::size_t i = 0; i < T; ++i) {
            cipher[i] = static_cast<std::uint8_t>((i * 3u + 5u) % 29u);
        }
        REQUIRE(BigramCountOnce::tiles_for_public(T) >= 1);

        std::vector<std::uint32_t> gpu(BigramCountOnce::bins, 0u);
        REQUIRE(BigramCountOnce::count_host(cipher, gpu).ok());
        REQUIRE(std::accumulate(gpu.begin(), gpu.end(), 0u) == T - 1u);

        StatusOr<HistAlphabetMap::BigramHist> cpu = HistAlphabetMap::count_bigrams(cipher);
        REQUIRE(cpu.ok());
        for (std::size_t i = 0; i < BigramCountOnce::bins; ++i) {
            REQUIRE(gpu[i] == cpu.value()[i]);
        }
    }
}

TEST_CASE("BigramCountOnce launch_async device path", "[cuda][hist][bigram]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> cipher = {0, 0, 1, 28, 7, 3};
    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(cipher);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(BigramCountOnce::bins);
    REQUIRE(device_counts.ok());

    REQUIRE(BigramCountOnce::launch_async(device_in.value().data(), device_counts.value().data(),
                                          cipher.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "BigramCountOnce sync").ok());

    std::vector<std::uint32_t> host(BigramCountOnce::bins, 0u);
    REQUIRE(device_counts.value().copy_to_host(host).ok());
    StatusOr<HistAlphabetMap::BigramHist> cpu = HistAlphabetMap::count_bigrams(cipher);
    REQUIRE(cpu.ok());
    for (std::size_t i = 0; i < BigramCountOnce::bins; ++i) {
        REQUIRE(host[i] == cpu.value()[i]);
    }
}

#else

TEST_CASE("BigramCountOnce skipped without CUDA", "[cuda][hist][bigram]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
