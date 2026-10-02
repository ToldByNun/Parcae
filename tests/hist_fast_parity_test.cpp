#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "hist_fast.hpp"
#include "hist_fast_parity.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

TEST_CASE("HistFast local_shared footprint fits typical shared budgets", "[cuda][hist][local]") {
    REQUIRE(HistFast::alphabet == 29);
    REQUIRE(HistFast::local_stride >= HistFast::alphabet);
    REQUIRE(HistFast::local_shared_uints == HistFast::threads * HistFast::local_stride);
    // 8192 * 4 = 32 KiB — below default 48 KiB shared on sm_120.
    REQUIRE(HistFast::local_shared_uints * sizeof(std::uint32_t) == 32768u);
}

TEST_CASE("HistFastParity identity hist warp == local", "[cuda][hist][parity]") {
    REQUIRE(ParcaeCuda::available());

    SECTION("empty") {
        std::vector<std::uint32_t> warp;
        std::vector<std::uint32_t> local;
        REQUIRE(HistFastParity::compare_identity_hist(nullptr, 0, warp, local).ok());
        REQUIRE(warp.size() == HistFastParity::alphabet);
        REQUIRE(local == warp);
        REQUIRE(std::accumulate(warp.begin(), warp.end(), 0u) == 0u);
    }

    SECTION("short remainder (not multiple of 4)") {
        std::vector<std::uint8_t> cipher = {0, 1, 2, 3, 4, 5, 28};
        std::vector<std::uint32_t> warp;
        std::vector<std::uint32_t> local;
        REQUIRE(HistFastParity::compare_identity_hist(cipher.data(), cipher.size(), warp, local)
                    .ok());
        REQUIRE(local == warp);
        REQUIRE(std::accumulate(warp.begin(), warp.end(), 0u) == cipher.size());
        REQUIRE(warp[0] == 1u);
        REQUIRE(warp[28] == 1u);
    }

    SECTION("multi-tile length") {
        // > 256*4 tokens so tiles_for > 1.
        constexpr std::size_t T = 4096;
        std::vector<std::uint8_t> cipher(T);
        for (std::size_t i = 0; i < T; ++i) {
            cipher[i] = static_cast<std::uint8_t>(i % 29u);
        }
        std::vector<std::uint32_t> warp;
        std::vector<std::uint32_t> local;
        REQUIRE(HistFastParity::compare_identity_hist(cipher.data(), cipher.size(), warp, local)
                    .ok());
        REQUIRE(local == warp);
        REQUIRE(std::accumulate(warp.begin(), warp.end(), 0u) == T);
        // T % 29 != 0 → bins 0..(T%29-1) get one extra count.
        const std::uint32_t base = static_cast<std::uint32_t>(T / 29u);
        const std::uint32_t rem = static_cast<std::uint32_t>(T % 29u);
        for (std::size_t b = 0; b < HistFastParity::alphabet; ++b) {
            const std::uint32_t expect = base + (b < rem ? 1u : 0u);
            REQUIRE(warp[b] == expect);
        }
    }
}

TEST_CASE("HistFastParity caesar hist warp == local", "[cuda][hist][parity]") {
    REQUIRE(ParcaeCuda::available());

    constexpr std::size_t T = 1024;
    std::vector<std::uint8_t> cipher(T);
    for (std::size_t i = 0; i < T; ++i) {
        cipher[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
    }

    for (std::uint8_t shift : {std::uint8_t{0}, std::uint8_t{7}, std::uint8_t{28}}) {
        std::vector<std::uint32_t> warp;
        std::vector<std::uint32_t> local;
        REQUIRE(
            HistFastParity::compare_caesar_hist(cipher.data(), cipher.size(), shift, warp, local)
                .ok());
        REQUIRE(local == warp);
        REQUIRE(std::accumulate(warp.begin(), warp.end(), 0u) == T);

        // CPU oracle for one shift.
        std::vector<std::uint32_t> cpu(HistFastParity::alphabet, 0u);
        for (std::uint8_t x : cipher) {
            cpu[HistFast::dec_caesar(x, shift)] += 1u;
        }
        REQUIRE(warp == cpu);
    }
}

TEST_CASE("HistFastParity rejects bad shift", "[cuda][hist][parity]") {
    REQUIRE(ParcaeCuda::available());
    std::vector<std::uint8_t> cipher = {0, 1, 2};
    std::vector<std::uint32_t> warp;
    std::vector<std::uint32_t> local;
    REQUIRE_FALSE(HistFastParity::compare_caesar_hist(cipher.data(), cipher.size(), 29u, warp, local)
                      .ok());
}

#endif
