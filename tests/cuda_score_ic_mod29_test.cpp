#include "parcae/core/index29.hpp"
#include "parcae/score/ic_mod29.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <vector>

#if defined(PARCAE_HAS_CUDA)

#include "ic_mod29_score.hpp"
#include "parcae_cuda.hpp"

static std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

TEST_CASE("CUDA ic_mod29 matches CPU hand vector", "[cuda][score][ic]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> all_same{
        Index29{5},
        Index29{5},
        Index29{5},
        Index29{5},
    };
    StatusOr<double> cpu = IcMod29::score(all_same);
    StatusOr<double> cuda = IcMod29Score::score_host(to_bytes(all_same));
    REQUIRE(cpu.ok());
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value() == cpu.value());
    REQUIRE(cuda.value() == 1.0);
}

TEST_CASE("CUDA ic_mod29 rejects short input", "[cuda][score][ic]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> empty;
    REQUIRE_FALSE(IcMod29Score::score_host(empty).ok());

    const std::vector<std::uint8_t> one{7};
    REQUIRE_FALSE(IcMod29Score::score_host(one).ok());
}

TEST_CASE("CUDA ic_mod29 random parity vs CPU", "[cuda][score][ic]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0x1C0029u);
    std::uniform_int_distribution<int> dist(0, 28);

    for (int trial = 0; trial < 10; ++trial) {
        std::vector<Index29> indices;
        const std::size_t n = 2 + static_cast<std::size_t>(trial) * 31;
        indices.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            indices.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }

        StatusOr<double> cpu = IcMod29::score(indices);
        StatusOr<double> cuda = IcMod29Score::score_host(to_bytes(indices));
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        REQUIRE(cuda.value() == cpu.value());
    }
}

#else

TEST_CASE("CUDA ic_mod29 skipped (PARCAE_HAS_CUDA unset)", "[cuda][score][ic]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise IcMod29Score");
}

#endif
