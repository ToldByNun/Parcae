#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

#include "parcae/core/index29.hpp"
#include "parcae/score/self_repeat_rate.hpp"

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"
#include "self_repeat_rate_score.hpp"

static std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

TEST_CASE("CUDA self_repeat_rate matches CPU hand vectors", "[cuda][score][self_repeat]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> all_same{Index29{3}, Index29{3}, Index29{3}, Index29{3}};
    StatusOr<double> cpu_same = SelfRepeatRate::score(all_same);
    StatusOr<double> cuda_same = SelfRepeatRateScore::score_host(to_bytes(all_same));
    REQUIRE(cpu_same.ok());
    REQUIRE(cuda_same.ok());
    REQUIRE(cuda_same.value() == cpu_same.value());
    REQUIRE(cuda_same.value() == 1.0);

    const std::vector<Index29> none{
        Index29{0}, Index29{1}, Index29{2}, Index29{3}, Index29{4},
    };
    StatusOr<double> cpu_none = SelfRepeatRate::score(none);
    StatusOr<double> cuda_none = SelfRepeatRateScore::score_host(to_bytes(none));
    REQUIRE(cpu_none.ok());
    REQUIRE(cuda_none.ok());
    REQUIRE(cuda_none.value() == cpu_none.value());
    REQUIRE(cuda_none.value() == 0.0);

    const std::vector<Index29> mixed{
        Index29{7}, Index29{7}, Index29{1}, Index29{1}, Index29{1},
    };
    StatusOr<double> cpu_mixed = SelfRepeatRate::score(mixed);
    StatusOr<double> cuda_mixed = SelfRepeatRateScore::score_host(to_bytes(mixed));
    REQUIRE(cpu_mixed.ok());
    REQUIRE(cuda_mixed.ok());
    REQUIRE(cuda_mixed.value() == cpu_mixed.value());
    REQUIRE(cuda_mixed.value() == 0.75);
}

TEST_CASE("CUDA self_repeat_rate rejects short input", "[cuda][score][self_repeat]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> empty;
    REQUIRE_FALSE(SelfRepeatRateScore::score_host(empty).ok());

    const std::vector<std::uint8_t> one{9};
    REQUIRE_FALSE(SelfRepeatRateScore::score_host(one).ok());
}

TEST_CASE("CUDA self_repeat_rate random parity vs CPU", "[cuda][score][self_repeat]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0x5E1F29u);
    std::uniform_int_distribution<int> dist(0, 28);

    for (int trial = 0; trial < 10; ++trial) {
        std::vector<Index29> indices;
        const std::size_t n = 2 + static_cast<std::size_t>(trial) * 37;
        indices.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            indices.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }

        StatusOr<double> cpu = SelfRepeatRate::score(indices);
        StatusOr<double> cuda = SelfRepeatRateScore::score_host(to_bytes(indices));
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        REQUIRE(cuda.value() == cpu.value());
    }
}

#else

TEST_CASE(
    "CUDA self_repeat_rate skipped (PARCAE_HAS_CUDA unset)",
    "[cuda][score][self_repeat]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise SelfRepeatRateScore");
}

#endif
