#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <parcae/bench/bench_timer.hpp>
#include <parcae/core/status.hpp>
#include <string>

TEST_CASE("BenchTimer protocol constants", "[bench][timer]") {
    REQUIRE(BenchTimer::warmup_launches == 4);
    REQUIRE(BenchTimer::timed_samples == 3);
}

TEST_CASE("BenchTimer time_cpu_once counts launches and returns rates", "[bench][timer]") {
    std::atomic<std::uint32_t> hits{0};
    StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cpu_once(
        /*repeats=*/8,
        /*candidates=*/4,
        /*tokens=*/16, [&]() -> Status {
            hits.fetch_add(1, std::memory_order_relaxed);
            return Status::success();
        });
    REQUIRE(sample.ok());
    REQUIRE(hits.load() == 8u);
    REQUIRE(sample.value().wall_seconds() >= 0.0);
    REQUIRE(sample.value().runes_per_sec() >= 0.0);
    REQUIRE(sample.value().keys_per_sec() >= 0.0);
    // Deterministic rate check when wall is positive: runes = 8*4*16 = 512
    if (sample.value().wall_seconds() > 0.0) {
        const double expected = BenchMetric::runes_per_sec(8, 4, 16, sample.value().wall_seconds());
        REQUIRE(sample.value().runes_per_sec() == expected);
    }
}

TEST_CASE("BenchTimer time_cpu does warmup plus three timed windows", "[bench][timer]") {
    std::atomic<std::uint32_t> hits{0};
    constexpr std::size_t reps = 2;
    StatusOr<BenchMetric::Sample> sample =
        BenchTimer::time_cpu(reps,
                             /*candidates=*/1,
                             /*tokens=*/1, [&]() -> Status {
                                 hits.fetch_add(1, std::memory_order_relaxed);
                                 return Status::success();
                             });
    REQUIRE(sample.ok());
    // 4 warmups + 3 samples × 2 reps = 4 + 6 = 10
    const std::uint32_t expected =
        static_cast<std::uint32_t>(BenchTimer::warmup_launches) +
        static_cast<std::uint32_t>(BenchTimer::timed_samples) * static_cast<std::uint32_t>(reps);
    REQUIRE(hits.load() == expected);
    REQUIRE(sample.value().runes_per_sec() >= 0.0);
}

TEST_CASE("BenchTimer time_cpu propagates launch errors", "[bench][timer]") {
    StatusOr<BenchMetric::Sample> sample =
        BenchTimer::time_cpu(1, 1, 1, []() -> Status { return Status::error("bench boom"); });
    REQUIRE_FALSE(sample.ok());
    REQUIRE(sample.status().message().find("bench boom") != std::string::npos);
}

TEST_CASE("BenchTimer time_cpu_once propagates launch errors", "[bench][timer]") {
    int calls = 0;
    StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cpu_once(3, 1, 1, [&]() -> Status {
        ++calls;
        if (calls == 2) {
            return Status::error("second fail");
        }
        return Status::success();
    });
    REQUIRE_FALSE(sample.ok());
    REQUIRE(calls == 2);
}

#if defined(PARCAE_HAS_CUDA)
TEST_CASE("BenchTimer time_cuda empty launches follow median-of-3 protocol",
          "[bench][timer][cuda]") {
    std::atomic<std::uint32_t> hits{0};
    constexpr std::size_t reps = 2;
    StatusOr<BenchMetric::Sample> sample =
        BenchTimer::time_cuda(reps,
                              /*candidates=*/8,
                              /*tokens=*/32, [&]() -> Status {
                                  hits.fetch_add(1, std::memory_order_relaxed);
                                  return Status::success();
                              });
    REQUIRE(sample.ok());
    const std::uint32_t expected =
        static_cast<std::uint32_t>(BenchTimer::warmup_launches) +
        static_cast<std::uint32_t>(BenchTimer::timed_samples) * static_cast<std::uint32_t>(reps);
    REQUIRE(hits.load() == expected);
    REQUIRE(sample.value().wall_seconds() >= 0.0);
    REQUIRE(std::isfinite(sample.value().runes_per_sec()));
}
#endif
