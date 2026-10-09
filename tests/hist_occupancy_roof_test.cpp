#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/bench/bench_metric.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/bench/bench_timer.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "hist_occupancy_roof.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("HistOccupancyRoof identity vs Atbash shared-cipher plate",
          "[cuda][hist][compute_roof][occupancy]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(HistOccupancyRoof::kOccupancyCandidates == 512u);

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t C = HistOccupancyRoof::kOccupancyCandidates;
    constexpr std::size_t T = 1048576;
    constexpr std::size_t reps = 8;

    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * HistOccupancyRoof::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    auto launch_identity = [&]() -> Status {
        return HistOccupancyRoof::launch_identity_async(
            device_in.value().data(), device_probs.value().data(), device_counts.value().data(),
            device_scores.value().data(), C, T);
    };
    auto launch_atbash = [&]() -> Status {
        return FamilyChi2Batch::launch_atbash_async(
            device_in.value().data(), device_probs.value().data(), device_counts.value().data(),
            device_scores.value().data(), C, T);
    };

    StatusOr<BenchMetric::Sample> id_sample =
        BenchTimer::time_cuda(reps, C, T, launch_identity);
    REQUIRE(id_sample.ok());
    StatusOr<BenchMetric::Sample> at_sample =
        BenchTimer::time_cuda(reps, C, T, launch_atbash);
    REQUIRE(at_sample.ok());

    const double identity_rps = id_sample.value().runes_per_sec();
    const double atbash_rps = at_sample.value().runes_per_sec();
    std::printf("COMPUTE_ROOF_CALIB C=%zu T=%zu reps=%zu\n", C, T, reps);
    std::printf("identity_rps=%.6e  atbash_rps=%.6e  ratio_atbash_over_id=%.4f\n", identity_rps,
                atbash_rps, atbash_rps / identity_rps);
    std::fflush(stdout);

    // Identity occupancy pad vs production Atbash Remap (CipherHistOnce + mirror).
    // Remap amortizes once-hist across C=512 → logical RPS ≫ identity pad.
    REQUIRE(identity_rps > 100.0e9);
    REQUIRE(atbash_rps > 100.0e9);
    REQUIRE(atbash_rps / identity_rps > 2.0);
    REQUIRE(atbash_rps / identity_rps < 20.0);

    // Spec freeze must keep quiet Atbash Remap ≤100% of high-C Remap roof.
    REQUIRE(BenchTierSpec::percent_peak(atbash_rps,
                                        BenchTierSpec::shared_cipher_compute_roof_rps()) <= 100.0);

    // Edge: reject empty.
    REQUIRE_FALSE(HistOccupancyRoof::launch_identity_async(
                       device_in.value().data(), device_probs.value().data(),
                       device_counts.value().data(), device_scores.value().data(), 0, T)
                      .ok());
    REQUIRE_FALSE(HistOccupancyRoof::launch_identity_async(
                       device_in.value().data(), device_probs.value().data(),
                       device_counts.value().data(), device_scores.value().data(), C, 0)
                      .ok());
}

#else

TEST_CASE("HistOccupancyRoof skipped without CUDA", "[cuda][hist][compute_roof]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
