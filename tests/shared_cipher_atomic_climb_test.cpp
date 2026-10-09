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
#include "hist_fast.hpp"
#include "hist_tile_cap.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_launch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] bool scores_close(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return false;
        }
        const double denom = std::max(1.0, std::abs(b[i]));
        if (std::abs(a[i] - b[i]) / denom > 1e-9) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("Shared-cipher Atbash/totient tile A/B + restrict/ldg climb",
          "[cuda][hist][atbash][totient][atomic_climb]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(HistFast::production_tile_cap == 64);
    REQUIRE(HistTileCap::kSlotCount == 8);
    REQUIRE(HistTileCap::slot_default(HistTileCap::kAtbash) == 64);
    REQUIRE(HistTileCap::slot_default(HistTileCap::kTotient) == 32);
    // Caesar clamp must stay independent of Atbash A/B.
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 0);
    REQUIRE(HistTileCap::effective(HistTileCap::kCaesar) == 64);

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t C = 512;
    constexpr std::size_t T = 1048576;
    constexpr std::size_t reps = 8;
    const double peak = BenchTierSpec::shared_cipher_compute_roof_rps();
    const double stretch_gate = 0.80 * peak;
    const double done_gate = 0.90 * peak;

    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
    }
    std::vector<std::uint8_t> shifts(T);
    for (std::size_t t = 0; t < T; ++t) {
        shifts[t] = static_cast<std::uint8_t>((t * 3u + 5u) % 29u);
    }
    std::vector<std::uint32_t> begin(C, 0);

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
        DeviceBuffer<std::uint32_t>::from_host(begin);
    REQUIRE(device_begin.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * FamilyChi2Batch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    auto launch_catalog = [&]() -> Status {
        return FamilyChi2Batch::launch_atbash_async(
            device_in.value().data(), device_probs.value().data(), device_counts.value().data(),
            device_scores.value().data(), C, T);
    };
    auto launch_shape = [&]() -> Status {
        return TheoryHistChi2Launch::launch_shape_atbash_async(
            device_in.value().data(), device_probs.value().data(), device_counts.value().data(),
            device_scores.value().data(), C, T);
    };
    auto launch_totient = [&]() -> Status {
        return FamilyChi2Batch::launch_totient_async(
            device_in.value().data(), device_shifts.value().data(), device_begin.value().data(),
            device_probs.value().data(), device_counts.value().data(),
            device_scores.value().data(), C, T);
    };

    // Score parity: catalog vs shape under production cap.
    HistTileCap::set(HistTileCap::kAtbash, 0);
    REQUIRE(launch_catalog().ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "atbash catalog sync").ok());
    std::vector<double> catalog_scores(C, 0.0);
    REQUIRE(device_scores.value().copy_to_host(catalog_scores).ok());
    REQUIRE(launch_shape().ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "atbash shape sync").ok());
    std::vector<double> shape_scores(C, 0.0);
    REQUIRE(device_scores.value().copy_to_host(shape_scores).ok());
    REQUIRE(scores_close(catalog_scores, shape_scores));

    auto time_atbash_cap = [&](int cap) -> double {
        HistTileCap::set(HistTileCap::kAtbash, cap);
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, launch_catalog);
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };
    auto time_shape_cap = [&](int cap) -> double {
        HistTileCap::set(HistTileCap::kAtbash, cap);
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, launch_shape);
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };
    auto time_totient_cap = [&](int cap) -> double {
        HistTileCap::set(HistTileCap::kTotient, cap);
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, launch_totient);
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };

    (void)time_atbash_cap(0); // warm
    (void)time_shape_cap(0);
    (void)time_totient_cap(0);

    const int caps[] = {32, 64, 128};
    double best_atb = 0.0;
    int best_atb_cap = 64;
    double best_tot = 0.0;
    int best_tot_cap = 64;

    std::printf("SHARED_CIPHER_ATOMIC_CLIMB C=%zu T=%zu reps=%zu peak=%.3e done=%.3e\n", C, T, reps,
                peak, done_gate);
    std::printf("cap\tatbash_rps\tatbash_pct\tshape_rps\tshape_pct\ttotient_rps\ttotient_pct\n");

    for (int cap : caps) {
        const double atb = time_atbash_cap(cap);
        const double shp = time_shape_cap(cap);
        const double tot = time_totient_cap(cap);
        std::printf("%d\t%.6e\t%.2f\t%.6e\t%.2f\t%.6e\t%.2f\n", cap, atb, 100.0 * atb / peak, shp,
                    100.0 * shp / peak, tot, 100.0 * tot / peak);
        if (atb > best_atb) {
            best_atb = atb;
            best_atb_cap = cap;
        }
        if (tot > best_tot) {
            best_tot = tot;
            best_tot_cap = cap;
        }
        // A/B samples can spike >100% of Spec on a noisy desktop; model gate is
        // on production defaults below. Spikes ⇒ raise compute roof, never lower.
        // Scores unchanged under fat tiles.
        REQUIRE(launch_catalog().ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "atbash score sync").ok());
        std::vector<double> scores(C, 0.0);
        REQUIRE(device_scores.value().copy_to_host(scores).ok());
        REQUIRE(scores_close(scores, catalog_scores));
    }
    HistTileCap::set(HistTileCap::kAtbash, 0);
    HistTileCap::set(HistTileCap::kTotient, 0);

    const double atb_prod = time_atbash_cap(0);
    const double shp_prod = time_shape_cap(0);
    const double tot_prod = time_totient_cap(0);
    HistTileCap::set(HistTileCap::kAtbash, 0);
    HistTileCap::set(HistTileCap::kTotient, 0);

    std::printf("  prod atbash=%.6e (%.2f%%) shape=%.6e (%.2f%%) totient=%.6e (%.2f%%)\n", atb_prod,
                100.0 * atb_prod / peak, shp_prod, 100.0 * shp_prod / peak, tot_prod,
                100.0 * tot_prod / peak);
    std::printf("  best_atbash cap=%d rps=%.6e  best_totient cap=%d rps=%.6e\n", best_atb_cap,
                best_atb, best_tot_cap, best_tot);
    std::printf("  twin_ratio shape/catalog=%.4f  stretch_hit=%d done_hit=%d\n",
                shp_prod / atb_prod, atb_prod >= stretch_gate ? 1 : 0,
                atb_prod >= done_gate ? 1 : 0);
    std::fflush(stdout);

    REQUIRE(FamilyChi2Batch::tiles_for_atbash(T) == 64);
    REQUIRE(FamilyChi2Batch::tiles_for_totient(T) == 32);
    REQUIRE(HistTileCap::effective(HistTileCap::kCaesar) == 64);
    REQUIRE(atb_prod > 100.0e9);
    REQUIRE(tot_prod > 100.0e9);
    REQUIRE(BenchTierSpec::percent_peak(atb_prod, peak) <= 100.0);
    REQUIRE(BenchTierSpec::percent_peak(shp_prod, peak) <= 100.0);
    REQUIRE(BenchTierSpec::percent_peak(tot_prod, peak) <= 100.0);
    // Twin class: shape ≈ catalog (busy-plate noise band).
    REQUIRE(shp_prod / atb_prod > 0.70);
    REQUIRE(shp_prod / atb_prod < 1.30);
    // Absolute floor: quiet Remap Atbash ~18 TB class; keep well above identity pad.
    REQUIRE(atb_prod > 5.0e12);
}

#else

TEST_CASE("Shared-cipher atomic climb skipped without CUDA",
          "[cuda][hist][atbash][totient][atomic_climb]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
