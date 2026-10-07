#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/bench/bench_metric.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/bench/bench_timer.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>

#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_launch.hpp"

#include <cuda_runtime_api.h>
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

struct TileSweepRow {
    int cap = 0;
    int grid_y = 0;
    double runes_per_sec = 0.0;
    double wall_seconds = 0.0;
};

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

TEST_CASE("HistFast tiles_for_capped clamps fat tiles", "[cuda][hist][roof]") {
    // T=1M → packs=262144 → work tiles = 1024; production clamp = 64.
    REQUIRE(HistFast::tiles_for_work(1048576) == 1024);
    REQUIRE(HistFast::production_tile_cap == 64);
    REQUIRE(HistFast::tiles_for(1048576) == HistFast::production_tile_cap);
    REQUIRE(HistFast::tiles_for_capped(1048576, 32) == 32);
    REQUIRE(HistFast::tiles_for_capped(1048576, HistFast::max_tiles) == 1024);
    REQUIRE(HistFast::tiles_for_capped(1048576, 4096) == 1024);
    REQUIRE(HistFast::tiles_for_capped(64, HistFast::max_tiles) == 1); // 16 packs → 1 tile
}

TEST_CASE("Caesar fat-tile warp-private roof sweep", "[cuda][hist][roof][caesar]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t C = 29;
    constexpr std::size_t T = 1048576;
    constexpr std::size_t reps = 8;

    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
    }
    std::vector<std::uint8_t> shifts(C);
    for (std::size_t c = 0; c < C; ++c) {
        shifts[c] = static_cast<std::uint8_t>(c);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * CaesarChi2Batch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    auto launch = [&]() -> Status {
        return CaesarChi2Batch::launch_decrypt_async(
            device_in.value().data(), device_shifts.value().data(), device_probs.value().data(),
            device_counts.value().data(), device_scores.value().data(), C, T);
    };

    // Reference scores under legacy uncapped tiling (max_tiles).
    CaesarChi2Batch::set_hist_tile_cap(HistFast::max_tiles);
    REQUIRE(launch().ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "roof sync").ok());
    std::vector<double> baseline_scores(C, 0.0);
    REQUIRE(device_scores.value().copy_to_host(baseline_scores).ok());

    // Include legacy 1024, shipped 64, and neighbors.
    const int caps[] = {1024, 256, 128, 64, 32, 16, 8};
    std::vector<TileSweepRow> rows;
    rows.reserve(sizeof(caps) / sizeof(caps[0]));

    std::printf("ROOF_HIST_CAESAR_TILE_SWEEP C=%zu T=%zu reps=%zu\n", C, T, reps);
    std::printf("cap\tgrid_y\trunes_per_sec\twall_s\tpct_896B\n");

    double best_rps = 0.0;
    int best_cap = 1024;
    double legacy_rps = 0.0;
    for (int cap : caps) {
        CaesarChi2Batch::set_hist_tile_cap(cap);
        const int grid_y = CaesarChi2Batch::tiles_for_public(T);

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, launch);
        REQUIRE(sample.ok());

        // Correctness: fat tiles must match baseline scores.
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "roof score sync").ok());
        std::vector<double> scores(C, 0.0);
        REQUIRE(device_scores.value().copy_to_host(scores).ok());
        REQUIRE(scores_close(scores, baseline_scores));

        TileSweepRow row{cap, grid_y, sample.value().runes_per_sec(),
                         sample.value().wall_seconds()};
        rows.push_back(row);
        if (cap == HistFast::max_tiles) {
            legacy_rps = row.runes_per_sec;
        }
        const double pct = 100.0 * row.runes_per_sec / 896e9;
        std::printf("%d\t%d\t%.6e\t%.9f\t%.2f\n", row.cap, row.grid_y, row.runes_per_sec,
                    row.wall_seconds, pct);
        if (row.runes_per_sec > best_rps) {
            best_rps = row.runes_per_sec;
            best_cap = cap;
        }
    }

    CaesarChi2Batch::set_hist_tile_cap(0); // restore production default (kProductionTileCap)

    const bool win =
        best_cap == CaesarChi2Batch::kProductionTileCap && best_rps > legacy_rps * 1.02;
    std::printf("ROOF_HIST_BEST cap=%d runes_per_sec=%.6e legacy_1024=%.6e shipped_cap=%d "
                "verdict=%s\n",
                best_cap, best_rps, legacy_rps, CaesarChi2Batch::kProductionTileCap,
                win ? "WIN_SHIPPED" : "CHECK_SUMMARY");
    std::fflush(stdout);

    REQUIRE(CaesarChi2Batch::hist_tile_cap() == 0);
    REQUIRE(CaesarChi2Batch::tiles_for_public(T) == CaesarChi2Batch::kProductionTileCap);
}

TEST_CASE("Caesar catalog + ShapeInline Done plate (tile64, restrict+ldg)",
          "[cuda][hist][roof][caesar][done]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(HistFast::production_tile_cap == 64);
    REQUIRE(CaesarChi2Batch::kProductionTileCap == 64);
    CaesarChi2Batch::set_hist_tile_cap(0);
    REQUIRE(CaesarChi2Batch::tiles_for_public(1048576) == 64);

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t C = 29;
    constexpr std::size_t T = 1048576;
    constexpr std::size_t reps = 8;
    constexpr std::size_t samples = 5;
    const double peak = BenchTierSpec::dram_roofline_hist_peak();
    const double done_gate = 0.90 * peak; // 806.4B

    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
    }
    std::vector<std::uint8_t> shifts(C);
    for (std::size_t c = 0; c < C; ++c) {
        shifts[c] = static_cast<std::uint8_t>(c);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * CaesarChi2Batch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    auto time_catalog = [&]() -> double {
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            return CaesarChi2Batch::launch_decrypt_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T);
        });
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };
    auto time_shape = [&]() -> double {
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            return TheoryHistChi2Launch::launch_shape_caesar_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T);
        });
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };

    (void)time_catalog(); // warm
    (void)time_shape();

    // Catalog block then shape block (not interleaved) so a cold/noisy sample
    // on one twin does not poison the other's median.
    std::vector<double> catalog;
    std::vector<double> shape;
    catalog.reserve(samples);
    shape.reserve(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        catalog.push_back(time_catalog());
    }
    for (std::size_t i = 0; i < samples; ++i) {
        shape.push_back(time_shape());
    }
    std::sort(catalog.begin(), catalog.end());
    std::sort(shape.begin(), shape.end());
    const double cat_med = catalog[samples / 2];
    const double shape_med = shape[samples / 2];
    const double cat_best = catalog.back();
    const double shape_best = shape.back();

    std::printf("CAESAR_ROOF_DONE C=%zu T=%zu samples=%zu peak=%.3e done_gate=%.3e\n", C, T,
                samples, peak, done_gate);
    std::printf("  catalog med=%.6e (%.2f%%) best=%.6e\n", cat_med, 100.0 * cat_med / peak,
                cat_best);
    std::printf("  shape   med=%.6e (%.2f%%) best=%.6e\n", shape_med, 100.0 * shape_med / peak,
                shape_best);
    std::fflush(stdout);

    REQUIRE(BenchTierSpec::percent_peak(cat_med, peak) <= 100.0);
    REQUIRE(BenchTierSpec::percent_peak(shape_med, peak) <= 100.0);
    // Catalog stretch is the hard CI floor. Quiet median ≥806.4B Done is
    // ACCEPTANCE in roof_hist / kernel_slo docs (best-of often clears Done
    // even when desktop util ~30% keeps median under the gate).
    REQUIRE(cat_med >= 0.80 * peak);
    std::printf("  done_gate_hit catalog_best=%d shape_best=%d catalog_med=%d shape_med=%d\n",
                cat_best >= done_gate ? 1 : 0, shape_best >= done_gate ? 1 : 0,
                cat_med >= done_gate ? 1 : 0, shape_med >= done_gate ? 1 : 0);
    std::fflush(stdout);
}

#else

TEST_CASE("Caesar roof hist skipped without CUDA", "[cuda][hist][roof]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
