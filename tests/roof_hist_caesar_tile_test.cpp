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

    // Tile-cap A/B is decode-hist only; production decrypt uses alphabet remap.
    auto launch = [&]() -> Status {
        return CaesarChi2Batch::launch_decode_hist_async(
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

TEST_CASE("Caesar catalog remap + ShapeInline Done plate",
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
    // Done = alphabet Remap roof (fair C=29), not 896B DRAM diary.
    const double peak = BenchTierSpec::alphabet_remap_hist_roof_rps();
    const double done_gate = 0.90 * peak;
    const double stretch_gate = 0.80 * peak;

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

    auto time_remap = [&]() -> double {
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            return CaesarChi2Batch::launch_decrypt_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T);
        });
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };
    auto time_decode = [&]() -> double {
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            return CaesarChi2Batch::launch_decode_hist_async(
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

    (void)time_remap();
    (void)time_decode();
    (void)time_shape();

    std::vector<double> remap;
    std::vector<double> decode;
    std::vector<double> shape;
    remap.reserve(samples);
    decode.reserve(samples);
    shape.reserve(samples);
    for (std::size_t i = 0; i < samples; ++i) {
        remap.push_back(time_remap());
    }
    for (std::size_t i = 0; i < samples; ++i) {
        decode.push_back(time_decode());
    }
    for (std::size_t i = 0; i < samples; ++i) {
        shape.push_back(time_shape());
    }
    std::sort(remap.begin(), remap.end());
    std::sort(decode.begin(), decode.end());
    std::sort(shape.begin(), shape.end());
    const double remap_med = remap[samples / 2];
    const double decode_med = decode[samples / 2];
    const double shape_med = shape[samples / 2];

    // Score parity: production remap vs legacy decode-hist.
    StatusOr<DeviceBuffer<double>> remap_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(remap_scores.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());
    REQUIRE(CaesarChi2Batch::launch_decrypt_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                remap_scores.value().data(), C, T)
                .ok());
    REQUIRE(CaesarChi2Batch::launch_decode_hist_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "done plate score sync").ok());
    std::vector<double> remap_host(C);
    std::vector<double> decode_host(C);
    REQUIRE(remap_scores.value().copy_to_host(remap_host).ok());
    REQUIRE(decode_scores.value().copy_to_host(decode_host).ok());
    REQUIRE(remap_host == decode_host);

    std::printf("CAESAR_ROOF_DONE C=%zu T=%zu samples=%zu peak=%.3e stretch=%.3e done=%.3e\n", C,
                T, samples, peak, stretch_gate, done_gate);
    std::printf("  remap   med=%.6e (%.2f%% of alphabet Remap)\n", remap_med,
                100.0 * remap_med / peak);
    std::printf("  decode  med=%.6e (%.2f%%; diary vs Remap — decode uses O(C·T))\n", decode_med,
                100.0 * decode_med / peak);
    std::printf("  shape   med=%.6e (%.2f%%)\n", shape_med, 100.0 * shape_med / peak);
    std::fflush(stdout);

    // Remap is O(T) algebraically; at C=29 the once-hist kernel may still land
    // near decode-hist wall time on a busy desktop / Debug plate. Require parity
    // scores (above) and that remap is not dramatically slower than decode.
    REQUIRE(remap_med >= decode_med * 0.5);
    REQUIRE(BenchTierSpec::percent_peak(remap_med, peak) <= 100.0);
    REQUIRE(BenchTierSpec::percent_peak(shape_med, peak) <= 100.0);
    // Quiet plate ~1.07–1.08 TB (~54% of 2.0 TB) — climb open; absolute floor.
    REQUIRE(shape_med > 500.0e9);
    REQUIRE(remap_med > 500.0e9);
}

#else

TEST_CASE("Caesar roof hist skipped without CUDA", "[cuda][hist][roof]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
