#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/bench/bench_metric.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/bench/bench_timer.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>

#include "autokey_ring_device.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"
#include "hist_tile_cap.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_launch.hpp"
#include "theory_hist_chi2_s4.hpp"
#include "theory_hist_chi2_s5.hpp"

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

[[nodiscard]] std::uint8_t host_autokey_out(std::uint8_t x, std::size_t t, std::uint8_t lag,
                                            const std::uint8_t* stream) {
    const std::uint8_t key = AutokeyRingDevice::shift(stream, t, lag);
    return HistFast::dec_sub(x, key);
}

} // namespace

TEST_CASE("S4 AutokeyRing uchar4 pack + tile A/B; S5 poly fair Spec",
          "[cuda][hist][s4][s5][autokey][uchar4]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(HistTileCap::kSlotCount == 8);
    REQUIRE(HistTileCap::slot_default(HistTileCap::kS4) == 32);
    REQUIRE(HistTileCap::slot_default(HistTileCap::kS5) == 32);
    REQUIRE(HistTileCap::effective(HistTileCap::kCaesar) == 64);

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t C4 = 28;
    constexpr std::size_t C5 = 841;
    constexpr std::size_t T = 1048576;
    constexpr std::size_t reps = 8;
    const double peak_s4 = BenchTierSpec::lag_remap_hist_roof_rps();
    const double peak_s5 = BenchTierSpec::column_remap_hist_roof_rps();

    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
    }
    std::vector<std::uint8_t> lags(C4);
    for (std::size_t c = 0; c < C4; ++c) {
        lags[c] = static_cast<std::uint8_t>((c % 28) + 1);
    }
    std::vector<std::uint8_t> b0(C5);
    std::vector<std::uint8_t> b1(C5);
    std::vector<std::uint8_t> b2(C5);
    for (std::size_t c = 0; c < C5; ++c) {
        b0[c] = static_cast<std::uint8_t>(c % 29);
        b1[c] = static_cast<std::uint8_t>((c / 29) % 29);
        b2[c] = static_cast<std::uint8_t>((c % 28) + 1);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_lags =
        DeviceBuffer<std::uint8_t>::from_host(lags);
    REQUIRE(device_lags.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b0 = DeviceBuffer<std::uint8_t>::from_host(b0);
    REQUIRE(device_b0.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b1 = DeviceBuffer<std::uint8_t>::from_host(b1);
    REQUIRE(device_b1.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b2 = DeviceBuffer<std::uint8_t>::from_host(b2);
    REQUIRE(device_b2.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts4 =
        DeviceBuffer<std::uint32_t>::allocate(C4 * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts4.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts5 =
        DeviceBuffer<std::uint32_t>::allocate(C5 * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts5.ok());
    StatusOr<DeviceBuffer<double>> device_scores4 = DeviceBuffer<double>::allocate(C4);
    REQUIRE(device_scores4.ok());
    StatusOr<DeviceBuffer<double>> device_scores5 = DeviceBuffer<double>::allocate(C5);
    REQUIRE(device_scores5.ok());

    auto launch_s4 = [&]() -> Status {
        return TheoryHistChi2Launch::launch_s4_autokey_async(
            device_in.value().data(), device_lags.value().data(), device_probs.value().data(),
            device_counts4.value().data(), device_scores4.value().data(), C4, T,
            /*cipher_minus_ks=*/true);
    };
    auto launch_s5 = [&]() -> Status {
        return TheoryHistChi2Launch::launch_s5_poly_async(
            device_in.value().data(), device_b0.value().data(), device_b1.value().data(),
            device_b2.value().data(), device_probs.value().data(), device_counts5.value().data(),
            device_scores5.value().data(), C5, T, /*cipher_minus_ks=*/true);
    };

    // Host oracle spot-check (lag=5, first 64 tokens) — scores path uses full hist.
    {
        constexpr std::uint8_t lag = 5;
        for (std::size_t t = 0; t < 64; ++t) {
            const std::uint8_t got = host_autokey_out(host_in[t], t, lag, host_in.data());
            const std::uint8_t key = AutokeyRingDevice::shift(host_in.data(), t, lag);
            REQUIRE(got == HistFast::dec_sub(host_in[t], key));
            if (t < lag) {
                REQUIRE(key == 0);
            } else {
                REQUIRE(key == host_in[t - lag]);
            }
        }
    }

    HistTileCap::set(HistTileCap::kS4, 0);
    REQUIRE(launch_s4().ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s4 sync").ok());
    std::vector<double> s4_base(C4, 0.0);
    REQUIRE(device_scores4.value().copy_to_host(s4_base).ok());

    auto time_s4 = [&](int cap) -> double {
        HistTileCap::set(HistTileCap::kS4, cap);
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C4, T, launch_s4);
        REQUIRE(sample.ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s4 score sync").ok());
        std::vector<double> scores(C4, 0.0);
        REQUIRE(device_scores4.value().copy_to_host(scores).ok());
        REQUIRE(scores_close(scores, s4_base));
        return sample.value().runes_per_sec();
    };
    auto time_s5 = [&](int cap) -> double {
        HistTileCap::set(HistTileCap::kS5, cap);
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C5, T, launch_s5);
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };

    (void)time_s4(0);
    (void)time_s5(0);

    const int caps[] = {32, 64, 128};
    double best_s4 = 0.0;
    int best_s4_cap = 64;
    double best_s5 = 0.0;
    int best_s5_cap = 32;

    std::printf("AUTOKEY_POLY_UCHAR4 C4=%zu C5=%zu T=%zu peak_s4=%.3e peak_s5=%.3e\n", C4, C5, T,
                peak_s4, peak_s5);
    std::printf("cap\ts4_rps\ts4_pct\ts5_rps\ts5_pct\n");
    for (int cap : caps) {
        const double r4 = time_s4(cap);
        const double r5 = time_s5(cap);
        std::printf("%d\t%.6e\t%.2f\t%.6e\t%.2f\n", cap, r4, 100.0 * r4 / peak_s4, r5,
                    100.0 * r5 / peak_s5);
        if (r4 > best_s4) {
            best_s4 = r4;
            best_s4_cap = cap;
        }
        if (r5 > best_s5) {
            best_s5 = r5;
            best_s5_cap = cap;
        }
    }
    HistTileCap::set(HistTileCap::kS4, 0);
    HistTileCap::set(HistTileCap::kS5, 0);

    const double s4_prod = time_s4(0);
    const double s5_prod = time_s5(0);
    HistTileCap::set(HistTileCap::kS4, 0);
    HistTileCap::set(HistTileCap::kS5, 0);

    std::printf("  prod s4=%.6e (%.2f%%) s5=%.6e (%.2f%%) best_s4_cap=%d best_s5_cap=%d\n",
                s4_prod, 100.0 * s4_prod / peak_s4, s5_prod, 100.0 * s5_prod / peak_s5,
                best_s4_cap, best_s5_cap);
    std::fflush(stdout);

    REQUIRE(TheoryHistChi2S4::tiles_for_public(T) == 32);
    REQUIRE(TheoryHistChi2S5::tiles_for_public(T) == 32);
    REQUIRE(BenchTierSpec::percent_peak(s4_prod, peak_s4) <= 100.0);
    REQUIRE(BenchTierSpec::percent_peak(s5_prod, peak_s5) <= 100.0);
    REQUIRE(s5_prod > 100.0e9);
    // Quiet lag-remap plate ~28–29B (climb vs 2.0 TB lag roof); keep above noise.
    REQUIRE(s4_prod > 10.0e9);
    // Column Remap quiet class ~18–22 TB; keep well above decode-era ~400B floor.
    REQUIRE(s5_prod > 400.0e9);
    REQUIRE(HistTileCap::effective(HistTileCap::kCaesar) == 64);
}

#else

TEST_CASE("S4/S5 uchar4 climb skipped without CUDA", "[cuda][hist][s4][s5][autokey][uchar4]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
