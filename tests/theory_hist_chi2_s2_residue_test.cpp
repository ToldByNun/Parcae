#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/bench/bench_metric.hpp>
#include <parcae/bench/bench_tier_spec.hpp>
#include <parcae/bench/bench_timer.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_tile_cap.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_s2.hpp"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

using Catch::Approx;

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

/// Host helpers for S2 residue Catch2 (no C++ namespaces).
class TheoryHistChi2S2ResidueTest {
public:
    [[nodiscard]] static StatusOr<ExpectedFrequencyTable> load_freqs() {
        return ExpectedFrequencyLoader::load_from_file(
            std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    }

    /// Host oracle: same math as S2 decrypt (`x - (b0 + b1·(t mod 29))` mod 29).
    [[nodiscard]] static std::vector<double>
    host_s2_chi2(const std::vector<std::uint8_t>& cipher, const std::vector<std::uint8_t>& b0,
                 const std::vector<std::uint8_t>& b1, const ExpectedFrequencyTable& freqs) {
        const std::size_t C = b0.size();
        const std::size_t T = cipher.size();
        std::vector<double> scores(C, 0.0);
        const auto& p = freqs.probabilities();
        for (std::size_t c = 0; c < C; ++c) {
            std::uint32_t counts[29] = {};
            for (std::size_t t = 0; t < T; ++t) {
                const std::uint8_t ks = static_cast<std::uint8_t>(
                    (static_cast<unsigned>(b0[c]) +
                     static_cast<unsigned>(b1[c]) * static_cast<unsigned>(t % 29u)) %
                    29u);
                const std::uint8_t out = static_cast<std::uint8_t>(
                    (static_cast<unsigned>(cipher[t]) + 29u - static_cast<unsigned>(ks)) % 29u);
                ++counts[out];
            }
            double chi2 = 0.0;
            for (std::size_t b = 0; b < 29; ++b) {
                const double expected = p[b] * static_cast<double>(T);
                if (expected > 0.0) {
                    const double d = static_cast<double>(counts[b]) - expected;
                    chi2 += d * d / expected;
                }
            }
            scores[c] = chi2;
        }
        return scores;
    }

private:
    TheoryHistChi2S2ResidueTest() = delete;
};

TEST_CASE("TheoryHistChi2S2 running residue matches host (T%4!=0 epilogue)",
          "[cuda][golden][hist][s2][residue]") {
    REQUIRE(ParcaeCuda::available());
    StatusOr<ExpectedFrequencyTable> freqs = TheoryHistChi2S2ResidueTest::load_freqs();
    REQUIRE(freqs.ok());

    // T=51 → n4=12, remainder 3 (exercises epilogue).
    constexpr std::size_t T = 51;
    constexpr std::size_t C = 9;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 5u + 2u) % 29u);
    }
    std::vector<std::uint8_t> host_b0;
    std::vector<std::uint8_t> host_b1;
    for (std::uint8_t b0 = 0; b0 < 3; ++b0) {
        for (std::uint8_t b1 = 0; b1 < 3; ++b1) {
            host_b0.push_back(b0);
            host_b1.push_back(b1);
        }
    }
    REQUIRE(host_b0.size() == C);

    const std::vector<double> expect =
        TheoryHistChi2S2ResidueTest::host_s2_chi2(host_in, host_b0, host_b1, freqs.value());

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b0 =
        DeviceBuffer<std::uint8_t>::from_host(host_b0);
    REQUIRE(device_b0.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b1 =
        DeviceBuffer<std::uint8_t>::from_host(host_b1);
    REQUIRE(device_b1.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2S2::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    REQUIRE(TheoryHistChi2S2::launch_linear_async(
                device_in.value().data(), device_b0.value().data(), device_b1.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T, /*cipher_minus_ks=*/true)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S2 residue sync").ok());

    std::vector<double> got(C, 0.0);
    REQUIRE(device_scores.value().copy_to_host(got).ok());
    for (std::size_t c = 0; c < C; ++c) {
        REQUIRE(got[c] == Approx(expect[c]).epsilon(1e-9).margin(1e-9));
    }
}

TEST_CASE("TheoryHistChi2S2 enc path + empty reject", "[cuda][hist][s2][edge]") {
    REQUIRE(ParcaeCuda::available());
    StatusOr<ExpectedFrequencyTable> freqs = TheoryHistChi2S2ResidueTest::load_freqs();
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 7; // T%4 != 0, short
    constexpr std::size_t C = 2;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 1u) % 29u);
    }
    std::vector<std::uint8_t> host_b0 = {0, 5};
    std::vector<std::uint8_t> host_b1 = {1, 2};

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b0 =
        DeviceBuffer<std::uint8_t>::from_host(host_b0);
    REQUIRE(device_b0.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b1 =
        DeviceBuffer<std::uint8_t>::from_host(host_b1);
    REQUIRE(device_b1.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2S2::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    // Encrypt form (cipher_minus_ks=false) must launch cleanly.
    REQUIRE(TheoryHistChi2S2::launch_linear_async(
                device_in.value().data(), device_b0.value().data(), device_b1.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T, /*cipher_minus_ks=*/false)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S2 enc sync").ok());

    REQUIRE_FALSE(TheoryHistChi2S2::launch_linear_async(
                       device_in.value().data(), device_b0.value().data(), device_b1.value().data(),
                       device_probs.value().data(), device_counts.value().data(),
                       device_scores.value().data(), 0, T, true)
                      .ok());
    REQUIRE_FALSE(TheoryHistChi2S2::launch_linear_async(
                       device_in.value().data(), device_b0.value().data(), device_b1.value().data(),
                       device_probs.value().data(), device_counts.value().data(),
                       device_scores.value().data(), C, 0, true)
                      .ok());
}

TEST_CASE("TheoryHistChi2S2 fair plate + tile-cap sweep (diary)",
          "[cuda][hist][s2][bench][residue]") {
    REQUIRE(ParcaeCuda::available());
    StatusOr<ExpectedFrequencyTable> freqs = TheoryHistChi2S2ResidueTest::load_freqs();
    REQUIRE(freqs.ok());

    constexpr std::size_t C = 841; // full (b0,b1) grid — Spec S2
    constexpr std::size_t T = 1048576;
    constexpr std::size_t reps = 8;

    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 7u) % 29u);
    }
    std::vector<std::uint8_t> host_b0(C);
    std::vector<std::uint8_t> host_b1(C);
    for (std::size_t c = 0; c < C; ++c) {
        host_b0[c] = static_cast<std::uint8_t>(c / 29u);
        host_b1[c] = static_cast<std::uint8_t>(c % 29u);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b0 =
        DeviceBuffer<std::uint8_t>::from_host(host_b0);
    REQUIRE(device_b0.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b1 =
        DeviceBuffer<std::uint8_t>::from_host(host_b1);
    REQUIRE(device_b1.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2S2::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    auto time_at_cap = [&](int cap) -> double {
        HistTileCap::set(HistTileCap::kS2, cap);
        auto launch = [&]() -> Status {
            return TheoryHistChi2S2::launch_linear_async(
                device_in.value().data(), device_b0.value().data(), device_b1.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T, /*cipher_minus_ks=*/true);
        };
        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, launch);
        REQUIRE(sample.ok());
        return sample.value().runes_per_sec();
    };

    // Production (cap=0 → HistTileCap S2 default 32), then A/B 64/128.
    const double rps_prod = time_at_cap(0);
    const double rps_32 = time_at_cap(32);
    const double rps_64 = time_at_cap(64);
    const double rps_128 = time_at_cap(128);
    HistTileCap::set(HistTileCap::kS2, 0); // restore

    const double peak = BenchTierSpec::theory_s2_linear.estimated_peak;
    std::printf("S2_RESIDUE_PLATE C=%zu T=%zu reps=%zu peak=%.3e\n", C, T, reps, peak);
    std::printf("  cap0/prod=%.6e (%.2f%%)  cap32=%.6e  cap64=%.6e  cap128=%.6e\n", rps_prod,
                100.0 * rps_prod / peak, rps_32, rps_64, rps_128);
    std::fflush(stdout);

    REQUIRE(HistTileCap::slot_default(HistTileCap::kS2) == 32);
    REQUIRE(rps_prod > 100.0e9);
    REQUIRE(BenchTierSpec::percent_peak(rps_prod, peak) <= 100.0);
    // Absolute floor: prior ks29 plate ~400B; residue+tile32 must stay above that class.
    REQUIRE(rps_prod > 400.0e9);
}

#else

TEST_CASE("TheoryHistChi2S2 residue skipped without CUDA", "[cuda][hist][s2][residue]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
