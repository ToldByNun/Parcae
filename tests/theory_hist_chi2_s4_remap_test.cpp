#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_s4.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("TheoryHistChi2S4 remap matches decode-hist (mixed L, minus)",
          "[cuda][theory][s4][remap][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 300;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 4u + 9u) % 29u);
    }

    // Mixed lags including 0 (identity) and L>T edge via large lag on short? keep L < T here.
    const std::vector<std::uint8_t> lags = {0, 3, 5, 3, 7};
    constexpr std::size_t C = 5;

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_lags = DeviceBuffer<std::uint8_t>::from_host(lags);
    REQUIRE(device_lags.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());

    StatusOr<DeviceBuffer<std::uint32_t>> decode_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(decode_counts.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> prod_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(prod_counts.ok());
    StatusOr<DeviceBuffer<double>> prod_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(prod_scores.ok());

    REQUIRE(TheoryHistChi2S4::launch_autokey_decode_hist_async(
                device_in.value().data(), device_lags.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T,
                /*cipher_minus_ks=*/true)
                .ok());
    REQUIRE(TheoryHistChi2S4::launch_autokey_async(
                device_in.value().data(), device_lags.value().data(), device_probs.value().data(),
                prod_counts.value().data(), prod_scores.value().data(), C, T,
                /*cipher_minus_ks=*/true)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s4 minus wire sync").ok());

    std::vector<double> decode_host(C);
    std::vector<double> prod_host(C);
    REQUIRE(decode_scores.value().copy_to_host(decode_host).ok());
    REQUIRE(prod_scores.value().copy_to_host(prod_host).ok());
    REQUIRE(prod_host == decode_host);

    std::vector<std::uint32_t> decode_hist(C * 29);
    std::vector<std::uint32_t> prod_hist(C * 29);
    REQUIRE(decode_counts.value().copy_to_host(decode_hist).ok());
    REQUIRE(prod_counts.value().copy_to_host(prod_hist).ok());
    REQUIRE(prod_hist == decode_hist);

    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<HistAlphabetMap::Hist> host =
            HistAlphabetMap::ring_plain_hist_from_once(host_in, lags[c], /*cipher_minus_ks=*/true);
        REQUIRE(host.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(prod_hist[c * 29u + b] == host.value()[b]);
        }
    }
}

TEST_CASE("TheoryHistChi2S4 remap matches decode-hist (add form + L>=T)",
          "[cuda][theory][s4][remap][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 8;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 7u + 2u) % 29u);
    }

    // L=2 (normal), L=8 (==T), L=20 (>T), L=0.
    const std::vector<std::uint8_t> lags = {2, 8, 20, 0};
    constexpr std::size_t C = 4;

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_lags = DeviceBuffer<std::uint8_t>::from_host(lags);
    REQUIRE(device_lags.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());

    StatusOr<DeviceBuffer<std::uint32_t>> decode_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(decode_counts.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> prod_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(prod_counts.ok());
    StatusOr<DeviceBuffer<double>> prod_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(prod_scores.ok());

    REQUIRE(TheoryHistChi2S4::launch_autokey_decode_hist_async(
                device_in.value().data(), device_lags.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T,
                /*cipher_minus_ks=*/false)
                .ok());
    REQUIRE(TheoryHistChi2S4::launch_autokey_async(
                device_in.value().data(), device_lags.value().data(), device_probs.value().data(),
                prod_counts.value().data(), prod_scores.value().data(), C, T,
                /*cipher_minus_ks=*/false)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s4 add wire sync").ok());

    std::vector<std::uint32_t> decode_hist(C * 29);
    std::vector<std::uint32_t> prod_hist(C * 29);
    REQUIRE(decode_counts.value().copy_to_host(decode_hist).ok());
    REQUIRE(prod_counts.value().copy_to_host(prod_hist).ok());
    REQUIRE(prod_hist == decode_hist);

    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<HistAlphabetMap::Hist> host =
            HistAlphabetMap::ring_plain_hist_from_once(host_in, lags[c], /*cipher_minus_ks=*/false);
        REQUIRE(host.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(prod_hist[c * 29u + b] == host.value()[b]);
        }
    }
}

#else

TEST_CASE("TheoryHistChi2S4 remap skipped without CUDA", "[cuda][theory][s4][remap]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
