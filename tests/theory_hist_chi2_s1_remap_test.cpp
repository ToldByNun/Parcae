#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_s1.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("TheoryHistChi2S1 LUT remap matches decode-hist golden",
          "[cuda][theory][s1][remap][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 280;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 5u + 2u) % 29u);
    }

    // Three LUTs: identity, atbash, and Caesar-decrypt shift=7 (plain = cipher - 7).
    constexpr std::size_t C = 3;
    std::vector<std::uint8_t> host_luts(C * 29);
    for (std::uint8_t x = 0; x < 29; ++x) {
        host_luts[x] = x;
        host_luts[29u + x] = static_cast<std::uint8_t>(28u - x);
        const unsigned y = static_cast<unsigned>(x) + 29u - 7u;
        host_luts[58u + x] = static_cast<std::uint8_t>(y >= 29u ? y - 29u : y);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_luts =
        DeviceBuffer<std::uint8_t>::from_host(host_luts);
    REQUIRE(device_luts.ok());
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

    REQUIRE(TheoryHistChi2S1::launch_lut_decode_hist_async(
                device_in.value().data(), device_luts.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(TheoryHistChi2S1::launch_lut_async(
                device_in.value().data(), device_luts.value().data(), device_probs.value().data(),
                prod_counts.value().data(), prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s1 lut wire sync").ok());

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

    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(host_in);
    REQUIRE(H.ok());
    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<HistAlphabetMap::Hist> P = HistAlphabetMap::apply_bin_map(
            H.value(), std::span<const std::uint8_t>(host_luts.data() + c * 29u, 29));
        REQUIRE(P.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(prod_hist[c * 29u + b] == P.value()[b]);
        }
    }
}

#else

TEST_CASE("TheoryHistChi2S1 remap skipped without CUDA", "[cuda][theory][s1][remap]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
