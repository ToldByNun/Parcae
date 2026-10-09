#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"

#include "alphabet_chi2_batch.hpp"
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

namespace {

std::vector<std::uint8_t> make_cipher(std::size_t T) {
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 5u + 2u) % 29u);
    }
    return host_in;
}

/// Identity / Atbash / Caesar-decrypt(shift=7) mono-LUTs, optionally tiled for occupancy.
std::vector<std::uint8_t> make_mono_luts(std::size_t candidate_count) {
    std::vector<std::uint8_t> host_luts(candidate_count * 29u);
    for (std::size_t c = 0; c < candidate_count; ++c) {
        const std::size_t kind = c % 3u;
        for (std::uint8_t x = 0; x < 29; ++x) {
            std::uint8_t y = x;
            if (kind == 1u) {
                y = static_cast<std::uint8_t>(28u - x);
            } else if (kind == 2u) {
                const unsigned d = static_cast<unsigned>(x) + 29u - 7u;
                y = static_cast<std::uint8_t>(d >= 29u ? d - 29u : d);
            }
            host_luts[c * 29u + x] = y;
        }
    }
    return host_luts;
}

} // namespace

TEST_CASE("TheoryHistChi2S1 mono-LUT remap matches decode-hist golden",
          "[cuda][theory][s1][remap][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 280;
    constexpr std::size_t C = 3;
    const std::vector<std::uint8_t> host_in = make_cipher(T);
    const std::vector<std::uint8_t> host_luts = make_mono_luts(C);

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
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s1 mono-LUT wire sync").ok());

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

TEST_CASE("TheoryHistChi2S1 mono-LUT remap occupancy pad matches decode-hist",
          "[cuda][theory][s1][remap][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 400;
    constexpr std::size_t C = 64;
    const std::vector<std::uint8_t> host_in = make_cipher(T);
    const std::vector<std::uint8_t> host_luts = make_mono_luts(C);

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
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s1 occupancy sync").ok());

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
}

TEST_CASE("TheoryHistChi2S1 mono-LUT remap non-bijective matches decode-hist",
          "[cuda][theory][s1][remap][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 120;
    constexpr std::size_t C = 2;
    const std::vector<std::uint8_t> host_in = make_cipher(T);
    // Collapse all cipher bins onto plain 0 / plain 3 — stream hist does the same.
    std::vector<std::uint8_t> host_luts(C * 29, 0);
    for (std::uint8_t x = 0; x < 29; ++x) {
        host_luts[29u + x] = 3;
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
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "s1 collapse sync").ok());

    std::vector<std::uint32_t> decode_hist(C * 29);
    std::vector<std::uint32_t> prod_hist(C * 29);
    REQUIRE(decode_counts.value().copy_to_host(decode_hist).ok());
    REQUIRE(prod_counts.value().copy_to_host(prod_hist).ok());
    REQUIRE(prod_hist == decode_hist);

    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(host_in);
    REQUIRE(H.ok());
    REQUIRE(prod_hist[0] == HistAlphabetMap::hist_total(H.value()));
    REQUIRE(prod_hist[29u + 3u] == HistAlphabetMap::hist_total(H.value()));
}

TEST_CASE("AlphabetChi2Batch mono-LUT remap matches HistAlphabetMap host math",
          "[cuda][batch][chi2][remap][lut][s1]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 320;
    constexpr std::size_t C = 3;
    const std::vector<std::uint8_t> host_in = make_cipher(T);
    const std::vector<std::uint8_t> host_luts = make_mono_luts(C);

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_luts =
        DeviceBuffer<std::uint8_t>::from_host(host_luts);
    REQUIRE(device_luts.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_hist =
        DeviceBuffer<std::uint32_t>::allocate(29);
    REQUIRE(device_hist.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    REQUIRE(AlphabetChi2Batch::launch_lut_decrypt(
                device_in.value().data(), device_luts.value().data(), device_probs.value().data(),
                device_hist.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T)
                .ok());

    std::vector<std::uint32_t> counts_host(C * 29);
    REQUIRE(device_counts.value().copy_to_host(counts_host).ok());
    std::vector<double> scores_host(C);
    REQUIRE(device_scores.value().copy_to_host(scores_host).ok());
    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(host_in);
    REQUIRE(H.ok());
    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<HistAlphabetMap::Hist> P = HistAlphabetMap::apply_bin_map(
            H.value(), std::span<const std::uint8_t>(host_luts.data() + c * 29u, 29));
        REQUIRE(P.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(counts_host[c * 29u + b] == P.value()[b]);
        }
        StatusOr<double> chi2 =
            HistAlphabetMap::chi2_from_hist(P.value(), freqs.value().probabilities());
        REQUIRE(chi2.ok());
        REQUIRE(scores_host[c] == chi2.value());
    }
}

#else

TEST_CASE("TheoryHistChi2S1 mono-LUT remap skipped without CUDA",
          "[cuda][theory][s1][remap]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
