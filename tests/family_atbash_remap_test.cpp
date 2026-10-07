#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "alphabet_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("FamilyChi2Batch Atbash remap matches decode-hist golden",
          "[cuda][batch][chi2][remap][atbash][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    std::vector<Index29> plain;
    plain.reserve(256);
    for (std::size_t i = 0; i < 256; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 3u) % 29u)});
    }
    StatusOr<std::vector<Index29>> cipher =
        AtbashTransform{}.apply(plain, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(cipher.ok());

    constexpr std::size_t C = 8; // occupancy-style pad; all lanes identical
    const std::size_t T = cipher.value().size();
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = cipher.value()[i].value();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());

    StatusOr<DeviceBuffer<std::uint32_t>> decode_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * FamilyChi2Batch::alphabet_size);
    REQUIRE(decode_counts.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> prod_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * FamilyChi2Batch::alphabet_size);
    REQUIRE(prod_counts.ok());
    StatusOr<DeviceBuffer<double>> prod_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(prod_scores.ok());

    REQUIRE(FamilyChi2Batch::launch_atbash_decode_hist_async(
                device_in.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(FamilyChi2Batch::launch_atbash_async(
                device_in.value().data(), device_probs.value().data(), prod_counts.value().data(),
                prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "atbash wire sync").ok());

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
    const HistAlphabetMap::Hist mirrored = HistAlphabetMap::mirror_atbash(H.value());
    for (std::size_t b = 0; b < 29; ++b) {
        REQUIRE(prod_hist[b] == mirrored[b]);
    }
}

TEST_CASE("FamilyChi2Batch Atbash∘Caesar remap matches decode-hist golden",
          "[cuda][batch][chi2][remap][atbash_caesar][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    std::vector<Index29> plain;
    plain.reserve(180);
    for (std::size_t i = 0; i < 180; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>((i * 2u + 9u) % 29u)});
    }
    StatusOr<std::vector<Index29>> after_atbash =
        AtbashTransform{}.apply(plain, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(after_atbash.ok());
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        after_atbash.value(), nlohmann::json{{"shift", 7}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    // Scoring path applies enc_caesar(dec_atbash(cipher), shift) for each candidate shift.
    constexpr std::size_t C = 29;
    const std::size_t T = cipher.value().size();
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = cipher.value()[i].value();
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

    REQUIRE(FamilyChi2Batch::launch_atbash_caesar_decode_hist_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), decode_counts.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(FamilyChi2Batch::launch_atbash_caesar_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), prod_counts.value().data(),
                prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "atbash_caesar wire sync").ok());

    std::vector<double> decode_host(C);
    std::vector<double> prod_host(C);
    REQUIRE(decode_scores.value().copy_to_host(decode_host).ok());
    REQUIRE(prod_scores.value().copy_to_host(prod_host).ok());
    REQUIRE(prod_host == decode_host);

    std::vector<std::uint32_t> prod_hist(C * 29);
    REQUIRE(prod_counts.value().copy_to_host(prod_hist).ok());

    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(host_in);
    REQUIRE(H.ok());
    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        StatusOr<HistAlphabetMap::Hist> P =
            HistAlphabetMap::compose_atbash_caesar_encrypt(H.value(), shift);
        REQUIRE(P.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(prod_hist[static_cast<std::size_t>(shift) * 29u + b] == P.value()[b]);
        }
    }
}

TEST_CASE("AlphabetChi2Batch Atbash remap host math parity",
          "[cuda][batch][chi2][remap][atbash]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 512;
    constexpr std::size_t C = 1;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 3u + 1u) % 29u);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
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

    REQUIRE(AlphabetChi2Batch::launch_atbash(device_in.value().data(), device_probs.value().data(),
                                             device_hist.value().data(),
                                             device_counts.value().data(),
                                             device_scores.value().data(), C, T)
                .ok());

    std::vector<std::uint32_t> counts_host(29);
    REQUIRE(device_counts.value().copy_to_host(counts_host).ok());
    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(host_in);
    REQUIRE(H.ok());
    const HistAlphabetMap::Hist mirrored = HistAlphabetMap::mirror_atbash(H.value());
    for (std::size_t b = 0; b < 29; ++b) {
        REQUIRE(counts_host[b] == mirrored[b]);
    }
}

TEST_CASE("FamilyChi2Batch Atbash remap rejects bad args",
          "[cuda][batch][chi2][remap][atbash][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{0, 1, 2, 3});
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::allocate(29);
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts = DeviceBuffer<std::uint32_t>::allocate(29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(1);
    REQUIRE(device_scores.ok());

    REQUIRE_FALSE(FamilyChi2Batch::launch_atbash_async(device_in.value().data(),
                                                       device_probs.value().data(),
                                                       device_counts.value().data(),
                                                       device_scores.value().data(), 0, 4)
                      .ok());
    REQUIRE_FALSE(FamilyChi2Batch::launch_atbash_async(nullptr, device_probs.value().data(),
                                                       device_counts.value().data(),
                                                       device_scores.value().data(), 1, 4)
                      .ok());
}

#endif
