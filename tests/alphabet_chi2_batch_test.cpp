#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "alphabet_chi2_batch.hpp"
#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("AlphabetChi2Batch Caesar remap scores match decode-hist golden",
          "[cuda][batch][chi2][remap][caesar][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    std::vector<Index29> plain;
    plain.reserve(512);
    for (std::size_t i = 0; i < 512; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>((i * 3u + 5u) % 29u)});
    }
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain, nlohmann::json{{"shift", 11}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    constexpr std::size_t C = Index29::modulus;
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
        DeviceBuffer<std::uint32_t>::allocate(C * AlphabetChi2Batch::alphabet_size);
    REQUIRE(decode_counts.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());

    StatusOr<DeviceBuffer<std::uint32_t>> remap_hist =
        DeviceBuffer<std::uint32_t>::allocate(AlphabetChi2Batch::alphabet_size);
    REQUIRE(remap_hist.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> remap_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * AlphabetChi2Batch::alphabet_size);
    REQUIRE(remap_counts.ok());
    StatusOr<DeviceBuffer<double>> remap_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(remap_scores.ok());

    REQUIRE(CaesarChi2Batch::launch_decode_hist_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), decode_counts.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "decode golden sync").ok());

    REQUIRE(AlphabetChi2Batch::launch_caesar_decrypt(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), remap_hist.value().data(),
                remap_counts.value().data(), remap_scores.value().data(), C, T)
                .ok());

    std::vector<double> decode_host(C, 0.0);
    std::vector<double> remap_host(C, 0.0);
    REQUIRE(decode_scores.value().copy_to_host(decode_host).ok());
    REQUIRE(remap_scores.value().copy_to_host(remap_host).ok());
    REQUIRE(remap_host == decode_host);

    std::vector<std::uint32_t> decode_hist_host(C * AlphabetChi2Batch::alphabet_size, 0u);
    std::vector<std::uint32_t> remap_hist_host(C * AlphabetChi2Batch::alphabet_size, 0u);
    REQUIRE(decode_counts.value().copy_to_host(decode_hist_host).ok());
    REQUIRE(remap_counts.value().copy_to_host(remap_hist_host).ok());
    REQUIRE(remap_hist_host == decode_hist_host);

    // CPU χ² oracle on true plaintext for the correct shift.
    ScoreRequest request;
    request.expected_frequencies = &freqs.value();
    StatusOr<double> cpu_plain =
        ScoreRegistry::score("chi2_english_gp_v0", plain, "v0", nlohmann::json::object(), request);
    REQUIRE(cpu_plain.ok());
    REQUIRE(remap_host[11] == cpu_plain.value());
}

TEST_CASE("AlphabetChi2Batch Caesar remap matches HistAlphabetMap host math",
          "[cuda][batch][chi2][remap][caesar]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 1024;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 7u + 3u) % 29u);
    }
    constexpr std::size_t C = 29;
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
    StatusOr<DeviceBuffer<std::uint32_t>> device_cipher_hist =
        DeviceBuffer<std::uint32_t>::allocate(AlphabetChi2Batch::alphabet_size);
    REQUIRE(device_cipher_hist.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * AlphabetChi2Batch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    REQUIRE(AlphabetChi2Batch::launch_caesar_decrypt(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_cipher_hist.value().data(),
                device_counts.value().data(), device_scores.value().data(), C, T)
                .ok());

    std::vector<std::uint32_t> counts_host(C * AlphabetChi2Batch::alphabet_size, 0u);
    REQUIRE(device_counts.value().copy_to_host(counts_host).ok());

    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(host_in);
    REQUIRE(H.ok());
    REQUIRE(HistAlphabetMap::hist_total(H.value()) == T);

    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        StatusOr<HistAlphabetMap::Hist> P = HistAlphabetMap::rotate_decrypt(H.value(), shift);
        REQUIRE(P.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(counts_host[static_cast<std::size_t>(shift) * 29u + b] == P.value()[b]);
        }
        StatusOr<double> chi2 =
            HistAlphabetMap::chi2_from_hist(P.value(), freqs.value().probabilities());
        REQUIRE(chi2.ok());
        std::vector<double> scores_host(C, 0.0);
        REQUIRE(device_scores.value().copy_to_host(scores_host).ok());
        REQUIRE(scores_host[shift] == chi2.value());
    }
}

TEST_CASE("AlphabetChi2Batch rejects bad args", "[cuda][batch][chi2][remap][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{0, 1, 2, 3});
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{0});
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<double>> device_probs =
        DeviceBuffer<double>::allocate(AlphabetChi2Batch::alphabet_size);
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_hist =
        DeviceBuffer<std::uint32_t>::allocate(AlphabetChi2Batch::alphabet_size);
    REQUIRE(device_hist.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(AlphabetChi2Batch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(1);
    REQUIRE(device_scores.ok());

    REQUIRE_FALSE(AlphabetChi2Batch::launch_caesar_decrypt_async(
                      device_in.value().data(), device_shifts.value().data(),
                      device_probs.value().data(), device_hist.value().data(),
                      device_counts.value().data(), device_scores.value().data(), 0, 4)
                      .ok());
    REQUIRE_FALSE(AlphabetChi2Batch::launch_caesar_decrypt_async(
                      device_in.value().data(), device_shifts.value().data(),
                      device_probs.value().data(), device_hist.value().data(),
                      device_counts.value().data(), device_scores.value().data(), 1, 0)
                      .ok());
    REQUIRE_FALSE(AlphabetChi2Batch::launch_caesar_decrypt_async(
                      nullptr, device_shifts.value().data(), device_probs.value().data(),
                      device_hist.value().data(), device_counts.value().data(),
                      device_scores.value().data(), 1, 4)
                      .ok());
}

TEST_CASE("AlphabetChi2Batch Caesar remap fair T matches decode-hist",
          "[cuda][batch][chi2][remap][caesar][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 1048576;
    constexpr std::size_t C = 29;
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

    StatusOr<DeviceBuffer<std::uint32_t>> decode_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(decode_counts.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> remap_hist = DeviceBuffer<std::uint32_t>::allocate(29);
    REQUIRE(remap_hist.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> remap_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(remap_counts.ok());
    StatusOr<DeviceBuffer<double>> remap_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(remap_scores.ok());

    REQUIRE(CaesarChi2Batch::launch_decode_hist_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), decode_counts.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "fair decode sync").ok());

    REQUIRE(AlphabetChi2Batch::launch_caesar_decrypt(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), remap_hist.value().data(),
                remap_counts.value().data(), remap_scores.value().data(), C, T)
                .ok());

    std::vector<double> decode_host(C);
    std::vector<double> remap_host(C);
    REQUIRE(decode_scores.value().copy_to_host(decode_host).ok());
    REQUIRE(remap_scores.value().copy_to_host(remap_host).ok());
    REQUIRE(remap_host == decode_host);

    std::vector<std::uint32_t> base(29, 0u);
    REQUIRE(remap_hist.value().copy_to_host(base).ok());
    REQUIRE(std::accumulate(base.begin(), base.end(), 0u) == T);
}

TEST_CASE("CaesarChi2Batch launch_decrypt_async production remap matches decode-hist legacy",
          "[cuda][batch][chi2][remap][caesar][wire]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t C = 29;
    constexpr std::size_t T = 4096;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 11u + 2u) % 29u);
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

    StatusOr<DeviceBuffer<std::uint32_t>> legacy_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(legacy_counts.ok());
    StatusOr<DeviceBuffer<double>> legacy_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(legacy_scores.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> prod_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(prod_counts.ok());
    StatusOr<DeviceBuffer<double>> prod_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(prod_scores.ok());

    REQUIRE(CaesarChi2Batch::launch_decode_hist_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), legacy_counts.value().data(),
                legacy_scores.value().data(), C, T)
                .ok());
    REQUIRE(CaesarChi2Batch::launch_decrypt_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), prod_counts.value().data(),
                prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "wire sync").ok());

    std::vector<double> legacy_host(C);
    std::vector<double> prod_host(C);
    REQUIRE(legacy_scores.value().copy_to_host(legacy_host).ok());
    REQUIRE(prod_scores.value().copy_to_host(prod_host).ok());
    REQUIRE(prod_host == legacy_host);

    std::vector<std::uint32_t> legacy_hist(C * 29);
    std::vector<std::uint32_t> prod_hist(C * 29);
    REQUIRE(legacy_counts.value().copy_to_host(legacy_hist).ok());
    REQUIRE(prod_counts.value().copy_to_host(prod_hist).ok());
    REQUIRE(prod_hist == legacy_hist);
}

#endif
