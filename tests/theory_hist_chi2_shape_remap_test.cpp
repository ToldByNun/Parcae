#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_shape.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("TheoryHistChi2Shape Atbash remap matches decode-hist golden",
          "[cuda][theory][shape][remap][atbash][golden]") {
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

    constexpr std::size_t C = 8;
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
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Shape::alphabet_size);
    REQUIRE(decode_counts.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> prod_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Shape::alphabet_size);
    REQUIRE(prod_counts.ok());
    StatusOr<DeviceBuffer<double>> prod_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(prod_scores.ok());

    REQUIRE(TheoryHistChi2Shape::launch_atbash_decode_hist_async(
                device_in.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(TheoryHistChi2Shape::launch_atbash_async(
                device_in.value().data(), device_probs.value().data(), prod_counts.value().data(),
                prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "shape atbash wire sync").ok());

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

TEST_CASE("TheoryHistChi2Shape Caesar remap matches decode-hist golden",
          "[cuda][theory][shape][remap][caesar][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    std::vector<Index29> plain;
    plain.reserve(220);
    for (std::size_t i = 0; i < 220; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>((i * 3u + 11u) % 29u)});
    }
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain, nlohmann::json{{"shift", 12}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

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

    REQUIRE(TheoryHistChi2Shape::launch_caesar_decode_hist_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), decode_counts.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(TheoryHistChi2Shape::launch_caesar_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), prod_counts.value().data(),
                prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "shape caesar wire sync").ok());

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
        StatusOr<HistAlphabetMap::Hist> P =
            HistAlphabetMap::rotate_decrypt(H.value(), shifts[c]);
        REQUIRE(P.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(prod_hist[c * 29u + b] == P.value()[b]);
        }
    }
}

TEST_CASE("TheoryHistChi2Shape Affine remap 812 matches decode-hist golden",
          "[cuda][theory][shape][remap][affine][golden]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(AffineCandidateGenerator::candidate_count == 812u);

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    std::vector<Index29> plain;
    plain.reserve(300);
    for (std::size_t i = 0; i < 300; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>((i * 7u + 4u) % 29u)});
    }
    StatusOr<std::vector<Index29>> cipher = AffineTransform{}.apply(
        plain, nlohmann::json{{"a", 5}, {"b", 11}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    constexpr std::size_t C = AffineCandidateGenerator::candidate_count;
    const std::size_t T = cipher.value().size();
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = cipher.value()[i].value();
    }

    std::vector<std::uint8_t> host_a(C);
    std::vector<std::uint8_t> host_b(C);
    std::size_t lane = 0;
    for (std::uint8_t a = 1; a < 29; ++a) {
        for (std::uint8_t b = 0; b < 29; ++b) {
            host_a[lane] = a;
            host_b[lane] = b;
            ++lane;
        }
    }
    REQUIRE(lane == C);

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_a = DeviceBuffer<std::uint8_t>::from_host(host_a);
    REQUIRE(device_a.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b = DeviceBuffer<std::uint8_t>::from_host(host_b);
    REQUIRE(device_b.ok());
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

    REQUIRE(TheoryHistChi2Shape::launch_affine_decode_hist_async(
                device_in.value().data(), device_a.value().data(), device_b.value().data(),
                device_probs.value().data(), decode_counts.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(TheoryHistChi2Shape::launch_affine_async(
                device_in.value().data(), device_a.value().data(), device_b.value().data(),
                device_probs.value().data(), prod_counts.value().data(),
                prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "shape affine wire sync").ok());

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

    const std::size_t correct_lane =
        static_cast<std::size_t>(5 - 1) * 29u + static_cast<std::size_t>(11);
    std::vector<std::uint8_t> plain_bytes(plain.size());
    for (std::size_t i = 0; i < plain.size(); ++i) {
        plain_bytes[i] = plain[i].value();
    }
    StatusOr<HistAlphabetMap::Hist> plain_hist = HistAlphabetMap::count_stream_hist(plain_bytes);
    REQUIRE(plain_hist.ok());
    for (std::size_t b = 0; b < 29; ++b) {
        REQUIRE(prod_hist[correct_lane * 29u + b] == plain_hist.value()[b]);
    }
}

#else

TEST_CASE("TheoryHistChi2Shape remap skipped without CUDA", "[cuda][theory][shape][remap]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
