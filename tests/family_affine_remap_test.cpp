#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"
#include "parcae/transform/affine_transform.hpp"
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

TEST_CASE("FamilyChi2Batch Affine remap 812 matches decode-hist golden",
          "[cuda][batch][chi2][remap][affine][golden]") {
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

    REQUIRE(FamilyChi2Batch::launch_affine_decode_hist_async(
                device_in.value().data(), device_a.value().data(), device_b.value().data(),
                device_probs.value().data(), decode_counts.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(FamilyChi2Batch::launch_affine_async(
                device_in.value().data(), device_a.value().data(), device_b.value().data(),
                device_probs.value().data(), prod_counts.value().data(),
                prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "affine wire sync").ok());

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

    // Correct (a,b)=(5,11) lane recovers the plaintext hist.
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

TEST_CASE("AlphabetChi2Batch Affine remap matches HistAlphabetMap host math",
          "[cuda][batch][chi2][remap][affine]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 400;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 4u + 8u) % 29u);
    }

    // Spot-check a few (a,b) pairs including edges.
    const std::uint8_t as[] = {1, 7, 28};
    const std::uint8_t bs[] = {0, 13, 28};
    constexpr std::size_t C = 3;
    std::vector<std::uint8_t> host_a(C);
    std::vector<std::uint8_t> host_b(C);
    for (std::size_t i = 0; i < C; ++i) {
        host_a[i] = as[i];
        host_b[i] = bs[i];
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_a = DeviceBuffer<std::uint8_t>::from_host(host_a);
    REQUIRE(device_a.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b = DeviceBuffer<std::uint8_t>::from_host(host_b);
    REQUIRE(device_b.ok());
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

    REQUIRE(AlphabetChi2Batch::launch_affine_decrypt(
                device_in.value().data(), device_a.value().data(), device_b.value().data(),
                device_probs.value().data(), device_hist.value().data(),
                device_counts.value().data(), device_scores.value().data(), C, T)
                .ok());

    std::vector<std::uint32_t> counts_host(C * 29);
    REQUIRE(device_counts.value().copy_to_host(counts_host).ok());
    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(host_in);
    REQUIRE(H.ok());
    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<HistAlphabetMap::Hist> P =
            HistAlphabetMap::permute_affine_decrypt(H.value(), host_a[c], host_b[c]);
        REQUIRE(P.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(counts_host[c * 29u + b] == P.value()[b]);
        }
    }
}

TEST_CASE("FamilyChi2Batch Affine remap rejects bad args",
          "[cuda][batch][chi2][remap][affine][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{0, 1, 2, 3});
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_a =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{1});
    REQUIRE(device_a.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{0});
    REQUIRE(device_b.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::allocate(29);
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts = DeviceBuffer<std::uint32_t>::allocate(29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(1);
    REQUIRE(device_scores.ok());

    REQUIRE_FALSE(FamilyChi2Batch::launch_affine_async(
                      device_in.value().data(), device_a.value().data(), device_b.value().data(),
                      device_probs.value().data(), device_counts.value().data(),
                      device_scores.value().data(), 0, 4)
                      .ok());
    REQUIRE_FALSE(FamilyChi2Batch::launch_affine_async(
                      nullptr, device_a.value().data(), device_b.value().data(),
                      device_probs.value().data(), device_counts.value().data(),
                      device_scores.value().data(), 1, 4)
                      .ok());
}

#endif
