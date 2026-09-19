#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "caesar_chi2_batch.hpp"
#include "device_buffer.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <span>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("CaesarChi2Batch matches CPU chi2 per shift", "[cuda][batch][chi2][fuse]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    std::vector<Index29> plain;
    plain.reserve(128);
    for (std::uint8_t i = 0; i < 128; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>(i % 29)});
    }
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain, nlohmann::json{{"shift", 7}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    constexpr std::size_t C = Index29::modulus;
    const std::size_t T = cipher.value().size();
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = cipher.value()[i].value();
    }
    std::vector<std::uint8_t> shifts(C);
    std::vector<std::uint8_t> dirs(C, static_cast<std::uint8_t>(CudaDir::Decrypt));
    for (std::size_t c = 0; c < C; ++c) {
        shifts[c] = static_cast<std::uint8_t>(c);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_dirs = DeviceBuffer<std::uint8_t>::from_host(dirs);
    REQUIRE(device_dirs.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    REQUIRE(CaesarChi2Batch::launch(
                device_in.value().data(),
                device_shifts.value().data(),
                device_dirs.value().data(),
                device_probs.value().data(),
                device_counts.value().data(),
                device_scores.value().data(),
                C,
                T)
                .ok());

    std::vector<double> gpu(C, 0.0);
    REQUIRE(device_scores.value().copy_to_host(gpu).ok());

    ScoreRequest request;
    request.expected_frequencies = &freqs.value();
    for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
        StatusOr<std::vector<Index29>> out = CaesarTransform{}.apply(
            cipher.value(),
            nlohmann::json{{"shift", static_cast<int>(shift)}},
            TransformDirection::Decrypt);
        REQUIRE(out.ok());
        StatusOr<double> cpu = ScoreRegistry::score(
            "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(cpu.ok());
        REQUIRE(gpu[shift] == cpu.value());
    }
}

#endif
