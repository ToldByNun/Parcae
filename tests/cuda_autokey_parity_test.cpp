#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/transform/ciphertext_autokey_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "ciphertext_autokey_kernel.hpp"
#include "cuda_error.hpp"
#include "deep_score_batch.hpp"
#include "device_buffer.hpp"
#include "interrupt_device_view.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

[[nodiscard]] std::vector<Index29> from_bytes(const std::vector<std::uint8_t>& bytes) {
    std::vector<Index29> out;
    out.reserve(bytes.size());
    for (std::uint8_t value : bytes) {
        out.push_back(Index29{value});
    }
    return out;
}

[[nodiscard]] std::vector<Index29> key_from_bytes(const std::vector<std::uint8_t>& bytes) {
    return from_bytes(bytes);
}

} // namespace

TEST_CASE("CUDA CTAK dense decrypt matches CPU hand vector and DeepScoreBatch formula",
          "[cuda][parity][autokey][ctak]") {
    REQUIRE(ParcaeCuda::available());

    // Same hand vector as CiphertextAutokeyTransform dense roundtrip:
    // primer [3,5], plain [1,2,4,7,8] → cipher [4,7,8,14,16]
    const std::vector<Index29> plain{Index29{1}, Index29{2}, Index29{4}, Index29{7}, Index29{8}};
    const std::vector<std::uint8_t> key{3, 5};
    const nlohmann::json params{{"key_indices", {3, 5}}};

    StatusOr<std::vector<Index29>> cipher =
        CiphertextAutokeyTransform{}.apply(plain, params, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value() == std::vector<Index29>{Index29{4}, Index29{7}, Index29{8}, Index29{14},
                                                   Index29{16}});

    std::vector<Index29> cpu_plain(cipher.value().size());
    REQUIRE(CiphertextAutokeyTransform::kernel(cipher.value(), cpu_plain, key_from_bytes(key), {},
                                               TransformDirection::Decrypt)
                .ok());
    REQUIRE(cpu_plain == plain);

    const std::vector<std::uint8_t> host_cipher = to_bytes(cipher.value());
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), host_cipher.size());
    REQUIRE(view.ok());

    std::vector<std::uint8_t> host_out(host_cipher.size(), 0xFFu);
    REQUIRE(CiphertextAutokeyKernel::apply_host(host_cipher, host_out, key, view.value(),
                                                CudaDir::Decrypt)
                .ok());
    REQUIRE(from_bytes(host_out) == plain);
}

TEST_CASE("CUDA CTAK encrypt/decrypt roundtrip with interrupts matches CPU",
          "[cuda][parity][autokey][ctak]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> key{1, 2};
    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
    StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1});
    REQUIRE(interrupt.ok());

    std::vector<Index29> cpu_cipher(plain.size());
    REQUIRE(CiphertextAutokeyTransform::kernel(plain, cpu_cipher, key_from_bytes(key),
                                               interrupt.value().skip_indices(),
                                               TransformDirection::Encrypt)
                .ok());
    REQUIRE(cpu_cipher ==
            std::vector<Index29>{Index29{1}, Index29{1}, Index29{4}, Index29{4}});

    const std::vector<std::uint8_t> host_in = to_bytes(plain);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(interrupt.value(), host_in.size());
    REQUIRE(view.ok());

    std::vector<std::uint8_t> host_cipher(host_in.size(), 0xFFu);
    REQUIRE(CiphertextAutokeyKernel::apply_host(host_in, host_cipher, key, view.value(),
                                                CudaDir::Encrypt)
                .ok());
    REQUIRE(from_bytes(host_cipher) == cpu_cipher);

    std::vector<std::uint8_t> host_recovered(host_cipher.size());
    REQUIRE(CiphertextAutokeyKernel::apply_host(host_cipher, host_recovered, key, view.value(),
                                                CudaDir::Decrypt)
                .ok());
    REQUIRE(host_recovered == host_in);
}

TEST_CASE("CUDA CTAK random dense parity and DeepScoreBatch chi2 match CPU",
          "[cuda][parity][autokey][ctak]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xC7A011u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> plain;
    plain.reserve(128);
    for (std::size_t i = 0; i < 128; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }
    const std::vector<std::uint8_t> key{static_cast<std::uint8_t>(dist(rng)),
                                        static_cast<std::uint8_t>(dist(rng)),
                                        static_cast<std::uint8_t>(dist(rng))};

    std::vector<Index29> cpu_cipher(plain.size());
    REQUIRE(CiphertextAutokeyTransform::kernel(plain, cpu_cipher, key_from_bytes(key), {},
                                               TransformDirection::Encrypt)
                .ok());

    const std::vector<std::uint8_t> host_plain = to_bytes(plain);
    StatusOr<InterruptDeviceView> none =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), host_plain.size());
    REQUIRE(none.ok());

    std::vector<std::uint8_t> gpu_cipher(host_plain.size());
    REQUIRE(CiphertextAutokeyKernel::apply_host(host_plain, gpu_cipher, key, none.value(),
                                                CudaDir::Encrypt)
                .ok());
    REQUIRE(from_bytes(gpu_cipher) == cpu_cipher);

    std::vector<Index29> cpu_plain(cpu_cipher.size());
    REQUIRE(CiphertextAutokeyTransform::kernel(cpu_cipher, cpu_plain, key_from_bytes(key), {},
                                               TransformDirection::Decrypt)
                .ok());
    std::vector<std::uint8_t> gpu_plain(gpu_cipher.size());
    REQUIRE(CiphertextAutokeyKernel::apply_host(gpu_cipher, gpu_plain, key, none.value(),
                                                CudaDir::Decrypt)
                .ok());
    REQUIRE(from_bytes(gpu_plain) == cpu_plain);
    REQUIRE(gpu_plain == host_plain);

    // DeepScoreBatch autokey chi2 must match CPU chi2 on CTAK-decrypted plaintext.
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<std::uint8_t> host_cipher = to_bytes(cpu_cipher);
    const std::size_t T = host_cipher.size();
    std::vector<std::uint32_t> key_begin{0};
    std::vector<std::uint32_t> key_len{static_cast<std::uint32_t>(key.size())};

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(host_cipher);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_key = DeviceBuffer<std::uint8_t>::from_host(key);
    REQUIRE(device_key.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
        DeviceBuffer<std::uint32_t>::from_host(key_begin);
    REQUIRE(device_begin.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_len =
        DeviceBuffer<std::uint32_t>::from_host(key_len);
    REQUIRE(device_len.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts = DeviceBuffer<std::uint32_t>::allocate(29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(1);
    REQUIRE(device_scores.ok());

    REQUIRE(DeepScoreBatch::launch_autokey_chi2_async(
                device_in.value().data(), device_key.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                device_counts.value().data(), device_scores.value().data(), 1, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "autokey chi2 sync").ok());

    std::vector<double> gpu_scores(1, 0.0);
    REQUIRE(device_scores.value().copy_to_host(gpu_scores).ok());

    ScoreRequest request;
    request.expected_frequencies = &freqs.value();
    StatusOr<double> cpu_score = ScoreRegistry::score("chi2_english_gp_v0", cpu_plain, "v0",
                                                      nlohmann::json::object(), request);
    REQUIRE(cpu_score.ok());
    REQUIRE(gpu_scores[0] == cpu_score.value());
}

TEST_CASE("CUDA CTAK rejects empty key and size mismatch", "[cuda][parity][autokey][ctak]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> out(2);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), 2);
    REQUIRE(view.ok());

    REQUIRE_FALSE(
        CiphertextAutokeyKernel::apply_host(in, out, {}, view.value(), CudaDir::Encrypt).ok());

    std::vector<std::uint8_t> short_out(1);
    const std::vector<std::uint8_t> key{1};
    REQUIRE_FALSE(
        CiphertextAutokeyKernel::apply_host(in, short_out, key, view.value(), CudaDir::Encrypt)
            .ok());
}

#else

TEST_CASE("CUDA CTAK skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][autokey][ctak]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CiphertextAutokeyKernel");
}

#endif
