#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"

#include "cuda_error.hpp"
#include "deep_score_batch.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

class CtakRemapTestSupport {
public:
    [[nodiscard]] static std::vector<std::uint8_t> make_plain_bytes(std::size_t n) {
        std::vector<std::uint8_t> plain(n);
        for (std::size_t i = 0; i < n; ++i) {
            plain[i] = static_cast<std::uint8_t>((i * 5u + 3u) % 29u);
        }
        return plain;
    }

    /// Dense CTAK encrypt: key = primer for t < L else cipher[t-L].
    [[nodiscard]] static std::vector<std::uint8_t>
    encrypt_ctak(std::span<const std::uint8_t> plain, std::span<const std::uint8_t> primer) {
        std::vector<std::uint8_t> cipher(plain.size());
        const std::size_t L = primer.size();
        for (std::size_t t = 0; t < plain.size(); ++t) {
            const std::uint8_t key = t < L ? primer[t] : cipher[t - L];
            cipher[t] = Z29::add(Index29{plain[t]}, Index29{key}).value();
        }
        return cipher;
    }

private:
    CtakRemapTestSupport() = delete;
};

TEST_CASE("DeepScoreBatch CTAK remap matches decode-hist (mixed L)",
          "[cuda][batch][chi2][remap][ctak][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<std::uint8_t> plain = CtakRemapTestSupport::make_plain_bytes(240);
    const std::vector<std::uint8_t> primer_a = {3, 7, 11};
    const std::vector<std::uint8_t> cipher =
        CtakRemapTestSupport::encrypt_ctak(plain, primer_a);

    const std::vector<std::uint8_t> primer_b = {1, 2, 3, 4, 5};
    const std::vector<std::uint8_t> primer_c = {28, 0};

    constexpr std::size_t C = 3;
    const std::size_t T = cipher.size();
    std::vector<std::uint8_t> key_bytes;
    key_bytes.insert(key_bytes.end(), primer_a.begin(), primer_a.end());
    key_bytes.insert(key_bytes.end(), primer_b.begin(), primer_b.end());
    key_bytes.insert(key_bytes.end(), primer_c.begin(), primer_c.end());
    const std::vector<std::uint32_t> key_begin = {0u, 3u, 8u};
    const std::vector<std::uint32_t> key_len = {3u, 5u, 2u};

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(cipher);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_keys =
        DeviceBuffer<std::uint8_t>::from_host(key_bytes);
    REQUIRE(device_keys.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
        DeviceBuffer<std::uint32_t>::from_host(key_begin);
    REQUIRE(device_begin.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_len =
        DeviceBuffer<std::uint32_t>::from_host(key_len);
    REQUIRE(device_len.ok());
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

    REQUIRE(DeepScoreBatch::launch_autokey_chi2_decode_hist_async(
                device_in.value().data(), device_keys.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(DeepScoreBatch::launch_autokey_chi2_async(
                device_in.value().data(), device_keys.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                prod_counts.value().data(), prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "ctak wire sync").ok());

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

    StatusOr<HistAlphabetMap::Hist> plain_hist = HistAlphabetMap::count_stream_hist(plain);
    REQUIRE(plain_hist.ok());
    for (std::size_t b = 0; b < 29; ++b) {
        REQUIRE(prod_hist[b] == plain_hist.value()[b]);
    }
}

TEST_CASE("DeepScoreBatch CTAK remap edge L>=T matches decode-hist",
          "[cuda][batch][chi2][remap][ctak][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    // T=4; primers with L==T and L>T (prefix-only, zero lag-diff).
    const std::vector<std::uint8_t> plain = {1, 2, 3, 4};
    const std::vector<std::uint8_t> primer_eq = {5, 6, 7, 8};
    const std::vector<std::uint8_t> cipher =
        CtakRemapTestSupport::encrypt_ctak(plain, primer_eq);
    REQUIRE(cipher.size() == 4u);

    const std::vector<std::uint8_t> primer_gt = {9, 10, 11, 12, 13, 14};
    constexpr std::size_t C = 2;
    const std::size_t T = cipher.size();
    std::vector<std::uint8_t> key_bytes = primer_eq;
    key_bytes.insert(key_bytes.end(), primer_gt.begin(), primer_gt.end());
    const std::vector<std::uint32_t> key_begin = {0u, 4u};
    const std::vector<std::uint32_t> key_len = {4u, 6u};

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(cipher);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_keys =
        DeviceBuffer<std::uint8_t>::from_host(key_bytes);
    REQUIRE(device_keys.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
        DeviceBuffer<std::uint32_t>::from_host(key_begin);
    REQUIRE(device_begin.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_len =
        DeviceBuffer<std::uint32_t>::from_host(key_len);
    REQUIRE(device_len.ok());
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

    REQUIRE(DeepScoreBatch::launch_autokey_chi2_decode_hist_async(
                device_in.value().data(), device_keys.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(DeepScoreBatch::launch_autokey_chi2_async(
                device_in.value().data(), device_keys.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                prod_counts.value().data(), prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "ctak L>=T sync").ok());

    std::vector<std::uint32_t> decode_hist(C * 29);
    std::vector<std::uint32_t> prod_hist(C * 29);
    REQUIRE(decode_counts.value().copy_to_host(decode_hist).ok());
    REQUIRE(prod_counts.value().copy_to_host(prod_hist).ok());
    REQUIRE(prod_hist == decode_hist);

    // L==T recovers plaintext hist.
    StatusOr<HistAlphabetMap::Hist> plain_hist = HistAlphabetMap::count_stream_hist(plain);
    REQUIRE(plain_hist.ok());
    for (std::size_t b = 0; b < 29; ++b) {
        REQUIRE(prod_hist[b] == plain_hist.value()[b]);
    }

    // Host once-math for L>T (prefix over whole stream).
    StatusOr<HistAlphabetMap::Hist> host_gt =
        HistAlphabetMap::ctak_plain_hist_from_once(cipher, primer_gt);
    REQUIRE(host_gt.ok());
    for (std::size_t b = 0; b < 29; ++b) {
        REQUIRE(prod_hist[29u + b] == host_gt.value()[b]);
    }
}

TEST_CASE("DeepScoreBatch CTAK remap rejects L=0",
          "[cuda][batch][chi2][remap][ctak][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{0, 1, 2, 3});
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_keys =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{1});
    REQUIRE(device_keys.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
        DeviceBuffer<std::uint32_t>::from_host(std::vector<std::uint32_t>{0u});
    REQUIRE(device_begin.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_len =
        DeviceBuffer<std::uint32_t>::from_host(std::vector<std::uint32_t>{0u});
    REQUIRE(device_len.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::allocate(29);
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(1);
    REQUIRE(device_scores.ok());

    REQUIRE_FALSE(DeepScoreBatch::launch_autokey_chi2_async(
                      device_in.value().data(), device_keys.value().data(),
                      device_begin.value().data(), device_len.value().data(),
                      device_probs.value().data(), device_counts.value().data(),
                      device_scores.value().data(), 1, 4)
                      .ok());
}

#else

TEST_CASE("DeepScoreBatch CTAK remap skipped without CUDA",
          "[cuda][batch][chi2][remap][ctak]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
