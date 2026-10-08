#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
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

class BeaufortRemapTestSupport {
public:
    [[nodiscard]] static std::vector<Index29> make_plain(std::size_t n) {
        std::vector<Index29> plain;
        plain.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            plain.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 3u) % 29u)});
        }
        return plain;
    }

    [[nodiscard]] static std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
        std::vector<std::uint8_t> out(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            out[i] = indices[i].value();
        }
        return out;
    }

private:
    BeaufortRemapTestSupport() = delete;
};

TEST_CASE("FamilyChi2Batch Beaufort remap matches decode-hist (mixed L)",
          "[cuda][batch][chi2][remap][beaufort][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> plain = BeaufortRemapTestSupport::make_plain(360);
    const std::vector<std::uint8_t> key_a = {3, 7, 11};
    const std::vector<std::uint8_t> key_b = {1, 2, 3, 4, 5};
    const std::vector<std::uint8_t> key_c = {28, 0};

    StatusOr<std::vector<Index29>> cipher = BeaufortKeyTransform{}.apply(
        plain,
        nlohmann::json{{"key_indices", nlohmann::json::array({3, 7, 11})}},
        TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    const std::size_t T = cipher.value().size();
    std::vector<std::uint8_t> host_in = BeaufortRemapTestSupport::to_bytes(cipher.value());

    constexpr std::size_t C = 3;
    std::vector<std::uint8_t> key_bytes;
    key_bytes.insert(key_bytes.end(), key_a.begin(), key_a.end());
    key_bytes.insert(key_bytes.end(), key_b.begin(), key_b.end());
    key_bytes.insert(key_bytes.end(), key_c.begin(), key_c.end());
    const std::vector<std::uint32_t> key_begin = {0u, 3u, 8u};
    const std::vector<std::uint32_t> key_len = {3u, 5u, 2u};

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
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

    REQUIRE(FamilyChi2Batch::launch_beaufort_decode_hist_async(
                device_in.value().data(), device_keys.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                decode_counts.value().data(), decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(FamilyChi2Batch::launch_beaufort_async(
                device_in.value().data(), device_keys.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                prod_counts.value().data(), prod_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "beaufort wire sync").ok());

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

    StatusOr<HistAlphabetMap::Hist> plain_hist =
        HistAlphabetMap::count_stream_hist(BeaufortRemapTestSupport::to_bytes(plain));
    REQUIRE(plain_hist.ok());
    for (std::size_t b = 0; b < 29; ++b) {
        REQUIRE(prod_hist[b] == plain_hist.value()[b]);
    }
}

TEST_CASE("AlphabetChi2Batch Beaufort remap matches HistAlphabetMap host math",
          "[cuda][batch][chi2][remap][beaufort]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    constexpr std::size_t T = 500;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 4u + 9u) % 29u);
    }

    const std::vector<std::uint8_t> key_bytes = {2, 5, 8, 1, 4, 7, 10};
    const std::vector<std::uint32_t> key_begin = {0u, 3u};
    const std::vector<std::uint32_t> key_len = {3u, 4u};
    constexpr std::size_t C = 2;

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
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
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * 29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    REQUIRE(AlphabetChi2Batch::launch_beaufort(
                device_in.value().data(), device_keys.value().data(), device_begin.value().data(),
                device_len.value().data(), device_probs.value().data(),
                /*device_column_scratch=*/nullptr, device_counts.value().data(),
                device_scores.value().data(), C, T)
                .ok());

    std::vector<std::uint32_t> counts_host(C * 29);
    REQUIRE(device_counts.value().copy_to_host(counts_host).ok());

    for (std::size_t c = 0; c < C; ++c) {
        const std::uint32_t L = key_len[c];
        StatusOr<std::vector<std::uint32_t>> cols =
            HistAlphabetMap::count_column_hist(host_in, L);
        REQUIRE(cols.ok());
        std::span<const std::uint8_t> key(key_bytes.data() + key_begin[c], L);
        StatusOr<HistAlphabetMap::Hist> P =
            HistAlphabetMap::beaufort_plain_hist_from_columns(cols.value(), L, key);
        REQUIRE(P.ok());
        for (std::size_t b = 0; b < 29; ++b) {
            REQUIRE(counts_host[c * 29u + b] == P.value()[b]);
        }
    }
}

TEST_CASE("FamilyChi2Batch Beaufort remap rejects bad args",
          "[cuda][batch][chi2][remap][beaufort][edge]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{0, 1, 2, 3});
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_keys =
        DeviceBuffer<std::uint8_t>::from_host(std::vector<std::uint8_t>{1, 2});
    REQUIRE(device_keys.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
        DeviceBuffer<std::uint32_t>::from_host(std::vector<std::uint32_t>{0u});
    REQUIRE(device_begin.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_len =
        DeviceBuffer<std::uint32_t>::from_host(std::vector<std::uint32_t>{2u});
    REQUIRE(device_len.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::allocate(29);
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(29);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(1);
    REQUIRE(device_scores.ok());

    REQUIRE_FALSE(FamilyChi2Batch::launch_beaufort_async(
                      nullptr, device_keys.value().data(), device_begin.value().data(),
                      device_len.value().data(), device_probs.value().data(),
                      device_counts.value().data(), device_scores.value().data(), 1, 4)
                      .ok());
    REQUIRE_FALSE(FamilyChi2Batch::launch_beaufort_async(
                      device_in.value().data(), device_keys.value().data(),
                      device_begin.value().data(), device_len.value().data(),
                      device_probs.value().data(), device_counts.value().data(),
                      device_scores.value().data(), 0, 4)
                      .ok());
}

#else

TEST_CASE("FamilyChi2Batch Beaufort remap skipped without CUDA",
          "[cuda][batch][chi2][remap][beaufort]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
