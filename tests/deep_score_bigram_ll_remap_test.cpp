#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/score/hist_alphabet_map.hpp"

#include "cuda_error.hpp"
#include "deep_score_batch.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

class BigramLlRemapTestSupport {
public:
    [[nodiscard]] static std::array<float, HistAlphabetMap::bigram_bins> make_ll_table() {
        std::array<float, HistAlphabetMap::bigram_bins> ll{};
        for (std::size_t i = 0; i < ll.size(); ++i) {
            ll[i] = static_cast<float>(0.01 + 0.001 * static_cast<double>(i));
        }
        return ll;
    }

private:
    BigramLlRemapTestSupport() = delete;
};

TEST_CASE("DeepScoreBatch bigram-LL remap matches HistAlphabetMap canonical contract",
          "[cuda][batch][score][bigram][remap][golden]") {
    REQUIRE(ParcaeCuda::available());

    constexpr std::size_t T = 200;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 5u + 3u) % 29u);
    }

    constexpr std::size_t C = 29;
    std::vector<std::uint8_t> shifts(C);
    for (std::size_t c = 0; c < C; ++c) {
        shifts[c] = static_cast<std::uint8_t>(c);
    }
    const auto ll = BigramLlRemapTestSupport::make_ll_table();

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<float>> device_ll =
        DeviceBuffer<float>::from_host(std::span<const float>(ll.data(), ll.size()));
    REQUIRE(device_ll.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    REQUIRE(DeepScoreBatch::launch_caesar_bigram_ll_async(
                device_in.value().data(), device_shifts.value().data(), device_ll.value().data(),
                device_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "bigram remap sync").ok());

    std::vector<double> gpu(C);
    REQUIRE(device_scores.value().copy_to_host(gpu).ok());

    StatusOr<HistAlphabetMap::BigramHist> B = HistAlphabetMap::count_bigrams(host_in);
    REQUIRE(B.ok());
    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<double> host =
            HistAlphabetMap::bigram_ll_dot_rotated(B.value(), ll, shifts[c]);
        REQUIRE(host.ok());
        REQUIRE(gpu[c] == host.value());
    }
}

TEST_CASE("DeepScoreBatch bigram-LL remap vs decode ranking parity",
          "[cuda][batch][score][bigram][remap]") {
    REQUIRE(ParcaeCuda::available());

    constexpr std::size_t T = 128;
    std::vector<std::uint8_t> host_in(T);
    for (std::size_t i = 0; i < T; ++i) {
        host_in[i] = static_cast<std::uint8_t>((i * 7u + 11u) % 29u);
    }

    constexpr std::size_t C = 8;
    std::vector<std::uint8_t> shifts(C);
    for (std::size_t c = 0; c < C; ++c) {
        shifts[c] = static_cast<std::uint8_t>(c * 3u);
    }
    const auto ll = BigramLlRemapTestSupport::make_ll_table();

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<float>> device_ll =
        DeviceBuffer<float>::from_host(std::span<const float>(ll.data(), ll.size()));
    REQUIRE(device_ll.ok());
    StatusOr<DeviceBuffer<double>> decode_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(decode_scores.ok());
    StatusOr<DeviceBuffer<double>> remap_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(remap_scores.ok());

    REQUIRE(DeepScoreBatch::launch_caesar_bigram_ll_decode_async(
                device_in.value().data(), device_shifts.value().data(), device_ll.value().data(),
                decode_scores.value().data(), C, T)
                .ok());
    REQUIRE(DeepScoreBatch::launch_caesar_bigram_ll_async(
                device_in.value().data(), device_shifts.value().data(), device_ll.value().data(),
                remap_scores.value().data(), C, T)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "bigram wire sync").ok());

    std::vector<double> decode_host(C);
    std::vector<double> remap_host(C);
    REQUIRE(decode_scores.value().copy_to_host(decode_host).ok());
    REQUIRE(remap_scores.value().copy_to_host(remap_host).ok());

    // Canonical remap vs stream decode: same ranking; absolute FP may differ.
    std::size_t best_decode = 0;
    std::size_t best_remap = 0;
    for (std::size_t c = 1; c < C; ++c) {
        if (decode_host[c] < decode_host[best_decode]) {
            best_decode = c;
        }
        if (remap_host[c] < remap_host[best_remap]) {
            best_remap = c;
        }
        REQUIRE(remap_host[c] == Catch::Approx(decode_host[c]).margin(1e-6));
    }
    REQUIRE(best_remap == best_decode);
}

TEST_CASE("DeepScoreBatch bigram-LL remap T=1 yields zero scores",
          "[cuda][batch][score][bigram][remap][edge]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> host_in = {5};
    const std::vector<std::uint8_t> shifts = {0, 11};
    const auto ll = BigramLlRemapTestSupport::make_ll_table();
    constexpr std::size_t C = 2;

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    REQUIRE(device_shifts.ok());
    StatusOr<DeviceBuffer<float>> device_ll =
        DeviceBuffer<float>::from_host(std::span<const float>(ll.data(), ll.size()));
    REQUIRE(device_ll.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());

    REQUIRE(DeepScoreBatch::launch_caesar_bigram_ll_async(
                device_in.value().data(), device_shifts.value().data(), device_ll.value().data(),
                device_scores.value().data(), C, 1)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "bigram T=1 sync").ok());

    std::vector<double> gpu(C, 1.0);
    REQUIRE(device_scores.value().copy_to_host(gpu).ok());
    REQUIRE(gpu[0] == 0.0);
    REQUIRE(gpu[1] == 0.0);
}

#else

TEST_CASE("DeepScoreBatch bigram-LL remap skipped without CUDA",
          "[cuda][batch][score][bigram][remap]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
