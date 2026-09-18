#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "parcae/core/index29.hpp"
#include "parcae/score/chi2_english_gp.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

#if defined(PARCAE_HAS_CUDA)

#include "chi2_english_gp_score.hpp"
#include "parcae_cuda.hpp"

static std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

static ExpectedFrequencyTable load_locked_table() {
    StatusOr<ExpectedFrequencyTable> table = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(table.ok());
    return table.value();
}

static ExpectedFrequencyTable synthetic_uniform_table() {
    std::array<std::uint64_t, ExpectedFrequencyTable::alphabet_size> raw{};
    for (std::size_t i = 0; i < raw.size(); ++i) {
        raw[i] = 1;
    }
    StatusOr<ExpectedFrequencyTable> table =
        ExpectedFrequencyTable::from_raw_counts("synth-uniform", raw, {});
    REQUIRE(table.ok());
    return table.value();
}

TEST_CASE("CUDA chi2_english_gp_v0 matches CPU locked table", "[cuda][score][chi2]") {
    REQUIRE(ParcaeCuda::available());

    const ExpectedFrequencyTable table = load_locked_table();
    const std::vector<Index29> xs{
        Index29{0}, Index29{1}, Index29{2}, Index29{3}, Index29{4},
        Index29{5}, Index29{6}, Index29{7}, Index29{8}, Index29{9},
    };

    StatusOr<double> cpu = Chi2EnglishGp::score(xs, table);
    StatusOr<double> cuda =
        Chi2EnglishGpScore::score_host(to_bytes(xs), table.probabilities());
    REQUIRE(cpu.ok());
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value() == cpu.value());
}

TEST_CASE("CUDA chi2_english_gp_v0 rejects empty and bad probs", "[cuda][score][chi2]") {
    REQUIRE(ParcaeCuda::available());

    const ExpectedFrequencyTable table = synthetic_uniform_table();
    const std::vector<std::uint8_t> empty;
    REQUIRE_FALSE(Chi2EnglishGpScore::score_host(empty, table.probabilities()).ok());

    std::array<double, 29> bad = table.probabilities();
    bad[3] = 0.0;
    const std::vector<std::uint8_t> xs{1, 2, 3};
    REQUIRE_FALSE(
        Chi2EnglishGpScore::score_host(xs, std::span<const double>(bad.data(), bad.size())).ok());
}

TEST_CASE("CUDA chi2_english_gp_v0 random parity vs CPU", "[cuda][score][chi2]") {
    REQUIRE(ParcaeCuda::available());

    const ExpectedFrequencyTable table = load_locked_table();
    std::mt19937 rng(0xC41229u);
    std::uniform_int_distribution<int> dist(0, 28);

    for (int trial = 0; trial < 8; ++trial) {
        std::vector<Index29> indices;
        const std::size_t n = 16 + static_cast<std::size_t>(trial) * 29;
        indices.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            indices.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }

        StatusOr<double> cpu = Chi2EnglishGp::score(indices, table);
        StatusOr<double> cuda =
            Chi2EnglishGpScore::score_host(to_bytes(indices), table.probabilities());
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        REQUIRE(cuda.value() == cpu.value());
    }
}

#else

TEST_CASE("CUDA chi2_english_gp_v0 skipped (PARCAE_HAS_CUDA unset)", "[cuda][score][chi2]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise Chi2EnglishGpScore");
}

#endif
