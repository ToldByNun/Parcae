#include "parcae/core/index29.hpp"
#include "parcae/score/exact_match.hpp"
#include "parcae/score/hamming_agreement.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <vector>

#if defined(PARCAE_HAS_CUDA)

#include "exact_match_score.hpp"
#include "hamming_agreement_score.hpp"
#include "parcae_cuda.hpp"

static std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

TEST_CASE("CUDA exact_match matches CPU", "[cuda][score][exact]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> a{Index29{0}, Index29{1}, Index29{28}};
    const std::vector<Index29> b{Index29{0}, Index29{1}, Index29{28}};
    const std::vector<Index29> c{Index29{0}, Index29{1}, Index29{27}};

    StatusOr<double> cpu_eq = ExactMatch::score(a, b);
    StatusOr<double> cpu_ne = ExactMatch::score(a, c);
    REQUIRE(cpu_eq.ok());
    REQUIRE(cpu_ne.ok());

    StatusOr<double> cuda_eq = ExactMatchScore::score_host(to_bytes(a), to_bytes(b));
    StatusOr<double> cuda_ne = ExactMatchScore::score_host(to_bytes(a), to_bytes(c));
    REQUIRE(cuda_eq.ok());
    REQUIRE(cuda_ne.ok());
    REQUIRE(cuda_eq.value() == cpu_eq.value());
    REQUIRE(cuda_ne.value() == cpu_ne.value());
    REQUIRE(cuda_eq.value() == 1.0);
    REQUIRE(cuda_ne.value() == 0.0);
}

TEST_CASE("CUDA exact_match empty and length mismatch", "[cuda][score][exact]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> empty;
    StatusOr<double> empty_score = ExactMatchScore::score_host(empty, empty);
    REQUIRE(empty_score.ok());
    REQUIRE(empty_score.value() == 1.0);

    const std::vector<std::uint8_t> short_a{1, 2};
    const std::vector<std::uint8_t> short_b{1};
    StatusOr<double> mismatch = ExactMatchScore::score_host(short_a, short_b);
    REQUIRE(mismatch.ok());
    REQUIRE(mismatch.value() == 0.0);
}

TEST_CASE("CUDA exact_match random parity vs CPU", "[cuda][score][exact]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0x00E7ACC1u);
    std::uniform_int_distribution<int> dist(0, 28);

    for (int trial = 0; trial < 8; ++trial) {
        std::vector<Index29> cand;
        std::vector<Index29> ref;
        cand.reserve(257);
        ref.reserve(257);
        for (std::size_t i = 0; i < 257; ++i) {
            cand.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
            ref.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }
        if (trial % 2 == 0) {
            ref = cand;
        }

        StatusOr<double> cpu = ExactMatch::score(cand, ref);
        StatusOr<double> cuda = ExactMatchScore::score_host(to_bytes(cand), to_bytes(ref));
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        REQUIRE(cuda.value() == cpu.value());
    }
}

TEST_CASE("CUDA hamming_agreement matches CPU", "[cuda][score][hamming]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> cand{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
    const std::vector<Index29> ref{Index29{0}, Index29{9}, Index29{2}, Index29{8}};

    StatusOr<double> cpu = HammingAgreement::score(cand, ref);
    StatusOr<double> cuda = HammingAgreementScore::score_host(to_bytes(cand), to_bytes(ref));
    REQUIRE(cpu.ok());
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value() == cpu.value());
    REQUIRE(cuda.value() == 0.5);
}

TEST_CASE("CUDA hamming_agreement rejects empty and length mismatch", "[cuda][score][hamming]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> empty;
    REQUIRE_FALSE(HammingAgreementScore::score_host(empty, empty).ok());

    const std::vector<std::uint8_t> a{1, 2, 3};
    const std::vector<std::uint8_t> b{1, 2};
    REQUIRE_FALSE(HammingAgreementScore::score_host(a, b).ok());
}

TEST_CASE("CUDA hamming_agreement random parity vs CPU", "[cuda][score][hamming]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xA771u);
    std::uniform_int_distribution<int> dist(0, 28);

    for (int trial = 0; trial < 8; ++trial) {
        std::vector<Index29> cand;
        std::vector<Index29> ref;
        const std::size_t n = 64 + static_cast<std::size_t>(trial) * 17;
        cand.reserve(n);
        ref.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            cand.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
            ref.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }

        StatusOr<double> cpu = HammingAgreement::score(cand, ref);
        StatusOr<double> cuda = HammingAgreementScore::score_host(to_bytes(cand), to_bytes(ref));
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        REQUIRE(cuda.value() == cpu.value());
    }
}

#else

TEST_CASE("CUDA exact_match skipped (PARCAE_HAS_CUDA unset)", "[cuda][score][exact]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise ExactMatchScore");
}

TEST_CASE("CUDA hamming_agreement skipped (PARCAE_HAS_CUDA unset)", "[cuda][score][hamming]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise HammingAgreementScore");
}

#endif
