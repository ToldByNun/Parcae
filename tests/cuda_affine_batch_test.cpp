#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "affine_batch_kernel.hpp"
#include "candidate_batch_buffers.hpp"
#include "parcae_cuda.hpp"
#include "params.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstdint>
#include <random>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] std::vector<Index29> cipher_fixture() {
    return {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
}

}  // namespace

TEST_CASE(
    "CUDA affine batch 812 matches AffineCandidateGenerator",
    "[cuda][batch][affine]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> cipher = cipher_fixture();
    StatusOr<std::vector<TransformCandidate>> cpu =
        AffineCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().size() == AffineCandidateGenerator::candidate_count);

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Affine;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(
        AffineCandidateGenerator::candidate_count, cipher.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(buffers.value().fill_affine_ab_nested().ok());
    REQUIRE(AffineBatchKernel::apply_host(buffers.value()).ok());

    for (std::size_t c = 0; c < AffineCandidateGenerator::candidate_count; ++c) {
        std::vector<Index29> cuda_out(cipher.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
        REQUIRE(
            buffers.value().affine_a()[c] ==
            static_cast<std::uint8_t>(cpu.value()[c].params().at("a").get<int>()));
        REQUIRE(
            buffers.value().affine_b()[c] ==
            static_cast<std::uint8_t>(cpu.value()[c].params().at("b").get<int>()));
    }
}

TEST_CASE("CUDA affine batch encrypt sweep matches CPU kernel", "[cuda][batch][affine]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> plain = cipher_fixture();

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Affine;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(
        AffineCandidateGenerator::candidate_count, plain.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(plain).ok());
    REQUIRE(buffers.value().fill_affine_ab_nested().ok());
    for (std::size_t c = 0; c < buffers.value().candidate_count(); ++c) {
        buffers.value().directions()[c] = static_cast<std::uint8_t>(CudaDir::Encrypt);
    }
    REQUIRE(AffineBatchKernel::apply_host(buffers.value()).ok());

    // Spot-check a few (a,b) lanes against CPU; full 812 encrypt already covered by decrypt
    // generator parity in the sibling case for the inverse maps.
    const std::size_t samples[] = {0, 28, 29, 100, 811};
    for (std::size_t c : samples) {
        const std::uint8_t a = buffers.value().affine_a()[c];
        const std::uint8_t b = buffers.value().affine_b()[c];
        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(AffineTransform::kernel(
                    plain,
                    cpu_out,
                    Index29{a},
                    Index29{b},
                    TransformDirection::Encrypt)
                    .ok());

        std::vector<Index29> cuda_out(plain.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu_out);
    }
}

TEST_CASE("CUDA affine batch random stream parity", "[cuda][batch][affine]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xAFF1AEu);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> cipher;
    cipher.reserve(64);
    for (std::size_t i = 0; i < 64; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }

    StatusOr<std::vector<TransformCandidate>> cpu =
        AffineCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Affine;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(
        AffineCandidateGenerator::candidate_count, cipher.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(buffers.value().fill_affine_ab_nested().ok());
    REQUIRE(AffineBatchKernel::apply_host(buffers.value()).ok());

    for (std::size_t c = 0; c < AffineCandidateGenerator::candidate_count; ++c) {
        std::vector<Index29> cuda_out(cipher.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
    }
}

TEST_CASE("CUDA affine batch rejects bad a and wrong family", "[cuda][batch][affine]") {
    REQUIRE(ParcaeCuda::available());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = false;
    StatusOr<CandidateBatchBuffers> wrong = CandidateBatchBuffers::allocate(2, 3, options);
    REQUIRE(wrong.ok());
    REQUIRE_FALSE(AffineBatchKernel::apply_host(wrong.value()).ok());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> a{0};  // invalid
    std::vector<std::uint8_t> b{0};
    std::vector<std::uint8_t> dirs{0};
    std::vector<std::uint8_t> out(2);
    REQUIRE_FALSE(AffineBatchKernel::apply_host(in, a, b, dirs, out).ok());
}

#else

TEST_CASE("CUDA affine batch skipped (PARCAE_HAS_CUDA unset)", "[cuda][batch][affine]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise AffineBatchKernel");
}

#endif
