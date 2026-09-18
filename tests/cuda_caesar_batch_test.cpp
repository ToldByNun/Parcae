#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "caesar_batch_kernel.hpp"
#include "candidate_batch_buffers.hpp"
#include "parcae_cuda.hpp"
#include "params.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/transform/caesar_transform.hpp"
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

TEST_CASE("CUDA caesar batch 29 shifts matches CaesarCandidateGenerator", "[cuda][batch][caesar]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> cipher = cipher_fixture();

    StatusOr<std::vector<TransformCandidate>> cpu =
        CaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().size() == Index29::modulus);

    CandidateBatchBuffers::AllocateOptions options;
    options.token_layout = CandidateBatchBuffers::TokenLayout::Shared;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, cipher.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    // directions default to Decrypt (0)

    REQUIRE(CaesarBatchKernel::apply_host(buffers.value()).ok());

    for (std::size_t c = 0; c < Index29::modulus; ++c) {
        std::vector<Index29> cuda_out(cipher.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
        REQUIRE(cpu.value()[c].params().at("shift").get<int>() == static_cast<int>(c));
    }
}

TEST_CASE("CUDA caesar batch encrypt sweep matches CPU kernel", "[cuda][batch][caesar]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> plain = cipher_fixture();

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, plain.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(plain).ok());
    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    for (std::size_t c = 0; c < Index29::modulus; ++c) {
        buffers.value().directions()[c] = static_cast<std::uint8_t>(CudaDir::Encrypt);
    }

    REQUIRE(CaesarBatchKernel::apply_host(buffers.value()).ok());

    for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(CaesarTransform::kernel(
                    plain, cpu_out, Index29{shift}, TransformDirection::Encrypt)
                    .ok());

        std::vector<Index29> cuda_out(plain.size());
        REQUIRE(buffers.value().copy_out_candidate(shift, cuda_out).ok());
        REQUIRE(cuda_out == cpu_out);
    }
}

TEST_CASE("CUDA caesar batch random stream parity", "[cuda][batch][caesar]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xBA7CE5u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> cipher;
    cipher.reserve(128);
    for (std::size_t i = 0; i < 128; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }

    StatusOr<std::vector<TransformCandidate>> cpu =
        CaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, cipher.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    REQUIRE(CaesarBatchKernel::apply_host(buffers.value()).ok());

    for (std::size_t c = 0; c < Index29::modulus; ++c) {
        std::vector<Index29> cuda_out(cipher.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
    }
}

TEST_CASE("CUDA caesar batch rejects bad layout and shifts", "[cuda][batch][caesar]") {
    REQUIRE(ParcaeCuda::available());

    CandidateBatchBuffers::AllocateOptions per;
    per.token_layout = CandidateBatchBuffers::TokenLayout::PerCandidate;
    per.family = CudaFamilyId::Caesar;
    StatusOr<CandidateBatchBuffers> bad_layout = CandidateBatchBuffers::allocate(2, 3, per);
    REQUIRE(bad_layout.ok());
    REQUIRE_FALSE(CaesarBatchKernel::apply_host(bad_layout.value()).ok());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> shifts{0, 29};
    std::vector<std::uint8_t> dirs{0, 0};
    std::vector<std::uint8_t> out(4);
    REQUIRE_FALSE(CaesarBatchKernel::apply_host(in, shifts, dirs, out).ok());
}

#else

TEST_CASE("CUDA caesar batch skipped (PARCAE_HAS_CUDA unset)", "[cuda][batch][caesar]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CaesarBatchKernel");
}

#endif
