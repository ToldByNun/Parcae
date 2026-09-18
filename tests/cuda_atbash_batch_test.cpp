#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "atbash_batch_kernel.hpp"
#include "atbash_caesar_batch_kernel.hpp"
#include "candidate_batch_buffers.hpp"
#include "parcae_cuda.hpp"
#include "params.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] std::vector<Index29> cipher_fixture() {
    return {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
}

}  // namespace

TEST_CASE("CUDA atbash batch matches AtbashCandidateGenerator", "[cuda][batch][atbash]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> cipher = cipher_fixture();
    StatusOr<std::vector<TransformCandidate>> cpu =
        AtbashCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().size() == 1);

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Atbash;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(1, cipher.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(AtbashBatchKernel::apply_host(buffers.value()).ok());

    std::vector<Index29> cuda_out(cipher.size());
    REQUIRE(buffers.value().copy_out_candidate(0, cuda_out).ok());
    REQUIRE(cuda_out == cpu.value()[0].output_indices());
}

TEST_CASE(
    "CUDA atbash_caesar batch 29 shifts matches AtbashCaesarCandidateGenerator",
    "[cuda][batch][atbash]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> cipher = cipher_fixture();
    StatusOr<std::vector<TransformCandidate>> cpu =
        AtbashCaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().size() == Index29::modulus);

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Compose;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, cipher.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().caesar_shifts().size() == Index29::modulus);
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    REQUIRE(AtbashCaesarBatchKernel::apply_host(buffers.value()).ok());

    for (std::size_t c = 0; c < Index29::modulus; ++c) {
        std::vector<Index29> cuda_out(cipher.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
    }
}

TEST_CASE("CUDA atbash_caesar batch encrypt round-trip parity", "[cuda][batch][atbash]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> plain = cipher_fixture();
    constexpr std::uint8_t kShift = 7;

    StatusOr<std::vector<Index29>> cipher =
        ComposeTransform::apply_atbash_then_caesar(plain, kShift, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Compose;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> enc =
        CandidateBatchBuffers::allocate(Index29::modulus, plain.size(), options);
    REQUIRE(enc.ok());
    REQUIRE(enc.value().set_shared_tokens(plain).ok());
    REQUIRE(enc.value().fill_caesar_shifts_iota().ok());
    for (std::size_t c = 0; c < Index29::modulus; ++c) {
        enc.value().directions()[c] = static_cast<std::uint8_t>(CudaDir::Encrypt);
    }
    REQUIRE(AtbashCaesarBatchKernel::apply_host(enc.value()).ok());

    std::vector<Index29> cuda_cipher(plain.size());
    REQUIRE(enc.value().copy_out_candidate(kShift, cuda_cipher).ok());
    REQUIRE(cuda_cipher == cipher.value());

    StatusOr<CandidateBatchBuffers> dec =
        CandidateBatchBuffers::allocate(Index29::modulus, cipher.value().size(), options);
    REQUIRE(dec.ok());
    REQUIRE(dec.value().set_shared_tokens(cipher.value()).ok());
    REQUIRE(dec.value().fill_caesar_shifts_iota().ok());
    REQUIRE(AtbashCaesarBatchKernel::apply_host(dec.value()).ok());

    std::vector<Index29> cuda_plain(plain.size());
    REQUIRE(dec.value().copy_out_candidate(kShift, cuda_plain).ok());
    REQUIRE(cuda_plain == plain);
}

TEST_CASE("CUDA atbash_caesar batch random stream parity", "[cuda][batch][atbash]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xA7BA54u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> cipher;
    cipher.reserve(96);
    for (std::size_t i = 0; i < 96; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }

    StatusOr<std::vector<TransformCandidate>> cpu =
        AtbashCaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Compose;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, cipher.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    REQUIRE(AtbashCaesarBatchKernel::apply_host(buffers.value()).ok());

    for (std::size_t c = 0; c < Index29::modulus; ++c) {
        std::vector<Index29> cuda_out(cipher.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
    }
}

TEST_CASE("CUDA atbash batch rejects wrong family", "[cuda][batch][atbash]") {
    REQUIRE(ParcaeCuda::available());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = false;
    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(1, 3, options);
    REQUIRE(buffers.ok());
    REQUIRE_FALSE(AtbashBatchKernel::apply_host(buffers.value()).ok());
    REQUIRE_FALSE(AtbashCaesarBatchKernel::apply_host(buffers.value()).ok());
}

#else

TEST_CASE("CUDA atbash batch skipped (PARCAE_HAS_CUDA unset)", "[cuda][batch][atbash]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise atbash batch kernels");
}

#endif
