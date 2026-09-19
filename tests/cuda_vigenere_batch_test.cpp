#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "candidate_batch_buffers.hpp"
#include "interrupt_device_view.hpp"
#include "parcae_cuda.hpp"
#include "params.hpp"
#include "vigenere_batch_kernel.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] std::vector<Index29> cipher_fixture() {
    return {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
}

[[nodiscard]] StatusOr<CandidateBatchBuffers> make_vigenere_buffers(
    const std::vector<Index29>& tokens,
    const std::vector<std::vector<Index29>>& keys) {
    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::VigenereKey;
    options.with_scores = false;
    options.key_arena_capacity = CandidateBatchBuffers::key_arena_bytes_needed(keys);

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(keys.size(), tokens.size(), options);
    if (!buffers.ok()) {
        return buffers.status();
    }
    Status set = buffers.value().set_shared_tokens(tokens);
    if (!set.ok()) {
        return set;
    }
    Status packed = buffers.value().pack_explicit_keys(keys);
    if (!packed.ok()) {
        return packed;
    }
    return buffers;
}

}  // namespace

TEST_CASE(
    "CUDA vigenere batch matches VigenereExplicitKeyCandidateGenerator",
    "[cuda][batch][vigenere]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> cipher = cipher_fixture();
    const std::vector<std::vector<Index29>> keys = {
        {I(9), I(9)},
        {I(1), I(2)},
        {I(3), I(4), I(5)},
    };

    StatusOr<std::vector<TransformCandidate>> cpu =
        VigenereExplicitKeyCandidateGenerator::generate(cipher, keys);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().size() == keys.size());

    StatusOr<CandidateBatchBuffers> buffers = make_vigenere_buffers(cipher, keys);
    REQUIRE(buffers.ok());

    StatusOr<InterruptDeviceView> interrupts =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), cipher.size());
    REQUIRE(interrupts.ok());
    REQUIRE(VigenereBatchKernel::apply_host(buffers.value(), interrupts.value()).ok());

    for (std::size_t c = 0; c < keys.size(); ++c) {
        std::vector<Index29> cuda_out(cipher.size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
    }
}

TEST_CASE("CUDA vigenere batch with shared interrupts", "[cuda][batch][vigenere]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> plain = cipher_fixture();
    const std::vector<Index29> correct_key = {I(1), I(2)};
    StatusOr<InterruptPolicy> policy = InterruptPolicy::from_skip_indices({1, 3});
    REQUIRE(policy.ok());

    StatusOr<std::vector<Index29>> cipher = VigenereKeyTransform{}.apply(
        plain,
        nlohmann::json{{"key_indices", {1, 2}}},
        TransformDirection::Encrypt,
        policy.value());
    REQUIRE(cipher.ok());

    const std::vector<std::vector<Index29>> keys = {
        {I(7), I(8)},
        correct_key,
        {I(1), I(3)},
    };

    StatusOr<std::vector<TransformCandidate>> cpu = VigenereExplicitKeyCandidateGenerator::generate(
        cipher.value(), keys, TransformDirection::Decrypt, policy.value());
    REQUIRE(cpu.ok());

    StatusOr<CandidateBatchBuffers> buffers = make_vigenere_buffers(cipher.value(), keys);
    REQUIRE(buffers.ok());
    StatusOr<InterruptDeviceView> interrupts =
        InterruptDeviceView::from_policy(policy.value(), cipher.value().size());
    REQUIRE(interrupts.ok());
    REQUIRE(VigenereBatchKernel::apply_host(buffers.value(), interrupts.value()).ok());

    for (std::size_t c = 0; c < keys.size(); ++c) {
        std::vector<Index29> cuda_out(cipher.value().size());
        REQUIRE(buffers.value().copy_out_candidate(c, cuda_out).ok());
        REQUIRE(cuda_out == cpu.value()[c].output_indices());
    }

    std::vector<Index29> recovered(plain.size());
    REQUIRE(buffers.value().copy_out_candidate(1, recovered).ok());
    REQUIRE(recovered == plain);
}

TEST_CASE("CUDA vigenere batch encrypt directions", "[cuda][batch][vigenere]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> plain = {I(0), I(5), I(10), I(28)};
    const std::vector<std::vector<Index29>> keys = {{I(3), I(7)}};

    StatusOr<CandidateBatchBuffers> buffers = make_vigenere_buffers(plain, keys);
    REQUIRE(buffers.ok());
    buffers.value().directions()[0] = static_cast<std::uint8_t>(CudaDir::Encrypt);

    StatusOr<InterruptDeviceView> interrupts =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), plain.size());
    REQUIRE(interrupts.ok());
    REQUIRE(VigenereBatchKernel::apply_host(buffers.value(), interrupts.value()).ok());

    StatusOr<std::vector<Index29>> cpu = VigenereKeyTransform{}.apply(
        plain,
        nlohmann::json{{"key_indices", {3, 7}}},
        TransformDirection::Encrypt);
    REQUIRE(cpu.ok());

    std::vector<Index29> cuda_out(plain.size());
    REQUIRE(buffers.value().copy_out_candidate(0, cuda_out).ok());
    REQUIRE(cuda_out == cpu.value());
}

TEST_CASE("CUDA vigenere batch rejects empty keys and wrong family", "[cuda][batch][vigenere]") {
    REQUIRE(ParcaeCuda::available());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = false;
    StatusOr<CandidateBatchBuffers> wrong = CandidateBatchBuffers::allocate(1, 3, options);
    REQUIRE(wrong.ok());
    StatusOr<InterruptDeviceView> interrupts =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), 3);
    REQUIRE(interrupts.ok());
    REQUIRE_FALSE(VigenereBatchKernel::apply_host(wrong.value(), interrupts.value()).ok());

    const std::vector<std::vector<Index29>> keys = {{I(1)}, {}};
    CandidateBatchBuffers::AllocateOptions vig;
    vig.family = CudaFamilyId::VigenereKey;
    vig.key_arena_capacity = 4;
    vig.with_scores = false;
    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(2, 3, vig);
    REQUIRE(buffers.ok());
    REQUIRE_FALSE(buffers.value().pack_explicit_keys(keys).ok());
}

#else

TEST_CASE("CUDA vigenere batch skipped (PARCAE_HAS_CUDA unset)", "[cuda][batch][vigenere]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise VigenereBatchKernel");
}

#endif
