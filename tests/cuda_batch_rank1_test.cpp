#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/batch/batch_result.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include "affine_batch_kernel.hpp"
#include "atbash_batch_kernel.hpp"
#include "atbash_caesar_batch_kernel.hpp"
#include "caesar_batch_kernel.hpp"
#include "candidate_batch_buffers.hpp"
#include "cuda_batch_score.hpp"
#include "interrupt_device_view.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"
#include "vigenere_batch_kernel.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] const std::vector<Index29>& plain_fixture() {
    static const std::vector<Index29> plain = {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
    return plain;
}

[[nodiscard]] ScoreRequest exact_match_request(const std::vector<Index29>& plain) {
    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);
    return request;
}

/// Require exact-match=1.0 at rank 1; return its source_index.
[[nodiscard]] std::size_t require_rank1_exact_match(const BatchResult& result) {
    REQUIRE_FALSE(result.top().empty());
    REQUIRE(result.top()[0].score() == 1.0);
    return result.top()[0].source_index();
}

[[nodiscard]] std::size_t count_perfect_scores(std::span<const double> scores) {
    std::size_t hits = 0;
    for (double score : scores) {
        if (score == 1.0) {
            ++hits;
        }
    }
    return hits;
}

} // namespace

TEST_CASE("CUDA batch rank-1 recovery under exact_match", "[cuda][batch][rank1]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(CudaBatchScore::available());

    const std::vector<Index29>& plain = plain_fixture();
    const ScoreRequest request = exact_match_request(plain);

    SECTION("caesar") {
        constexpr int kShift = 7;
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain, nlohmann::json{{"shift", kShift}}, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        CandidateBatchBuffers::AllocateOptions options;
        options.family = CudaFamilyId::Caesar;
        options.with_scores = true;

        StatusOr<CandidateBatchBuffers> buffers =
            CandidateBatchBuffers::allocate(Index29::modulus, cipher.value().size(), options);
        REQUIRE(buffers.ok());
        REQUIRE(buffers.value().set_shared_tokens(cipher.value()).ok());
        REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
        REQUIRE(CaesarBatchKernel::apply_host(buffers.value()).ok());

        std::vector<std::string> ids;
        ids.reserve(Index29::modulus);
        for (std::uint8_t s = 0; s < Index29::modulus; ++s) {
            ids.push_back(CaesarCandidateGenerator::make_candidate_id(s));
        }

        StatusOr<BatchResult> result = CudaBatchScore::score_and_top_k(
            buffers.value(), ids, "exact_match", /*k=*/Index29::modulus, request);
        REQUIRE(result.ok());
        REQUIRE(count_perfect_scores(buffers.value().scores()) == 1);
        REQUIRE(require_rank1_exact_match(result.value()) == static_cast<std::size_t>(kShift));
        REQUIRE(result.value().top()[0].candidate_id() ==
                CaesarCandidateGenerator::make_candidate_id(static_cast<std::uint8_t>(kShift)));
    }

    SECTION("atbash") {
        StatusOr<std::vector<Index29>> cipher =
            AtbashTransform{}.apply(plain, nlohmann::json::object(), TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        CandidateBatchBuffers::AllocateOptions options;
        options.family = CudaFamilyId::Atbash;
        options.with_scores = true;

        StatusOr<CandidateBatchBuffers> buffers =
            CandidateBatchBuffers::allocate(1, cipher.value().size(), options);
        REQUIRE(buffers.ok());
        REQUIRE(buffers.value().set_shared_tokens(cipher.value()).ok());
        REQUIRE(AtbashBatchKernel::apply_host(buffers.value()).ok());

        const std::vector<std::string> ids = {AtbashCandidateGenerator::make_candidate_id()};
        StatusOr<BatchResult> result =
            CudaBatchScore::score_and_top_k(buffers.value(), ids, "exact_match", /*k=*/1, request);
        REQUIRE(result.ok());
        REQUIRE(require_rank1_exact_match(result.value()) == 0);
        REQUIRE(result.value().top()[0].candidate_id() == "atbash");
    }

    SECTION("atbash_caesar") {
        constexpr std::uint8_t kShift = 3;
        StatusOr<std::vector<Index29>> cipher =
            ComposeTransform::apply_atbash_then_caesar(plain, kShift, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        CandidateBatchBuffers::AllocateOptions options;
        options.family = CudaFamilyId::Compose;
        options.with_scores = true;

        StatusOr<CandidateBatchBuffers> buffers =
            CandidateBatchBuffers::allocate(Index29::modulus, cipher.value().size(), options);
        REQUIRE(buffers.ok());
        REQUIRE(buffers.value().set_shared_tokens(cipher.value()).ok());
        REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
        REQUIRE(AtbashCaesarBatchKernel::apply_host(buffers.value()).ok());

        std::vector<std::string> ids;
        ids.reserve(Index29::modulus);
        for (std::uint8_t s = 0; s < Index29::modulus; ++s) {
            ids.push_back(AtbashCaesarCandidateGenerator::make_candidate_id(s));
        }

        StatusOr<BatchResult> result = CudaBatchScore::score_and_top_k(
            buffers.value(), ids, "exact_match", /*k=*/Index29::modulus, request);
        REQUIRE(result.ok());
        REQUIRE(count_perfect_scores(buffers.value().scores()) == 1);
        REQUIRE(require_rank1_exact_match(result.value()) == kShift);
    }

    SECTION("affine") {
        constexpr int kA = 2;
        constexpr int kB = 5;
        StatusOr<std::vector<Index29>> cipher = AffineTransform{}.apply(
            plain, nlohmann::json{{"a", kA}, {"b", kB}}, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        CandidateBatchBuffers::AllocateOptions options;
        options.family = CudaFamilyId::Affine;
        options.with_scores = true;

        StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(
            AffineCandidateGenerator::candidate_count, cipher.value().size(), options);
        REQUIRE(buffers.ok());
        REQUIRE(buffers.value().set_shared_tokens(cipher.value()).ok());
        REQUIRE(buffers.value().fill_affine_ab_nested().ok());
        REQUIRE(AffineBatchKernel::apply_host(buffers.value()).ok());

        std::vector<std::string> ids;
        ids.reserve(AffineCandidateGenerator::candidate_count);
        for (std::uint8_t a = 1; a < Index29::modulus; ++a) {
            for (std::uint8_t b = 0; b < Index29::modulus; ++b) {
                ids.push_back(AffineCandidateGenerator::make_candidate_id(a, b));
            }
        }

        StatusOr<BatchResult> result = CudaBatchScore::score_and_top_k(
            buffers.value(), ids, "exact_match",
            /*k=*/AffineCandidateGenerator::candidate_count, request);
        REQUIRE(result.ok());
        REQUIRE(count_perfect_scores(buffers.value().scores()) == 1);
        const std::size_t expected_index =
            static_cast<std::size_t>(kA - 1) * 29u + static_cast<std::size_t>(kB);
        REQUIRE(require_rank1_exact_match(result.value()) == expected_index);
        REQUIRE(result.value().top()[0].candidate_id() ==
                AffineCandidateGenerator::make_candidate_id(static_cast<std::uint8_t>(kA),
                                                            static_cast<std::uint8_t>(kB)));
    }

    SECTION("vigenere_explicit_keys") {
        const std::vector<Index29> correct_key = {I(1), I(2), I(5)};
        const std::vector<std::vector<Index29>> key_list = {
            {I(0), I(0), I(0)},
            correct_key,
            {I(4), I(4)},
            {I(1), I(2), I(6)},
        };

        StatusOr<std::vector<Index29>> cipher = VigenereKeyTransform{}.apply(
            plain, nlohmann::json{{"key_indices", {1, 2, 5}}}, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        CandidateBatchBuffers::AllocateOptions options;
        options.family = CudaFamilyId::VigenereKey;
        options.with_scores = true;
        options.key_arena_capacity = CandidateBatchBuffers::key_arena_bytes_needed(key_list);

        StatusOr<CandidateBatchBuffers> buffers =
            CandidateBatchBuffers::allocate(key_list.size(), cipher.value().size(), options);
        REQUIRE(buffers.ok());
        REQUIRE(buffers.value().set_shared_tokens(cipher.value()).ok());
        REQUIRE(buffers.value().pack_explicit_keys(key_list).ok());

        StatusOr<InterruptDeviceView> interrupts =
            InterruptDeviceView::from_policy(InterruptPolicy::none(), cipher.value().size());
        REQUIRE(interrupts.ok());
        REQUIRE(VigenereBatchKernel::apply_host(buffers.value(), interrupts.value()).ok());

        std::vector<std::string> ids;
        ids.reserve(key_list.size());
        for (std::size_t i = 0; i < key_list.size(); ++i) {
            ids.push_back(VigenereExplicitKeyCandidateGenerator::make_candidate_id(key_list[i], i));
        }

        StatusOr<BatchResult> result = CudaBatchScore::score_and_top_k(
            buffers.value(), ids, "exact_match", /*k=*/key_list.size(), request);
        REQUIRE(result.ok());
        REQUIRE(count_perfect_scores(buffers.value().scores()) == 1);
        REQUIRE(require_rank1_exact_match(result.value()) == 1);
    }

    SECTION("vigenere_explicit_keys with interrupts") {
        const std::vector<Index29> correct_key = {I(1), I(2)};
        StatusOr<InterruptPolicy> policy = InterruptPolicy::from_skip_indices({1, 3});
        REQUIRE(policy.ok());

        StatusOr<std::vector<Index29>> cipher =
            VigenereKeyTransform{}.apply(plain, nlohmann::json{{"key_indices", {1, 2}}},
                                         TransformDirection::Encrypt, policy.value());
        REQUIRE(cipher.ok());

        const std::vector<std::vector<Index29>> key_list = {
            {I(7), I(8)},
            correct_key,
            {I(1), I(3)},
        };

        CandidateBatchBuffers::AllocateOptions options;
        options.family = CudaFamilyId::VigenereKey;
        options.with_scores = true;
        options.key_arena_capacity = CandidateBatchBuffers::key_arena_bytes_needed(key_list);

        StatusOr<CandidateBatchBuffers> buffers =
            CandidateBatchBuffers::allocate(key_list.size(), cipher.value().size(), options);
        REQUIRE(buffers.ok());
        REQUIRE(buffers.value().set_shared_tokens(cipher.value()).ok());
        REQUIRE(buffers.value().pack_explicit_keys(key_list).ok());

        StatusOr<InterruptDeviceView> interrupts =
            InterruptDeviceView::from_policy(policy.value(), cipher.value().size());
        REQUIRE(interrupts.ok());
        REQUIRE(VigenereBatchKernel::apply_host(buffers.value(), interrupts.value()).ok());

        std::vector<std::string> ids;
        for (std::size_t i = 0; i < key_list.size(); ++i) {
            ids.push_back(VigenereExplicitKeyCandidateGenerator::make_candidate_id(key_list[i], i));
        }

        StatusOr<BatchResult> result = CudaBatchScore::score_and_top_k(
            buffers.value(), ids, "exact_match", /*k=*/key_list.size(), request);
        REQUIRE(result.ok());
        REQUIRE(count_perfect_scores(buffers.value().scores()) == 1);
        REQUIRE(require_rank1_exact_match(result.value()) == 1);
    }
}

#else

TEST_CASE("CUDA batch rank-1 skipped (PARCAE_HAS_CUDA unset)", "[cuda][batch][rank1]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CUDA batch rank-1 recovery");
}

#endif
