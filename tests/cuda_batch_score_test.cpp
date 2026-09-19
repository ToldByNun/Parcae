#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "caesar_batch_kernel.hpp"
#include "candidate_batch_buffers.hpp"
#include "cuda_batch_score.hpp"
#include "parcae_cuda.hpp"
#include "params.hpp"

#include "parcae/batch/batch_execution.hpp"
#include "parcae/batch/batch_runner.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] std::vector<std::string> caesar_candidate_ids() {
    std::vector<std::string> ids;
    ids.reserve(Index29::modulus);
    for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
        ids.push_back(CaesarCandidateGenerator::make_candidate_id(shift));
    }
    return ids;
}

}  // namespace

TEST_CASE(
    "CUDA batch score + top-k matches BatchRunner on caesar sweep",
    "[cuda][batch][score]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(CudaBatchScore::available());

    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
    constexpr int kShift = 11;

    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain,
        nlohmann::json{{"shift", kShift}},
        TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> cpu_candidates =
        CaesarCandidateGenerator::generate(cipher.value());
    REQUIRE(cpu_candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

    StatusOr<BatchResult> cpu = BatchRunner::run(
        cpu_candidates.value(),
        "exact_match",
        /*k=*/3,
        request,
        BatchExecution::Serial);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().top().size() == 3);
    REQUIRE(cpu.value().top()[0].source_index() == static_cast<std::size_t>(kShift));
    REQUIRE(cpu.value().top()[0].score() == 1.0);

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = true;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, cipher.value().size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(cipher.value()).ok());
    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    REQUIRE(CaesarBatchKernel::apply_host(buffers.value()).ok());

    const std::vector<std::string> ids = caesar_candidate_ids();
    StatusOr<BatchResult> cuda = CudaBatchScore::score_and_top_k(
        buffers.value(), ids, "exact_match", /*k=*/3, request);
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value().scored_count() == Index29::modulus);
    REQUIRE(cuda.value().top().size() == 3);

    REQUIRE(cuda.value().top()[0].candidate_id() == cpu.value().top()[0].candidate_id());
    REQUIRE(cuda.value().top()[0].score() == cpu.value().top()[0].score());
    REQUIRE(cuda.value().top()[0].source_index() == cpu.value().top()[0].source_index());
    for (std::size_t i = 0; i < 3; ++i) {
        REQUIRE(cuda.value().top()[i].candidate_id() == cpu.value().top()[i].candidate_id());
        REQUIRE(cuda.value().top()[i].score() == cpu.value().top()[i].score());
        REQUIRE(cuda.value().top()[i].source_index() == cpu.value().top()[i].source_index());
    }
}

TEST_CASE("CUDA batch score_lanes fills scores for ic_mod29", "[cuda][batch][score]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> stream = {I(0), I(0), I(1), I(1), I(2)};

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = true;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, stream.size(), options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().set_shared_tokens(stream).ok());
    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    REQUIRE(CaesarBatchKernel::apply_host(buffers.value()).ok());
    REQUIRE(CudaBatchScore::score_lanes(buffers.value(), "ic_mod29").ok());

    for (std::size_t c = 0; c < Index29::modulus; ++c) {
        std::vector<Index29> lane(stream.size());
        REQUIRE(buffers.value().copy_out_candidate(c, lane).ok());
        StatusOr<double> expected = CudaScore::score("ic_mod29", lane);
        REQUIRE(expected.ok());
        REQUIRE(buffers.value().scores()[c] == expected.value());
    }
}

TEST_CASE("CUDA batch top-k rejects k=0 and size mismatch", "[cuda][batch][score]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::string> ids = {"a", "b"};
    const std::vector<double> scores = {1.0, 0.5};
    REQUIRE_FALSE(CudaBatchScore::top_k(ids, scores, "exact_match", 0).ok());

    const std::vector<std::string> short_ids = {"a"};
    REQUIRE_FALSE(CudaBatchScore::top_k(short_ids, scores, "exact_match", 1).ok());

    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = false;
    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(2, 3, options);
    REQUIRE(buffers.ok());
    REQUIRE_FALSE(CudaBatchScore::score_lanes(buffers.value(), "ic_mod29").ok());
}

#else

TEST_CASE("CUDA batch score skipped (PARCAE_HAS_CUDA unset)", "[cuda][batch][score]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CudaBatchScore");
}

#endif
