#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <parcae/batch/batch_execution.hpp>
#include <parcae/batch/batch_hit.hpp>
#include <parcae/batch/batch_ordering.hpp>
#include <parcae/batch/batch_result.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/score/score_order.hpp>
#include <parcae/score/score_request.hpp>
#include <parcae/search/workspace_cipher.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/generate_candidates.hpp>
#include <parcae/tool/rank_candidates.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

#if defined(PARCAE_HAS_CUDA)
#include "cuda_score.hpp"
#include "parcae_cuda.hpp"
#endif

namespace {

[[nodiscard]] Context test_ctx() {
    return Context{std::string(PARCAE_TEST_DATA_DIR)};
}

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] TransformCandidate make_candidate(std::string id, std::vector<Index29> indices,
                                                nlohmann::json params = nlohmann::json::object()) {
    return TransformCandidate{
        std::move(id),     TransformId::identity(), TransformDirection::Decrypt,
        std::move(params), std::move(indices),
    };
}

void require_best_first(const BatchResult& result, ScoreOrder order) {
    const auto& top = result.top();
    for (std::size_t i = 1; i < top.size(); ++i) {
        REQUIRE(BatchOrdering::better(top[i - 1], top[i], order));
        REQUIRE_FALSE(BatchOrdering::better(top[i], top[i - 1], order));
    }
}

void require_same_ranking(const BatchResult& cpu, const BatchResult& cuda) {
    REQUIRE(cpu.scored_count() == cuda.scored_count());
    REQUIRE(cpu.top().size() == cuda.top().size());
    REQUIRE(cpu.score_id() == cuda.score_id());
    for (std::size_t i = 0; i < cpu.top().size(); ++i) {
        REQUIRE(cpu.top()[i].candidate_id() == cuda.top()[i].candidate_id());
        REQUIRE(cpu.top()[i].score() == cuda.top()[i].score());
        REQUIRE(cpu.top()[i].source_index() == cuda.top()[i].source_index());
    }
}

[[nodiscard]] StatusOr<BatchResult> rank_backend(std::span<const TransformCandidate> candidates,
                                                 std::string_view score_id, std::size_t k,
                                                 const Context& ctx, ScoreRequest request,
                                                 Backend backend) {
    return RankCandidates::run(candidates, score_id, k, &ctx, request, nlohmann::json::object(),
                               "v0", BatchExecution::Serial, backend);
}

} // namespace

TEST_CASE("RankCandidates CPU ordering contract: Asc χ² and Desc exact_match ties",
          "[tool][rank][order]") {
    const auto ctx = test_ctx();

    // Three identical plaintexts → identical Ic / χ²; order must be by candidate_id.
    const std::vector<Index29> plain = {I(0), I(0), I(0), I(0)};
    std::vector<TransformCandidate> tied = {
        make_candidate("z-last", plain),
        make_candidate("a-first", plain),
        make_candidate("m-mid", plain),
    };

    StatusOr<BatchResult> chi2 = RankCandidates::run(tied, "chi2_english_gp_v0", /*k=*/3, &ctx);
    REQUIRE(chi2.ok());
    REQUIRE(chi2.value().top().size() == 3);
    require_best_first(chi2.value(), ScoreOrder::Asc);
    REQUIRE(chi2.value().top()[0].candidate_id() == "a-first");
    REQUIRE(chi2.value().top()[1].candidate_id() == "m-mid");
    REQUIRE(chi2.value().top()[2].candidate_id() == "z-last");
    REQUIRE(chi2.value().top()[0].score() == chi2.value().top()[1].score());
    REQUIRE(chi2.value().top()[1].score() == chi2.value().top()[2].score());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);
    StatusOr<BatchResult> exact = RankCandidates::run(tied, "exact_match", /*k=*/3, &ctx, request);
    REQUIRE(exact.ok());
    require_best_first(exact.value(), ScoreOrder::Desc);
    REQUIRE(exact.value().top()[0].candidate_id() == "a-first");
    REQUIRE(exact.value().top()[0].score() == 1.0);
}

TEST_CASE("RankCandidates CPU Caesar grid ranking is BatchOrdering-stable", "[tool][rank][order]") {
    const auto ctx = test_ctx();
    const std::vector<Index29> plain = {I(1), I(2), I(3), I(4), I(5), I(10), I(14)};
    StatusOr<std::vector<Index29>> cipher =
        CaesarTransform{}.apply(plain, nlohmann::json{{"shift", 11}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> candidates =
        GenerateCandidates::from_indices("gen_caesar", cipher.value());
    REQUIRE(candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

    StatusOr<BatchResult> exact =
        RankCandidates::run(candidates.value(), "exact_match", /*k=*/5, &ctx, request);
    REQUIRE(exact.ok());
    require_best_first(exact.value(), ScoreOrder::Desc);
    REQUIRE(exact.value().top()[0].candidate_id() == "caesar:shift=11");
    REQUIRE(exact.value().top()[0].score() == 1.0);

    StatusOr<BatchResult> chi2 =
        RankCandidates::run(candidates.value(), "chi2_english_gp_v0", /*k=*/8, &ctx);
    REQUIRE(chi2.ok());
    require_best_first(chi2.value(), ScoreOrder::Asc);

    StatusOr<BatchResult> ic = RankCandidates::run(candidates.value(), "ic_mod29", /*k=*/5, &ctx);
    REQUIRE(ic.ok());
    require_best_first(ic.value(), ScoreOrder::Desc);
}

#if defined(PARCAE_HAS_CUDA)

TEST_CASE("RankCandidates CUDA/CPU top-k ordering contract (scores + ties)",
          "[tool][rank][order][cuda]") {
    if (!ParcaeCuda::available() || !CudaScore::available()) {
        SKIP("No CUDA device");
    }

    const auto ctx = test_ctx();

    {
        // Equal-score tie break must match CPU (candidate_id lexicographic).
        const std::vector<Index29> plain = {I(0), I(0), I(0), I(0)};
        std::vector<TransformCandidate> tied = {
            make_candidate("z-last", plain),
            make_candidate("a-first", plain),
            make_candidate("m-mid", plain),
        };
        StatusOr<BatchResult> cpu =
            rank_backend(tied, "chi2_english_gp_v0", 3, ctx, {}, Backend::Cpu);
        StatusOr<BatchResult> cuda =
            rank_backend(tied, "chi2_english_gp_v0", 3, ctx, {}, Backend::Cuda);
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        require_same_ranking(cpu.value(), cuda.value());
        require_best_first(cuda.value(), ScoreOrder::Asc);
    }

    {
        const std::vector<Index29> plain = {I(1), I(2), I(3), I(4), I(5), I(10), I(14)};
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain, nlohmann::json{{"shift", 11}}, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());
        StatusOr<std::vector<TransformCandidate>> candidates =
            GenerateCandidates::from_indices("gen_caesar", cipher.value());
        REQUIRE(candidates.ok());

        ScoreRequest request;
        request.reference = std::span<const Index29>(plain);

        for (const char* score_id : {"exact_match", "ic_mod29", "chi2_english_gp_v0"}) {
            ScoreRequest req =
                (std::string_view(score_id) == "exact_match") ? request : ScoreRequest{};
            constexpr std::size_t k = 7;
            StatusOr<BatchResult> cpu =
                rank_backend(candidates.value(), score_id, k, ctx, req, Backend::Cpu);
            StatusOr<BatchResult> cuda =
                rank_backend(candidates.value(), score_id, k, ctx, req, Backend::Cuda);
            REQUIRE(cpu.ok());
            REQUIRE(cuda.ok());
            require_same_ranking(cpu.value(), cuda.value());
        }
    }

    {
        // Locked Tier-A fixture: generate pool then rank χ² — ids/scores must match.
        StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::from_fixture(
            std::string(PARCAE_TEST_DATA_DIR), "_example", "a-warning");
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> atbash =
            GenerateCandidates::from_indices("gen_atbash", cipher.value().indices());
        StatusOr<std::vector<TransformCandidate>> caesar =
            GenerateCandidates::from_indices("gen_caesar", cipher.value().indices());
        REQUIRE(atbash.ok());
        REQUIRE(caesar.ok());

        std::vector<TransformCandidate> pool;
        pool.reserve(1 + caesar.value().size());
        pool.push_back(atbash.value()[0]);
        for (const TransformCandidate& c : caesar.value()) {
            pool.push_back(c);
        }

        StatusOr<BatchResult> cpu =
            rank_backend(pool, "chi2_english_gp_v0", /*k=*/5, ctx, {}, Backend::Cpu);
        StatusOr<BatchResult> cuda =
            rank_backend(pool, "chi2_english_gp_v0", /*k=*/5, ctx, {}, Backend::Cuda);
        REQUIRE(cpu.ok());
        REQUIRE(cuda.ok());
        require_same_ranking(cpu.value(), cuda.value());
        REQUIRE(cpu.value().top()[0].candidate_id() == "atbash");
        require_best_first(cpu.value(), ScoreOrder::Asc);
    }
}

#else

TEST_CASE("RankCandidates CUDA/CPU ordering contract skipped (PARCAE_HAS_CUDA unset)",
          "[tool][rank][order][cuda]") {
    SUCCEED("PARCAE_HAS_CUDA unset — CPU ordering covered above; CUDA compare not linked");
}

#endif
