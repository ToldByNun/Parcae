#include <parcae/batch/batch_execution.hpp>
#include <parcae/batch/batch_ordering.hpp>
#include <parcae/batch/batch_runner.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/score/exact_match.hpp>
#include <parcae/score/score_request.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

[[nodiscard]] TransformCandidate make_candidate(
    std::string id,
    std::vector<Index29> indices) {
    return TransformCandidate(
        std::move(id),
        TransformId::identity(),
        TransformDirection::Decrypt,
        nlohmann::json::object(),
        std::move(indices));
}

}  // namespace

TEST_CASE("BatchOrdering tie-breaks by score then candidate_id then source_index", "[batch]") {
    const BatchHit a{"b", 1.0, 5};
    const BatchHit b{"a", 1.0, 9};
    const BatchHit c{"a", 1.0, 2};
    const BatchHit d{"z", 2.0, 0};

    REQUIRE(BatchOrdering::better(d, a, ScoreOrder::Desc));   // higher score
    REQUIRE(BatchOrdering::better(b, a, ScoreOrder::Desc));   // same score, id "a" < "b"
    REQUIRE(BatchOrdering::better(c, b, ScoreOrder::Desc));   // same score+id, lower index
    REQUIRE(BatchOrdering::better(a, d, ScoreOrder::Asc));    // lower score wins for Asc
}

TEST_CASE("BatchRunner rejects k=0 and unknown score", "[batch]") {
    const std::vector<TransformCandidate> candidates = {
        make_candidate("x", {I(0), I(1)}),
    };
    REQUIRE_FALSE(BatchRunner::run(candidates, "ic_mod29", 0).ok());
    REQUIRE_FALSE(BatchRunner::run(candidates, "nope", 1).ok());
}

TEST_CASE("BatchRunner serial top-k with exact_match on caesar sweep", "[batch]") {
    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
    constexpr int kShift = 11;

    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain,
        nlohmann::json{{"shift", kShift}},
        TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> candidates =
        CaesarCandidateGenerator::generate(cipher.value());
    REQUIRE(candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

    StatusOr<BatchResult> result = BatchRunner::run(
        candidates.value(),
        "exact_match",
        /*k=*/3,
        request,
        BatchExecution::Serial);
    REQUIRE(result.ok());
    REQUIRE(result.value().scored_count() == 29);
    REQUIRE(result.value().top().size() == 3);
    REQUIRE(result.value().top()[0].score() == 1.0);
    REQUIRE(result.value().top()[0].candidate_id() == "caesar:shift=11");
    REQUIRE(result.value().top()[0].source_index() == static_cast<std::size_t>(kShift));
    // Remaining caesar shifts score 0; tie-break by candidate_id ascending.
    REQUIRE(result.value().top()[1].score() == 0.0);
    REQUIRE(result.value().top()[2].score() == 0.0);
    REQUIRE(result.value().top()[1].candidate_id() < result.value().top()[2].candidate_id());
}

TEST_CASE("BatchRunner parallel matches serial under BatchOrdering", "[batch][parallel]") {
    const std::vector<Index29> plain = {I(3), I(5), I(7), I(9), I(11), I(13), I(15)};
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain,
        nlohmann::json{{"shift", 4}},
        TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> candidates =
        CaesarCandidateGenerator::generate(cipher.value());
    REQUIRE(candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

    StatusOr<BatchResult> serial = BatchRunner::run(
        candidates.value(),
        "exact_match",
        5,
        request,
        BatchExecution::Serial);
    StatusOr<BatchResult> parallel = BatchRunner::run(
        candidates.value(),
        "exact_match",
        5,
        request,
        BatchExecution::Parallel);
    REQUIRE(serial.ok());
    REQUIRE(parallel.ok());
    REQUIRE(serial.value().top().size() == parallel.value().top().size());
    for (std::size_t i = 0; i < serial.value().top().size(); ++i) {
        REQUIRE(serial.value().top()[i].candidate_id() == parallel.value().top()[i].candidate_id());
        REQUIRE(serial.value().top()[i].score() == parallel.value().top()[i].score());
        REQUIRE(serial.value().top()[i].source_index() == parallel.value().top()[i].source_index());
    }
}

TEST_CASE("BatchRunner ic_mod29 desc keeps higher IC first", "[batch]") {
    // All-identical stream has IC=1; alternating has lower IC.
    const std::vector<TransformCandidate> candidates = {
        make_candidate("low", {I(0), I(1), I(0), I(1), I(0), I(1)}),
        make_candidate("high", {I(3), I(3), I(3), I(3), I(3), I(3)}),
        make_candidate("mid", {I(3), I(3), I(3), I(0), I(1), I(2)}),
    };

    StatusOr<BatchResult> result =
        BatchRunner::run(candidates, "ic_mod29", 2, {}, BatchExecution::Serial);
    REQUIRE(result.ok());
    REQUIRE(result.value().top().size() == 2);
    REQUIRE(result.value().top()[0].candidate_id() == "high");
    REQUIRE(result.value().top()[0].score() > result.value().top()[1].score());
}

TEST_CASE("BatchExecutionUtil stringifies modes", "[batch]") {
    REQUIRE(BatchExecutionUtil::to_string(BatchExecution::Serial) == "serial");
    REQUIRE(BatchExecutionUtil::to_string(BatchExecution::Parallel) == "parallel");
}
