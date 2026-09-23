#include <parcae/batch/batch_execution.hpp>
#include <parcae/batch/batch_ordering.hpp>
#include <parcae/batch/batch_runner.hpp>
#include <parcae/cli/console_progress_sink.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/score/exact_match.hpp>
#include <parcae/score/score_request.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
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

/// Thread-safe recording sink for BatchRunner progress tests.
class BatchProgressRecordingSink : public ConsoleProgressSink {
public:
    void on_progress(const ConsoleProgressSnapshot& snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++progress_count;
        last_done = snapshot.candidates_done();
        if (snapshot.candidates_total().has_value()) {
            last_total = snapshot.candidates_total().value();
        }
        if (snapshot.best_score().has_value()) {
            last_best = snapshot.best_score().value();
            last_best_label = snapshot.best_label();
        }
        max_done = std::max(max_done, snapshot.candidates_done());
    }

    void on_stage(
        std::string_view stage,
        const ConsoleProgressSnapshot& snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stages.emplace_back(stage);
        stage_total = snapshot.candidates_total();
    }

    std::mutex mutex_;
    std::size_t progress_count = 0;
    std::size_t last_done = 0;
    std::size_t last_total = 0;
    std::size_t max_done = 0;
    double last_best = 0.0;
    std::string last_best_label;
    std::vector<std::string> stages;
    std::optional<std::size_t> stage_total;
};

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

TEST_CASE("BatchRunner serial progress sink sees monotonic done", "[batch][progress]") {
    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain,
        nlohmann::json{{"shift", 11}},
        TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<TransformCandidate>> candidates =
        CaesarCandidateGenerator::generate(cipher.value());
    REQUIRE(candidates.ok());

    ScoreRequest request;
    request.reference = std::span<const Index29>(plain);

    BatchProgressRecordingSink sink;
    BatchRunner::Progress progress;
    progress.sink = &sink;
    progress.rune_count = plain.size();

    StatusOr<BatchResult> with_sink = BatchRunner::run(
        candidates.value(),
        "exact_match",
        3,
        request,
        BatchExecution::Serial,
        "v0",
        nlohmann::json::object(),
        progress);
    REQUIRE(with_sink.ok());

    StatusOr<BatchResult> without = BatchRunner::run(
        candidates.value(),
        "exact_match",
        3,
        request,
        BatchExecution::Serial);
    REQUIRE(without.ok());

    REQUIRE(with_sink.value().top().size() == without.value().top().size());
    for (std::size_t i = 0; i < without.value().top().size(); ++i) {
        REQUIRE(
            with_sink.value().top()[i].candidate_id() ==
            without.value().top()[i].candidate_id());
        REQUIRE(with_sink.value().top()[i].score() == without.value().top()[i].score());
        REQUIRE(
            with_sink.value().top()[i].source_index() ==
            without.value().top()[i].source_index());
    }

    REQUIRE(sink.stages.size() == 1);
    REQUIRE(sink.stages[0] == "score");
    REQUIRE(sink.stage_total.has_value());
    REQUIRE(sink.stage_total.value() == 29);
    REQUIRE(sink.progress_count == 29);
    REQUIRE(sink.max_done == 29);
    REQUIRE(sink.last_done == 29);
    REQUIRE(sink.last_total == 29);
    REQUIRE(sink.last_best == 1.0);
    REQUIRE(sink.last_best_label == "caesar:shift=11");
}

TEST_CASE(
    "BatchRunner parallel progress sink matches serial scores",
    "[batch][progress][parallel]") {
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

    BatchProgressRecordingSink sink;
    BatchRunner::Progress progress;
    progress.sink = &sink;
    progress.rune_count = plain.size();

    StatusOr<BatchResult> parallel = BatchRunner::run(
        candidates.value(),
        "exact_match",
        5,
        request,
        BatchExecution::Parallel,
        "v0",
        nlohmann::json::object(),
        progress);
    StatusOr<BatchResult> serial = BatchRunner::run(
        candidates.value(),
        "exact_match",
        5,
        request,
        BatchExecution::Serial);
    REQUIRE(parallel.ok());
    REQUIRE(serial.ok());
    REQUIRE(parallel.value().top().size() == serial.value().top().size());
    for (std::size_t i = 0; i < serial.value().top().size(); ++i) {
        REQUIRE(
            parallel.value().top()[i].candidate_id() == serial.value().top()[i].candidate_id());
        REQUIRE(parallel.value().top()[i].score() == serial.value().top()[i].score());
    }

    REQUIRE(sink.stages.size() == 1);
    REQUIRE(sink.progress_count == 29);
    REQUIRE(sink.max_done == 29);
    REQUIRE(sink.last_best == 1.0);
}
