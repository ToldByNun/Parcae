#include <parcae/batch/batch_ordering.hpp>
#include <parcae/batch/batch_runner.hpp>
#include <parcae/cli/console_progress_sink.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/generate/affine_candidate_generator.hpp>
#include <parcae/generate/atbash_candidate_generator.hpp>
#include <parcae/generate/atbash_caesar_candidate_generator.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/generate/vigenere_explicit_key_candidate_generator.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/score_order.hpp>
#include <parcae/score/score_registry.hpp>
#include <parcae/score/score_request.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/affine_transform.hpp>
#include <parcae/transform/atbash_transform.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/vigenere_key_transform.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::vector<Index29> synthetic_cipher() {
    std::vector<Index29> plain;
    plain.reserve(64);
    for (std::uint8_t i = 0; i < 64; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>(i % 29)});
    }
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain, nlohmann::json{{"shift", 7}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());
    return cipher.value();
}

[[nodiscard]] std::vector<double> cpu_chi2_by_shift(
    const std::vector<Index29>& cipher,
    const ExpectedFrequencyTable& freqs) {
    ScoreRequest request;
    request.expected_frequencies = &freqs;
    std::vector<double> scores(Index29::modulus, 0.0);
    for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
        StatusOr<std::vector<Index29>> out = CaesarTransform{}.apply(
            cipher,
            nlohmann::json{{"shift", static_cast<int>(shift)}},
            TransformDirection::Decrypt);
        REQUIRE(out.ok());
        StatusOr<double> score = ScoreRegistry::score(
            "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(score.ok());
        scores[shift] = score.value();
    }
    return scores;
}

}  // namespace

TEST_CASE(
    "GpuCandidateExport caesar_from_host_scores top-k matches BatchOrdering",
    "[search][export][caesar]") {
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = synthetic_cipher();
    const std::vector<double> scores = cpu_chi2_by_shift(cipher, freqs.value());

    constexpr std::size_t k = 5;
    StatusOr<GpuCandidateExport::Result> exported =
        GpuCandidateExport::caesar_from_host_scores(cipher, scores, k);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().size() == k);
    REQUIRE(exported.value().backend() == Backend::Cpu);

    // Best-first under Asc χ²: scores must be non-decreasing.
    for (std::size_t i = 1; i < exported.value().size(); ++i) {
        const auto& a = exported.value().rows()[i - 1];
        const auto& b = exported.value().rows()[i];
        REQUIRE(BatchOrdering::better(
            BatchHit{a.candidate().candidate_id(), a.score(), a.source_index()},
            BatchHit{b.candidate().candidate_id(), b.score(), b.source_index()},
            ScoreOrder::Asc));
        REQUIRE(a.rank() + 1 == b.rank());
    }

    // IDs match generator convention; plaintext matches Caesar apply for that shift.
    for (const GpuCandidateExport::Row& row : exported.value().rows()) {
        const std::uint8_t shift = static_cast<std::uint8_t>(row.source_index());
        REQUIRE(
            row.candidate().candidate_id() ==
            CaesarCandidateGenerator::make_candidate_id(shift));
        StatusOr<std::vector<Index29>> expected = CaesarTransform{}.apply(
            cipher,
            nlohmann::json{{"shift", static_cast<int>(shift)}},
            TransformDirection::Decrypt);
        REQUIRE(expected.ok());
        REQUIRE(row.candidate().output_indices() == expected.value());
        REQUIRE(row.score() == scores[shift]);
    }

    const std::vector<nlohmann::json> wires = exported.value().to_wire_lines();
    REQUIRE(wires.size() == k);
    REQUIRE(wires[0].at("rank").get<int>() == 0);
    REQUIRE(wires[0].at("score").at("score_id").get<std::string>() == "chi2_english_gp_v0");
    REQUIRE(wires[0].at("score").at("backend").get<std::string>() == "cpu");
}

TEST_CASE(
    "GpuCandidateExport caesar_from_host_scores rejects bad bounds",
    "[search][export][caesar]") {
    const std::vector<Index29> cipher = {Index29{1}, Index29{2}};
    std::vector<double> scores(29, 1.0);
    REQUIRE_FALSE(GpuCandidateExport::caesar_from_host_scores(cipher, scores, 0).ok());
    REQUIRE_FALSE(GpuCandidateExport::caesar_from_host_scores({}, scores, 1).ok());
    scores.resize(10);
    REQUIRE_FALSE(GpuCandidateExport::caesar_from_host_scores(cipher, scores, 1).ok());
}

TEST_CASE(
    "GpuCandidateExport::caesar without CUDA build fails loud",
    "[search][export][caesar]") {
#if !defined(PARCAE_HAS_CUDA)
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();
    StatusOr<GpuCandidateExport::Result> exported =
        GpuCandidateExport::caesar(cipher, freqs.value(), 3);
    REQUIRE_FALSE(exported.ok());
    REQUIRE(exported.status().message().find("CUDA") != std::string::npos);
#else
    SUCCEED("CUDA build — loud fail covered by runtime path when no device");
#endif
}

TEST_CASE(
    "GpuCandidateExport atbash_from_host_scores single lane",
    "[search][export][atbash]") {
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();

    StatusOr<std::vector<Index29>> plain =
        AtbashTransform{}.apply(cipher, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(plain.ok());
    ScoreRequest request;
    request.expected_frequencies = &freqs.value();
    StatusOr<double> score = ScoreRegistry::score(
        "chi2_english_gp_v0", plain.value(), "v0", nlohmann::json::object(), request);
    REQUIRE(score.ok());

    const std::vector<double> scores{score.value()};
    StatusOr<GpuCandidateExport::Result> exported =
        GpuCandidateExport::atbash_from_host_scores(cipher, scores, 1);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().size() == 1);
    REQUIRE(
        exported.value().rows()[0].candidate().candidate_id() ==
        AtbashCandidateGenerator::make_candidate_id());
    REQUIRE(exported.value().rows()[0].candidate().output_indices() == plain.value());
}

TEST_CASE(
    "GpuCandidateExport atbash_caesar_from_host_scores top-k",
    "[search][export][atbash_caesar]") {
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();

    ScoreRequest request;
    request.expected_frequencies = &freqs.value();
    std::vector<double> scores(Index29::modulus, 0.0);
    for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
        StatusOr<std::vector<Index29>> out =
            ComposeTransform::apply_atbash_then_caesar(cipher, shift, TransformDirection::Decrypt);
        REQUIRE(out.ok());
        StatusOr<double> score = ScoreRegistry::score(
            "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(score.ok());
        scores[shift] = score.value();
    }

    constexpr std::size_t k = 4;
    StatusOr<GpuCandidateExport::Result> exported =
        GpuCandidateExport::atbash_caesar_from_host_scores(cipher, scores, k);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().size() == k);
    for (std::size_t i = 1; i < exported.value().size(); ++i) {
        REQUIRE(exported.value().rows()[i - 1].score() <= exported.value().rows()[i].score());
    }
    const auto& best = exported.value().rows()[0];
    const std::uint8_t shift = static_cast<std::uint8_t>(best.source_index());
    REQUIRE(
        best.candidate().candidate_id() ==
        AtbashCaesarCandidateGenerator::make_candidate_id(shift));
    StatusOr<std::vector<Index29>> expected =
        ComposeTransform::apply_atbash_then_caesar(cipher, shift, TransformDirection::Decrypt);
    REQUIRE(expected.ok());
    REQUIRE(best.candidate().output_indices() == expected.value());
}

TEST_CASE(
    "GpuCandidateExport affine_from_host_scores top-k mapping a,b",
    "[search][export][affine]") {
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();

    // Sparse scoring: only fill a few lanes; rest stay high (worse for Asc χ²).
    std::vector<double> scores(AffineCandidateGenerator::candidate_count, 1.0e9);
    ScoreRequest request;
    request.expected_frequencies = &freqs.value();
    const std::pair<std::uint8_t, std::uint8_t> samples[] = {
        {std::uint8_t{1}, std::uint8_t{0}},
        {std::uint8_t{2}, std::uint8_t{5}},
        {std::uint8_t{28}, std::uint8_t{28}},
        {std::uint8_t{7}, std::uint8_t{3}},
    };
    for (const auto& [a, b] : samples) {
        const std::size_t index =
            static_cast<std::size_t>(a - 1) * Index29::modulus + static_cast<std::size_t>(b);
        StatusOr<std::vector<Index29>> out = AffineTransform{}.apply(
            cipher,
            nlohmann::json{{"a", static_cast<int>(a)}, {"b", static_cast<int>(b)}},
            TransformDirection::Decrypt);
        REQUIRE(out.ok());
        StatusOr<double> score = ScoreRegistry::score(
            "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(score.ok());
        scores[index] = score.value();
    }

    constexpr std::size_t k = 3;
    StatusOr<GpuCandidateExport::Result> exported =
        GpuCandidateExport::affine_from_host_scores(cipher, scores, k);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().size() == k);

    for (const GpuCandidateExport::Row& row : exported.value().rows()) {
        const std::size_t index = row.source_index();
        const std::uint8_t a = static_cast<std::uint8_t>(index / Index29::modulus + 1);
        const std::uint8_t b = static_cast<std::uint8_t>(index % Index29::modulus);
        REQUIRE(
            row.candidate().candidate_id() ==
            AffineCandidateGenerator::make_candidate_id(a, b));
        REQUIRE(row.candidate().params().at("a").get<int>() == static_cast<int>(a));
        REQUIRE(row.candidate().params().at("b").get<int>() == static_cast<int>(b));
    }
}

TEST_CASE(
    "GpuCandidateExport vigenere_from_host_scores explicit keys",
    "[search][export][vigenere]") {
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();

    const std::vector<std::vector<Index29>> keys = {
        {Index29{1}, Index29{2}, Index29{3}},
        {Index29{5}},
        {Index29{7}, Index29{8}},
        {Index29{10}, Index29{11}, Index29{12}, Index29{13}},
    };

    ScoreRequest request;
    request.expected_frequencies = &freqs.value();
    std::vector<double> scores;
    scores.reserve(keys.size());
    const VigenereKeyTransform transform;
    for (const auto& key : keys) {
        nlohmann::json params{{"key_indices", nlohmann::json::array()}};
        for (const Index29 idx : key) {
            params["key_indices"].push_back(static_cast<int>(idx.value()));
        }
        StatusOr<std::vector<Index29>> plain =
            transform.apply(cipher, params, TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> score = ScoreRegistry::score(
            "chi2_english_gp_v0", plain.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(score.ok());
        scores.push_back(score.value());
    }

    constexpr std::size_t k = 2;
    StatusOr<GpuCandidateExport::Result> exported =
        GpuCandidateExport::vigenere_from_host_scores(cipher, keys, scores, k);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().size() == k);
    for (std::size_t i = 1; i < exported.value().size(); ++i) {
        REQUIRE(exported.value().rows()[i - 1].score() <= exported.value().rows()[i].score());
    }

    const auto& best = exported.value().rows()[0];
    const std::size_t idx = best.source_index();
    REQUIRE(
        best.candidate().candidate_id() ==
        VigenereExplicitKeyCandidateGenerator::make_candidate_id(keys[idx], idx));
    REQUIRE(best.candidate().params().at("key_indices").size() == keys[idx].size());
}

TEST_CASE(
    "GpuCandidateExport default_bounded_key_grid is L=1..N synthetic",
    "[search][export][vigenere]") {
    StatusOr<std::vector<std::vector<Index29>>> keys =
        GpuCandidateExport::default_bounded_key_grid(5);
    REQUIRE(keys.ok());
    REQUIRE(keys.value().size() == 5);
    REQUIRE(keys.value()[0].size() == 1);
    REQUIRE(keys.value()[0][0].value() == 1);
    REQUIRE(keys.value()[4].size() == 5);
    REQUIRE(keys.value()[4][0].value() == 1);
    REQUIRE(keys.value()[4][4].value() == 5);
    REQUIRE_FALSE(GpuCandidateExport::default_bounded_key_grid(0).ok());
}

TEST_CASE(
    "GpuCandidateExport vigenere rejects empty keys",
    "[search][export][vigenere]") {
    const std::vector<Index29> cipher = {Index29{1}, Index29{2}};
    std::vector<std::vector<Index29>> keys;
    std::vector<double> scores;
    REQUIRE_FALSE(
        GpuCandidateExport::vigenere_from_host_scores(cipher, keys, scores, 1).ok());
    keys.push_back({});
    scores.push_back(1.0);
    REQUIRE_FALSE(
        GpuCandidateExport::vigenere_from_host_scores(cipher, keys, scores, 1).ok());
}

TEST_CASE(
    "GpuCandidateExport fused families without CUDA fail loud",
    "[search][export]") {
#if !defined(PARCAE_HAS_CUDA)
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();
    REQUIRE_FALSE(GpuCandidateExport::atbash(cipher, freqs.value(), 1).ok());
    REQUIRE_FALSE(GpuCandidateExport::atbash_caesar(cipher, freqs.value(), 3).ok());
    REQUIRE_FALSE(GpuCandidateExport::affine(cipher, freqs.value(), 3).ok());
    const std::vector<std::vector<Index29>> keys = {{Index29{1}, Index29{2}}};
    REQUIRE_FALSE(GpuCandidateExport::vigenere(cipher, freqs.value(), keys, 1).ok());
    REQUIRE_FALSE(GpuCandidateExport::vigenere_bounded(cipher, freqs.value(), 2, 4).ok());
#else
    SUCCEED("CUDA build — device path tested separately");
#endif
}

namespace {

class GpuExportProgressRecordingSink : public ConsoleProgressSink {
public:
    void on_progress(const ConsoleProgressSnapshot& /*snapshot*/) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++progress_count;
    }

    void on_stage(
        std::string_view stage,
        const ConsoleProgressSnapshot& snapshot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stages.emplace_back(stage);
        if (snapshot.candidates_total().has_value()) {
            last_stage_total = snapshot.candidates_total().value();
        }
        last_rune_count = snapshot.rune_count();
    }

    std::mutex mutex_;
    std::size_t progress_count = 0;
    std::size_t last_stage_total = 0;
    std::size_t last_rune_count = 0;
    std::vector<std::string> stages;
};

}  // namespace

TEST_CASE(
    "GpuCandidateExport host-score path emits materialize; rows match without sink",
    "[search][export][progress]") {
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = synthetic_cipher();
    const std::vector<double> scores = cpu_chi2_by_shift(cipher, freqs.value());

    GpuExportProgressRecordingSink sink;
    BatchRunner::Progress progress;
    progress.sink = &sink;

    constexpr std::size_t k = 5;
    StatusOr<GpuCandidateExport::Result> with_sink =
        GpuCandidateExport::caesar_from_host_scores(
            cipher,
            scores,
            k,
            TransformDirection::Decrypt,
            Backend::Cpu,
            progress);
    REQUIRE(with_sink.ok());

    StatusOr<GpuCandidateExport::Result> without =
        GpuCandidateExport::caesar_from_host_scores(cipher, scores, k);
    REQUIRE(without.ok());

    REQUIRE(with_sink.value().size() == without.value().size());
    for (std::size_t i = 0; i < without.value().size(); ++i) {
        REQUIRE(
            with_sink.value().rows()[i].candidate().candidate_id() ==
            without.value().rows()[i].candidate().candidate_id());
        REQUIRE(with_sink.value().rows()[i].score() == without.value().rows()[i].score());
    }

    REQUIRE(sink.stages.size() == 1);
    REQUIRE(sink.stages[0] == "materialize");
    REQUIRE(sink.last_stage_total == Index29::modulus);
    REQUIRE(sink.last_rune_count == cipher.size());
    REQUIRE(sink.progress_count == 0);
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"

TEST_CASE(
    "GpuCandidateExport fused caesar emits fuse/d2h/materialize; rows match host",
    "[search][export][progress][cuda]") {
    if (!ParcaeCuda::available()) {
        SKIP("No CUDA device");
    }

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = synthetic_cipher();
    const std::vector<double> cpu_scores = cpu_chi2_by_shift(cipher, freqs.value());

    GpuExportProgressRecordingSink sink;
    BatchRunner::Progress progress;
    progress.sink = &sink;

    constexpr std::size_t k = 5;
    StatusOr<GpuCandidateExport::Result> gpu = GpuCandidateExport::caesar(
        cipher, freqs.value(), k, TransformDirection::Decrypt, progress);
    REQUIRE(gpu.ok());

    StatusOr<GpuCandidateExport::Result> host =
        GpuCandidateExport::caesar_from_host_scores(cipher, cpu_scores, k);
    REQUIRE(host.ok());

    REQUIRE(gpu.value().size() == host.value().size());
    for (std::size_t i = 0; i < host.value().size(); ++i) {
        REQUIRE(
            gpu.value().rows()[i].candidate().candidate_id() ==
            host.value().rows()[i].candidate().candidate_id());
        REQUIRE(gpu.value().rows()[i].score() == host.value().rows()[i].score());
    }

    REQUIRE(sink.stages.size() == 3);
    REQUIRE(sink.stages[0] == "fuse");
    REQUIRE(sink.stages[1] == "d2h");
    REQUIRE(sink.stages[2] == "materialize");
    REQUIRE(sink.last_stage_total == Index29::modulus);
    REQUIRE(sink.last_rune_count == cipher.size());
}

TEST_CASE(
    "GpuCandidateExport::caesar fused matches host-score materialization",
    "[search][export][caesar][cuda]") {
    if (!ParcaeCuda::available()) {
        SKIP("No CUDA device");
    }

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = synthetic_cipher();
    const std::vector<double> cpu_scores = cpu_chi2_by_shift(cipher, freqs.value());

    constexpr std::size_t k = 8;
    StatusOr<GpuCandidateExport::Result> gpu =
        GpuCandidateExport::caesar(cipher, freqs.value(), k);
    REQUIRE(gpu.ok());
    REQUIRE(gpu.value().backend() == Backend::Cuda);
    REQUIRE(gpu.value().size() == k);

    StatusOr<GpuCandidateExport::Result> host =
        GpuCandidateExport::caesar_from_host_scores(cipher, cpu_scores, k);
    REQUIRE(host.ok());

    REQUIRE(gpu.value().size() == host.value().size());
    for (std::size_t i = 0; i < gpu.value().size(); ++i) {
        REQUIRE(
            gpu.value().rows()[i].candidate().candidate_id() ==
            host.value().rows()[i].candidate().candidate_id());
        REQUIRE(gpu.value().rows()[i].score() == host.value().rows()[i].score());
        REQUIRE(
            gpu.value().rows()[i].candidate().output_indices() ==
            host.value().rows()[i].candidate().output_indices());
    }
}

TEST_CASE(
    "GpuCandidateExport fused atbash/atbash_caesar/affine match host scores",
    "[search][export][cuda]") {
    if (!ParcaeCuda::available()) {
        SKIP("No CUDA device");
    }

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();
    ScoreRequest request;
    request.expected_frequencies = &freqs.value();

    {
        StatusOr<std::vector<Index29>> plain = AtbashTransform{}.apply(
            cipher, nlohmann::json::object(), TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> score = ScoreRegistry::score(
            "chi2_english_gp_v0", plain.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(score.ok());
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::atbash(cipher, freqs.value(), 1);
        REQUIRE(gpu.ok());
        StatusOr<GpuCandidateExport::Result> host =
            GpuCandidateExport::atbash_from_host_scores(cipher, std::vector<double>{score.value()}, 1);
        REQUIRE(host.ok());
        REQUIRE(gpu.value().rows()[0].score() == host.value().rows()[0].score());
        REQUIRE(
            gpu.value().rows()[0].candidate().output_indices() ==
            host.value().rows()[0].candidate().output_indices());
    }

    {
        std::vector<double> scores(29, 0.0);
        for (std::uint8_t shift = 0; shift < 29; ++shift) {
            StatusOr<std::vector<Index29>> out = ComposeTransform::apply_atbash_then_caesar(
                cipher, shift, TransformDirection::Decrypt);
            REQUIRE(out.ok());
            StatusOr<double> score = ScoreRegistry::score(
                "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
            REQUIRE(score.ok());
            scores[shift] = score.value();
        }
        constexpr std::size_t k = 5;
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::atbash_caesar(cipher, freqs.value(), k);
        REQUIRE(gpu.ok());
        StatusOr<GpuCandidateExport::Result> host =
            GpuCandidateExport::atbash_caesar_from_host_scores(cipher, scores, k);
        REQUIRE(host.ok());
        for (std::size_t i = 0; i < k; ++i) {
            REQUIRE(
                gpu.value().rows()[i].candidate().candidate_id() ==
                host.value().rows()[i].candidate().candidate_id());
            REQUIRE(gpu.value().rows()[i].score() == host.value().rows()[i].score());
        }
    }

    {
        StatusOr<GpuCandidateExport::Result> gpu =
            GpuCandidateExport::affine(cipher, freqs.value(), 4);
        REQUIRE(gpu.ok());
        REQUIRE(gpu.value().size() == 4);
        // Recompute host scores only for returned lanes (full 812 CPU grid is slow in CI).
        for (const GpuCandidateExport::Row& row : gpu.value().rows()) {
            StatusOr<std::vector<Index29>> out = AffineTransform{}.apply(
                cipher, row.candidate().params(), TransformDirection::Decrypt);
            REQUIRE(out.ok());
            StatusOr<double> score = ScoreRegistry::score(
                "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
            REQUIRE(score.ok());
            REQUIRE(row.score() == score.value());
            REQUIRE(row.candidate().output_indices() == out.value());
        }
    }
}

TEST_CASE(
    "GpuCandidateExport fused vigenere explicit + bounded match host",
    "[search][export][vigenere][cuda]") {
    if (!ParcaeCuda::available()) {
        SKIP("No CUDA device");
    }

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = synthetic_cipher();
    ScoreRequest request;
    request.expected_frequencies = &freqs.value();

    const std::vector<std::vector<Index29>> keys = {
        {Index29{1}, Index29{2}, Index29{3}},
        {Index29{4}, Index29{5}},
        {Index29{9}},
    };
    std::vector<double> scores;
    const VigenereKeyTransform transform;
    for (const auto& key : keys) {
        nlohmann::json params{{"key_indices", nlohmann::json::array()}};
        for (const Index29 idx : key) {
            params["key_indices"].push_back(static_cast<int>(idx.value()));
        }
        StatusOr<std::vector<Index29>> plain =
            transform.apply(cipher, params, TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> score = ScoreRegistry::score(
            "chi2_english_gp_v0", plain.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(score.ok());
        scores.push_back(score.value());
    }

    StatusOr<GpuCandidateExport::Result> gpu =
        GpuCandidateExport::vigenere(cipher, freqs.value(), keys, 2);
    REQUIRE(gpu.ok());
    StatusOr<GpuCandidateExport::Result> host =
        GpuCandidateExport::vigenere_from_host_scores(cipher, keys, scores, 2);
    REQUIRE(host.ok());
    for (std::size_t i = 0; i < 2; ++i) {
        REQUIRE(
            gpu.value().rows()[i].candidate().candidate_id() ==
            host.value().rows()[i].candidate().candidate_id());
        REQUIRE(gpu.value().rows()[i].score() == host.value().rows()[i].score());
    }

    StatusOr<GpuCandidateExport::Result> bounded =
        GpuCandidateExport::vigenere_bounded(cipher, freqs.value(), 3, 6);
    REQUIRE(bounded.ok());
    REQUIRE(bounded.value().size() == 3);
    for (const GpuCandidateExport::Row& row : bounded.value().rows()) {
        StatusOr<std::vector<Index29>> out = VigenereKeyTransform{}.apply(
            cipher, row.candidate().params(), TransformDirection::Decrypt);
        REQUIRE(out.ok());
        StatusOr<double> score = ScoreRegistry::score(
            "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
        REQUIRE(score.ok());
        REQUIRE(row.score() == score.value());
    }
}

#endif
