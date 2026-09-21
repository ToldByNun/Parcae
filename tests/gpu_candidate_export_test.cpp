#include <parcae/batch/batch_ordering.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/score_order.hpp>
#include <parcae/score/score_registry.hpp>
#include <parcae/score/score_request.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
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
    REQUIRE(exported.value().backend() == parcae::tool::Backend::Cpu);

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

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"

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
    REQUIRE(gpu.value().backend() == parcae::tool::Backend::Cuda);
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

#endif
