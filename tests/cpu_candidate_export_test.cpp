#include <parcae/core/index29.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/score/expected_frequency_table.hpp>
#include <parcae/score/score_registry.hpp>
#include <parcae/score/score_request.hpp>
#include <parcae/search/cpu_candidate_export.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_prior.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

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

[[nodiscard]] parcae::tool::Context test_context() {
    return parcae::tool::Context{std::string(PARCAE_TEST_DATA_DIR)};
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
    "CpuCandidateExport caesar top-k matches GpuCandidateExport host scores",
    "[search][export][cpu][caesar]") {
    const parcae::tool::Context ctx = test_context();
    const std::vector<Index29> cipher = synthetic_cipher();

    constexpr std::size_t k = 5;
    StatusOr<CpuCandidateExport::Result> cpu = CpuCandidateExport::run(
        cipher,
        "caesar",
        "chi2_english_gp_v0",
        k,
        ctx,
        TransformDirection::Decrypt);
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().size() == k);

    StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
    REQUIRE(freqs.ok());
    const std::vector<double> scores = cpu_chi2_by_shift(cipher, freqs.value());
    StatusOr<GpuCandidateExport::Result> gpu =
        GpuCandidateExport::caesar_from_host_scores(cipher, scores, k);
    REQUIRE(gpu.ok());

    for (std::size_t i = 0; i < k; ++i) {
        REQUIRE(
            cpu.value().rows()[i].candidate().candidate_id() ==
            gpu.value().rows()[i].candidate().candidate_id());
        REQUIRE(cpu.value().rows()[i].score() == gpu.value().rows()[i].score());
        REQUIRE(
            cpu.value().rows()[i].candidate().output_indices() ==
            gpu.value().rows()[i].candidate().output_indices());
    }
}

TEST_CASE("CpuCandidateExport exclusion removes rejected params", "[search][export][cpu]") {
    const parcae::tool::Context ctx = test_context();
    const std::vector<Index29> cipher = synthetic_cipher();

    const nlohmann::json shift7_params = {{"shift", 7}};
    const std::string shift7_hash = SearchPrior::param_hash_of(shift7_params);

    StatusOr<SearchPrior> prior = SearchPrior::make(
        "lp2-page-0-explore",
        {},
        {SearchPrior::Exclusion{shift7_hash, "h-reject-7", "rejected"}},
        "2026-09-21T18:00:00Z");
    REQUIRE(prior.ok());

    constexpr std::size_t k = 29;
    StatusOr<CpuCandidateExport::Result> with_prior = CpuCandidateExport::run(
        cipher,
        "caesar",
        "chi2_english_gp_v0",
        k,
        ctx,
        TransformDirection::Decrypt,
        nlohmann::json::object(),
        &prior.value());
    REQUIRE(with_prior.ok());
    REQUIRE(with_prior.value().size() == k - 1);

    for (const CpuCandidateExport::Row& row : with_prior.value().rows()) {
        REQUIRE(row.candidate().params() != shift7_params);
    }
}

TEST_CASE("CpuCandidateExport seed envelope is ranked when not in grid params", "[search][export][cpu]") {
    const parcae::tool::Context ctx = test_context();
    const std::vector<Index29> cipher = synthetic_cipher();

    // Compose envelope not produced by gen_caesar (different transform) — forced seed lane.
    const nlohmann::json envelope = {
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", 7}}},
    };

    // Exclude shift 7 from the grid so only the seed path retains that plaintext.
    const std::string shift7_hash = SearchPrior::param_hash_of(envelope.at("params"));
    StatusOr<SearchPrior> prior = SearchPrior::make(
        "lp2-page-0-explore",
        {SearchPrior::Seed{"h-promoted-7", envelope}},
        {SearchPrior::Exclusion{shift7_hash, "h-other", "rejected"}},
        "2026-09-21T18:00:00Z");
    REQUIRE(prior.ok());

    StatusOr<CpuCandidateExport::Result> exported = CpuCandidateExport::run(
        cipher,
        "caesar",
        "chi2_english_gp_v0",
        1,
        ctx,
        TransformDirection::Decrypt,
        nlohmann::json::object(),
        &prior.value());
    REQUIRE(exported.ok());
    REQUIRE(exported.value().size() == 1);
    REQUIRE(exported.value().rows()[0].candidate().candidate_id() == "prior-seed:h-promoted-7");
    REQUIRE(exported.value().rows()[0].candidate().params() == envelope.at("params"));
}

TEST_CASE("CpuCandidateExport rejects expansion above max_candidates", "[search][export][cpu]") {
    const parcae::tool::Context ctx = test_context();
    const std::vector<Index29> cipher = synthetic_cipher();

    StatusOr<CpuCandidateExport::Result> ok = CpuCandidateExport::run(
        cipher,
        "caesar",
        "chi2_english_gp_v0",
        5,
        ctx,
        TransformDirection::Decrypt,
        nlohmann::json::object(),
        nullptr,
        "v0",
        28);
    REQUIRE(!ok.ok());

    StatusOr<CpuCandidateExport::Result> pass = CpuCandidateExport::run(
        cipher,
        "caesar",
        "chi2_english_gp_v0",
        5,
        ctx,
        TransformDirection::Decrypt,
        nlohmann::json::object(),
        nullptr,
        "v0",
        29);
    REQUIRE(pass.ok());
}

TEST_CASE(
    "CpuCandidateExport opt-in beaufort / totient families",
    "[search][export][cpu][extended]") {
    const parcae::tool::Context ctx = test_context();
    const std::vector<Index29> cipher = synthetic_cipher();

    StatusOr<SearchJob> denied = SearchJob::make(
        "_example",
        "totient",
        "chi2_english_gp_v0",
        3,
        1,
        parcae::tool::Backend::Cpu,
        64);
    REQUIRE_FALSE(denied.ok());

    StatusOr<SearchJob> totient_job = SearchJob::make(
        "_example",
        "totient",
        "chi2_english_gp_v0",
        3,
        1,
        parcae::tool::Backend::Cpu,
        64,
        TransformDirection::Decrypt,
        nlohmann::json{{"prime_start_count", 8}},
        std::nullopt,
        "v0",
        true);
    REQUIRE(totient_job.ok());
    StatusOr<CpuCandidateExport::Result> totient =
        CpuCandidateExport::from_job(cipher, totient_job.value(), ctx);
    REQUIRE(totient.ok());
    REQUIRE(totient.value().size() == 3);
    REQUIRE(
        totient.value().rows()[0].candidate().transform_id() ==
        TransformId::totient_prime_stream());

    StatusOr<SearchJob> beaufort_job = SearchJob::make(
        "_example",
        "beaufort",
        "chi2_english_gp_v0",
        2,
        1,
        parcae::tool::Backend::Cpu,
        64,
        TransformDirection::Decrypt,
        nlohmann::json{{"max_key_length", 4}},
        std::nullopt,
        "v0",
        true);
    REQUIRE(beaufort_job.ok());
    StatusOr<CpuCandidateExport::Result> beaufort =
        CpuCandidateExport::from_job(cipher, beaufort_job.value(), ctx);
    REQUIRE(beaufort.ok());
    REQUIRE(beaufort.value().size() == 2);
    REQUIRE(
        beaufort.value().rows()[0].candidate().transform_id() == TransformId::beaufort_key());
}
