#include <parcae/core/index29.hpp>
#include <parcae/core/sha256.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/hypothesis_bridge.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_prior.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::string fake_digest(std::string_view label) {
    return Sha256::hex_digest(label);
}

[[nodiscard]] TransformCandidate caesar_candidate(std::uint8_t shift) {
    return TransformCandidate(
        "caesar:shift=" + std::to_string(shift),
        TransformId::caesar(),
        TransformDirection::Decrypt,
        nlohmann::json{{"shift", static_cast<int>(shift)}},
        {Index29{19}, Index29{7}, Index29{4}});
}

[[nodiscard]] StatusOr<BatchArtifact> make_test_batch(
    std::string_view workspace_id,
    std::string_view batch_id,
    std::vector<nlohmann::json> lines) {
    const std::size_t k = lines.empty() ? 1 : lines.size();
    StatusOr<SearchJob> job = SearchJob::make(
        workspace_id,
        "caesar",
        "chi2_english_gp_v0",
        k,
        1,
        parcae::tool::Backend::Cpu,
        64);
    if (!job.ok()) {
        return job.status();
    }
    StatusOr<SearchPrior> prior =
        SearchPrior::make(workspace_id, {}, {}, "2026-09-21T18:00:00Z");
    if (!prior.ok()) {
        return prior.status();
    }
    return BatchArtifact::make(
        workspace_id,
        batch_id,
        "2026-09-21T18:00:01Z",
        job.value().job_digest_sha256(),
        prior.value().prior_digest_sha256(),
        "caesar",
        "chi2_english_gp_v0",
        "v0",
        parcae::tool::Backend::Cpu,
        k,
        1,
        std::move(lines));
}

}  // namespace

TEST_CASE(
    "HypothesisBridge ingest creates proposed records from batch top-k",
    "[search][bridge]") {
    const auto tmp =
        std::filesystem::temp_directory_path() / "parcae_hypothesis_bridge_f19";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    std::vector<nlohmann::json> lines;
    lines.push_back(BatchArtifact::candidate_wire(
        caesar_candidate(3), "chi2_english_gp_v0", "v0", 12.34,
        parcae::tool::Backend::Cpu, 0));
    lines.push_back(BatchArtifact::candidate_wire(
        caesar_candidate(7), "chi2_english_gp_v0", "v0", 20.0,
        parcae::tool::Backend::Cpu, 1));

    StatusOr<BatchArtifact> art =
        make_test_batch("bridge-ws", "b-caesar-0001", std::move(lines));
    REQUIRE(art.ok());
    REQUIRE(art.value().store(tmp).ok());

    StatusOr<HypothesisBridge::Result> ingested =
        HypothesisBridge::ingest(tmp, art.value());
    REQUIRE(ingested.ok());
    REQUIRE(ingested.value().created_ids().size() == 2);
    REQUIRE(ingested.value().updated_ids().empty());
    REQUIRE(ingested.value().skipped_ids().empty());

    const std::string id0 = HypothesisBridge::hypothesis_id_for(
        "bridge-ws", "b-caesar-0001", "caesar:shift=3");
    const std::string id1 = HypothesisBridge::hypothesis_id_for(
        "bridge-ws", "b-caesar-0001", "caesar:shift=7");
    REQUIRE(id0 != id1);
    REQUIRE(id0.size() == 33);
    REQUIRE(id0[0] == 'h');

    StatusOr<HypothesisRecord> h0 = HypothesisRecord::load(tmp, "bridge-ws", id0);
    REQUIRE(h0.ok());
    REQUIRE(h0.value().status() == HypothesisStatus::Proposed);
    REQUIRE(h0.value().method().at("transform_id").get<std::string>() == "caesar");
    REQUIRE(h0.value().method().at("params").at("shift").get<int>() == 3);
    REQUIRE(h0.value().source().at("candidate_id").get<std::string>() == "caesar:shift=3");
    REQUIRE(h0.value().source().at("batch_id").get<std::string>() == "b-caesar-0001");
    REQUIRE(h0.value().source().at("generator_id").get<std::string>() == "gen_caesar");
    REQUIRE(h0.value().source().at("family").get<std::string>() == "caesar");
    REQUIRE(h0.value().scores().empty());
    REQUIRE_FALSE(h0.value().digests().at("method_sha256").is_null());

    StatusOr<HypothesisRecord> h1 = HypothesisRecord::load(tmp, "bridge-ws", id1);
    REQUIRE(h1.ok());
    REQUIRE(h1.value().status() == HypothesisStatus::Proposed);

    // Re-ingest is idempotent by id (updates, no duplicate files).
    StatusOr<HypothesisBridge::Result> again = HypothesisBridge::ingest(tmp, art.value());
    REQUIRE(again.ok());
    REQUIRE(again.value().created_ids().empty());
    REQUIRE(again.value().updated_ids().size() == 2);

    StatusOr<std::vector<std::string>> listed =
        HypothesisRecord::list_ids(tmp, "bridge-ws");
    REQUIRE(listed.ok());
    REQUIRE(listed.value().size() == 2);

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE(
    "HypothesisBridge overwrite_existing=false skips existing ids",
    "[search][bridge]") {
    const auto tmp =
        std::filesystem::temp_directory_path() / "parcae_hypothesis_bridge_f19_skip";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    std::vector<nlohmann::json> lines;
    lines.push_back(BatchArtifact::candidate_wire(
        caesar_candidate(3), "chi2_english_gp_v0", "v0", 1.0,
        parcae::tool::Backend::Cpu, 0));

    StatusOr<BatchArtifact> art =
        make_test_batch("bridge-ws-skip", "b-skip-0001", std::move(lines));
    REQUIRE(art.ok());
    REQUIRE(art.value().store(tmp).ok());
    REQUIRE(HypothesisBridge::ingest(tmp, art.value()).ok());

    HypothesisBridge::Options opts;
    opts.overwrite_existing = false;
    StatusOr<HypothesisBridge::Result> skipped =
        HypothesisBridge::ingest(tmp, art.value(), opts);
    REQUIRE(skipped.ok());
    REQUIRE(skipped.value().created_ids().empty());
    REQUIRE(skipped.value().updated_ids().empty());
    REQUIRE(skipped.value().skipped_ids().size() == 1);

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE(
    "HypothesisBridge does not auto-promote; preserves scored status on refresh",
    "[search][bridge]") {
    const auto tmp =
        std::filesystem::temp_directory_path() / "parcae_hypothesis_bridge_f19_scored";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    std::vector<nlohmann::json> lines;
    lines.push_back(BatchArtifact::candidate_wire(
        caesar_candidate(3), "chi2_english_gp_v0", "v0", 1.0,
        parcae::tool::Backend::Cpu, 0));

    StatusOr<BatchArtifact> art =
        make_test_batch("bridge-ws-scored", "b-scored-0001", std::move(lines));
    REQUIRE(art.ok());
    REQUIRE(HypothesisBridge::ingest(tmp, art.value()).ok());

    const std::string id = HypothesisBridge::hypothesis_id_for(
        "bridge-ws-scored", "b-scored-0001", "caesar:shift=3");
    StatusOr<HypothesisRecord> record = HypothesisRecord::load(tmp, "bridge-ws-scored", id);
    REQUIRE(record.ok());
    REQUIRE(record.value().set_status(HypothesisStatus::Scored).ok());
    REQUIRE(record.value()
                .append_score(nlohmann::json{
                    {"score_id", "chi2_english_gp_v0"},
                    {"score_version", "v0"},
                    {"value", 1.0},
                    {"backend", "cpu"},
                    {"scored_utc", "2026-09-21T18:05:00Z"},
                })
                .ok());
    REQUIRE(record.value().store(tmp).ok());

    StatusOr<HypothesisBridge::Result> refreshed =
        HypothesisBridge::ingest(tmp, art.value());
    REQUIRE(refreshed.ok());
    REQUIRE(refreshed.value().updated_ids().size() == 1);

    StatusOr<HypothesisRecord> again =
        HypothesisRecord::load(tmp, "bridge-ws-scored", id);
    REQUIRE(again.ok());
    REQUIRE(again.value().status() == HypothesisStatus::Scored);
    REQUIRE(again.value().scores().size() == 1);
    REQUIRE(again.value().status() != HypothesisStatus::Promoted);

    std::filesystem::remove_all(tmp, ec);
}
