#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <parcae/core/index29.hpp>
#include <parcae/core/sha256.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_prior.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <string>
#include <string_view>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::string fake_digest(std::string_view label) {
    return Sha256::hex_digest(label);
}

[[nodiscard]] TransformCandidate caesar_candidate(std::uint8_t shift) {
    StatusOr<TransformId> id = TransformId::from_string("caesar");
    REQUIRE(id.ok());
    return TransformCandidate("caesar-shift-" + std::to_string(shift), id.value(),
                              TransformDirection::Decrypt, nlohmann::json{{"shift", shift}},
                              {Index29{19}, Index29{7}, Index29{4}});
}

} // namespace

TEST_CASE("BatchArtifact store/load round-trip", "[search][batch]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_batch_artifact_b7";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    StatusOr<SearchJob> job =
        SearchJob::make("batch-ws", "caesar", "chi2_english_gp_v0", 2, 1, Backend::Cpu, 64);
    REQUIRE(job.ok());

    StatusOr<SearchPrior> prior = SearchPrior::make("batch-ws", {}, {}, "2026-09-21T18:00:00Z");
    REQUIRE(prior.ok());

    std::vector<nlohmann::json> lines;
    lines.push_back(BatchArtifact::candidate_wire(caesar_candidate(3), "chi2_english_gp_v0", "v0",
                                                  12.34, Backend::Cpu, 0));
    lines.push_back(BatchArtifact::candidate_wire(caesar_candidate(7), "chi2_english_gp_v0", "v0",
                                                  20.0, Backend::Cpu, 1));

    StatusOr<BatchArtifact> art =
        BatchArtifact::make("batch-ws", "b-caesar-20260921-0001", "2026-09-21T18:00:01Z",
                            job.value().job_digest_sha256(), prior.value().prior_digest_sha256(),
                            "caesar", "chi2_english_gp_v0", "v0", Backend::Cpu, 2, 1, lines,
                            nlohmann::json{{"candidate_count", 2}, {"notes", "b7"}});
    REQUIRE(art.ok());
    REQUIRE(art.value().candidate_count() == 2);
    REQUIRE(art.value().store(tmp).ok());

    StatusOr<BatchArtifact> loaded = BatchArtifact::load(tmp, "batch-ws", "b-caesar-20260921-0001");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().manifest_to_json() == art.value().manifest_to_json());
    REQUIRE(loaded.value().candidates() == art.value().candidates());
    REQUIRE(loaded.value().report().has_value());
    REQUIRE(loaded.value().report().value().at("notes").get<std::string>() == "b7");
    REQUIRE(WorkspacePaths::batch_dir(tmp, "batch-ws", "b-caesar-20260921-0001").ok());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("BatchArtifact rejects bad ranks, dup ids, timing report", "[search][batch]") {
    const std::string job_d = fake_digest("job");
    const std::string prior_d = fake_digest("prior");

    nlohmann::json good = BatchArtifact::candidate_wire(caesar_candidate(3), "chi2_english_gp_v0",
                                                        "v0", 1.0, Backend::Cpu, 0);

    {
        nlohmann::json bad = good;
        bad["rank"] = 1;
        REQUIRE_FALSE(BatchArtifact::make("batch-ws", "b-x", "2026-09-21T18:00:01Z", job_d, prior_d,
                                          "caesar", "chi2_english_gp_v0", "v0", Backend::Cpu, 1, 1,
                                          {bad})
                          .ok());
    }
    {
        nlohmann::json a = good;
        nlohmann::json b = good;
        b["rank"] = 1;
        REQUIRE_FALSE(BatchArtifact::make("batch-ws", "b-x", "2026-09-21T18:00:01Z", job_d, prior_d,
                                          "caesar", "chi2_english_gp_v0", "v0", Backend::Cpu, 2, 1,
                                          {a, b})
                          .ok());
    }
    {
        REQUIRE_FALSE(BatchArtifact::make("batch-ws", "b-x", "2026-09-21T18:00:01Z", job_d, prior_d,
                                          "caesar", "chi2_english_gp_v0", "v0", Backend::Cpu, 1, 1,
                                          {good}, nlohmann::json{{"tok_per_sec", 1.0}})
                          .ok());
    }
    {
        REQUIRE_FALSE(BatchArtifact::make("batch-ws", "BadId", "2026-09-21T18:00:01Z", job_d,
                                          prior_d, "caesar", "chi2_english_gp_v0", "v0",
                                          Backend::Cpu, 1, 1, {good})
                          .ok());
    }
}

TEST_CASE("BatchArtifact load rejects count mismatch", "[search][batch]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_batch_artifact_b7_bad";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    nlohmann::json line = BatchArtifact::candidate_wire(caesar_candidate(3), "chi2_english_gp_v0",
                                                        "v0", 1.0, Backend::Cpu, 0);
    StatusOr<BatchArtifact> art = BatchArtifact::make(
        "batch-ws", "b-bad-count", "2026-09-21T18:00:01Z", fake_digest("job2"),
        fake_digest("prior2"), "caesar", "chi2_english_gp_v0", "v0", Backend::Cpu, 1, 1, {line});
    REQUIRE(art.ok());
    REQUIRE(art.value().store(tmp).ok());

    StatusOr<std::filesystem::path> dir = WorkspacePaths::batch_dir(tmp, "batch-ws", "b-bad-count");
    REQUIRE(dir.ok());
    {
        std::ofstream out(dir.value() / "candidates.jsonl", std::ios::binary | std::ios::trunc);
        REQUIRE(out);
        out << line.dump() << '\n' << line.dump() << '\n';
    }
    REQUIRE_FALSE(BatchArtifact::load(tmp, "batch-ws", "b-bad-count").ok());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("BatchArtifact loads falsify-shaped research writer rows", "[search][batch][research]") {
    // Mirrors scripts/research/falsify_nt_keystreams_lp2.py schema writers:
    // ordering=batch_ordering_v0, backend=cpu, family=vigenere, transform_id=vigenere_key,
    // output_indices present, best-first chi2 Asc.
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_falsify_schema_batch";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    const std::string job_d = fake_digest("falsify-schema-self-check");
    const std::string prior_d = Sha256::hex_digest(std::string_view{});

    nlohmann::json row0 = {
        {"candidate_id", "lp2-00:B_nth_prime_mod:index:add:base=1"},
        {"envelope",
         {{"transform_id", "vigenere_key"},
          {"direction", "decrypt"},
          {"params",
           {{"stream", "B_nth_prime_mod"},
            {"domain", "index"},
            {"op", "add"},
            {"pos_base", 1},
            {"research", "nt_keystream_falsify"}}}}},
        {"output_indices", {4, 5, 6}},
        {"rank", 0},
        {"score",
         {{"score_id", "chi2_english_gp_v0"},
          {"score_version", "v0"},
          {"value", 90.0},
          {"backend", "cpu"}}},
    };
    nlohmann::json row1 = {
        {"candidate_id", "lp2-00:A_phi_pos:index:sub:base=1"},
        {"envelope",
         {{"transform_id", "vigenere_key"},
          {"direction", "decrypt"},
          {"params",
           {{"stream", "A_phi_pos"},
            {"domain", "index"},
            {"op", "sub"},
            {"pos_base", 1},
            {"research", "nt_keystream_falsify"}}}}},
        {"output_indices", {1, 2, 3}},
        {"rank", 1},
        {"score",
         {{"score_id", "chi2_english_gp_v0"},
          {"score_version", "v0"},
          {"value", 120.5},
          {"backend", "cpu"}}},
    };

    StatusOr<BatchArtifact> art = BatchArtifact::make(
        "_falsify_schema_check", "b_falsify_schema_v0", "2026-09-27T00:00:00Z", job_d, prior_d,
        "vigenere", "chi2_english_gp_v0", "v0", Backend::Cpu, 2, 0, {row0, row1});
    REQUIRE(art.ok());
    REQUIRE(art.value().store(tmp).ok());

    StatusOr<BatchArtifact> loaded =
        BatchArtifact::load(tmp, "_falsify_schema_check", "b_falsify_schema_v0");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().family() == "vigenere");
    REQUIRE(loaded.value().backend() == Backend::Cpu);
    REQUIRE(loaded.value().candidate_count() == 2);
    REQUIRE(loaded.value().candidates()[0].at("envelope").at("transform_id").get<std::string>() ==
            "vigenere_key");

    std::filesystem::remove_all(tmp, ec);
}
