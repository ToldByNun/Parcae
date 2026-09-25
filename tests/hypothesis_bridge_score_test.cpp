#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <parcae/core/index29.hpp>
#include <parcae/core/sha256.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/hypothesis_bridge.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_prior.hpp>
#include <parcae/tool/api.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/tool/transform_envelope.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

#if defined(PARCAE_HAS_CLI_GOLDENS)
#include "cli_spawn.hpp"
#include "parcae_cli_paths.h"
#if !defined(PARCAE_CLI_HYPOTHESIS)
#error "PARCAE_CLI_HYPOTHESIS required"
#endif
#endif

namespace {

[[nodiscard]] std::filesystem::path repo_data() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

[[nodiscard]] std::filesystem::path make_sandbox(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    std::filesystem::create_directories(root / "profiles" / "scores", ec);
    std::filesystem::create_directories(root / "profiles" / "gematria", ec);
    std::filesystem::copy_file(repo_data() / "profiles" / "scores" / "english-gp-expected-v0.json",
                               root / "profiles" / "scores" / "english-gp-expected-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(repo_data() / "profiles" / "gematria" / "gematria-primus-v0.json",
                               root / "profiles" / "gematria" / "gematria-primus-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    return root;
}

[[nodiscard]] TransformCandidate caesar_candidate(std::uint8_t shift,
                                                  const std::vector<Index29>& output) {
    return TransformCandidate("caesar:shift=" + std::to_string(shift), TransformId::caesar(),
                              TransformDirection::Decrypt,
                              nlohmann::json{{"shift", static_cast<int>(shift)}}, output);
}

[[nodiscard]] StatusOr<BatchArtifact> make_batch(std::string_view workspace_id,
                                                 std::string_view batch_id,
                                                 std::vector<nlohmann::json> lines) {
    const std::size_t k = lines.empty() ? 1 : lines.size();
    StatusOr<SearchJob> job =
        SearchJob::make(workspace_id, "caesar", "chi2_english_gp_v0", k, 1, Backend::Cpu, 64);
    if (!job.ok()) {
        return job.status();
    }
    StatusOr<SearchPrior> prior = SearchPrior::make(workspace_id, {}, {}, "2026-09-22T12:00:00Z");
    if (!prior.ok()) {
        return prior.status();
    }
    return BatchArtifact::make(workspace_id, batch_id, "2026-09-22T12:00:01Z",
                               job.value().job_digest_sha256(), prior.value().prior_digest_sha256(),
                               "caesar", "chi2_english_gp_v0", "v0", Backend::Cpu, k, 1,
                               std::move(lines));
}

/// Mirror `parcae-hypothesis score` library path (search-loop post-pass).
[[nodiscard]] Status score_hypothesis_like_cli(const std::filesystem::path& data_root,
                                               const Context& ctx, std::string_view workspace_id,
                                               std::string_view hypothesis_id,
                                               std::span<const Index29> cipher,
                                               std::string_view score_id, std::string_view utc) {
    StatusOr<HypothesisRecord> loaded =
        HypothesisRecord::load(data_root, workspace_id, hypothesis_id);
    if (!loaded.ok()) {
        return loaded.status();
    }
    HypothesisRecord record = std::move(loaded.value());

    StatusOr<TransformEnvelope> envelope = TransformEnvelope::from_json(record.method());
    if (!envelope.ok()) {
        return envelope.status();
    }
    StatusOr<std::vector<Index29>> plain = ToolApi::apply_to_indices(cipher, envelope.value());
    if (!plain.ok()) {
        return plain.status();
    }
    StatusOr<double> value = ToolApi::score(ctx, plain.value(), score_id);
    if (!value.ok()) {
        return value.status();
    }

    StatusOr<std::string> latin = ToolApi::to_latin(ctx, plain.value());
    if (!latin.ok()) {
        return latin.status();
    }
    std::string prefix = latin.value();
    if (prefix.size() > 64) {
        prefix.resize(64);
    }

    Status append = record.append_score(nlohmann::json{
        {"score_id", std::string(score_id)},
        {"score_version", "v0"},
        {"value", value.value()},
        {"backend", "cpu"},
        {"scored_utc", std::string(utc)},
    });
    if (!append.ok()) {
        return append;
    }

    record.set_preview(nlohmann::json{{"latin_prefix", prefix}, {"max_chars", 64}});
    record.set_output_indices_digest(plain.value());
    record.recompute_method_digest();
    record.set_updated_utc(std::string(utc));

    if (record.status() == HypothesisStatus::Draft) {
        Status to_proposed = record.set_status(HypothesisStatus::Proposed);
        if (!to_proposed.ok()) {
            return to_proposed;
        }
    }
    if (record.status() == HypothesisStatus::Proposed) {
        Status to_scored = record.set_status(HypothesisStatus::Scored);
        if (!to_scored.ok()) {
            return to_scored;
        }
    }
    return record.store(data_root);
}

#if defined(PARCAE_HAS_CLI_GOLDENS)
[[nodiscard]] CliSpawnResult run_hypothesis_cli(const std::vector<std::string>& args) {
    return run_cli_capture(PARCAE_CLI_HYPOTHESIS, args, "f22_hyp");
}
#endif

} // namespace

TEST_CASE("ingest → hypothesis_score path → set-status rejected/promoted",
          "[search][bridge][score]") {
    const auto root = make_sandbox("parcae_hypothesis_bridge_f22_score");
    const Context ctx{root};

    StatusOr<WorkspaceManifest> ws =
        WorkspaceManifest::make("f22-ws", "2026-09-22T12:00:00Z", "f22");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    const std::vector<Index29> plain = {Index29{0},  Index29{1},  Index29{2},  Index29{3},
                                        Index29{4},  Index29{5},  Index29{6},  Index29{7},
                                        Index29{8},  Index29{9},  Index29{10}, Index29{11},
                                        Index29{12}, Index29{13}, Index29{14}, Index29{15}};
    StatusOr<std::vector<Index29>> cipher =
        CaesarTransform{}.apply(plain, nlohmann::json{{"shift", 7}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<std::vector<Index29>> out7 = CaesarTransform{}.apply(
        cipher.value(), nlohmann::json{{"shift", 7}}, TransformDirection::Decrypt);
    StatusOr<std::vector<Index29>> out3 = CaesarTransform{}.apply(
        cipher.value(), nlohmann::json{{"shift", 3}}, TransformDirection::Decrypt);
    REQUIRE(out7.ok());
    REQUIRE(out3.ok());

    std::vector<nlohmann::json> lines;
    lines.push_back(BatchArtifact::candidate_wire(
        caesar_candidate(7, out7.value()), "chi2_english_gp_v0", "v0", 1.0, Backend::Cpu, 0));
    lines.push_back(BatchArtifact::candidate_wire(
        caesar_candidate(3, out3.value()), "chi2_english_gp_v0", "v0", 9.0, Backend::Cpu, 1));

    StatusOr<BatchArtifact> art = make_batch("f22-ws", "b-f22-0001", std::move(lines));
    REQUIRE(art.ok());
    REQUIRE(art.value().store(root).ok());

    StatusOr<HypothesisBridge::Result> ingested = HypothesisBridge::ingest(root, art.value());
    REQUIRE(ingested.ok());
    REQUIRE(ingested.value().created_ids().size() == 2);

    const std::string id7 =
        HypothesisBridge::hypothesis_id_for("f22-ws", "b-f22-0001", "caesar:shift=7");
    const std::string id3 =
        HypothesisBridge::hypothesis_id_for("f22-ws", "b-f22-0001", "caesar:shift=3");

    StatusOr<HypothesisRecord> before = HypothesisRecord::load(root, "f22-ws", id7);
    REQUIRE(before.ok());
    REQUIRE(before.value().status() == HypothesisStatus::Proposed);
    REQUIRE(before.value().scores().empty());
    REQUIRE(before.value().source().at("batch_id").get<std::string>() == "b-f22-0001");

    REQUIRE(score_hypothesis_like_cli(root, ctx, "f22-ws", id7, cipher.value(),
                                      "chi2_english_gp_v0", "2026-09-22T12:01:00Z")
                .ok());
    REQUIRE(score_hypothesis_like_cli(root, ctx, "f22-ws", id3, cipher.value(),
                                      "chi2_english_gp_v0", "2026-09-22T12:01:00Z")
                .ok());

    StatusOr<HypothesisRecord> scored7 = HypothesisRecord::load(root, "f22-ws", id7);
    REQUIRE(scored7.ok());
    REQUIRE(scored7.value().status() == HypothesisStatus::Scored);
    REQUIRE(scored7.value().scores().size() == 1);
    REQUIRE(scored7.value().scores().at(0).at("score_id").get<std::string>() ==
            "chi2_english_gp_v0");
    REQUIRE(scored7.value().scores().at(0).at("backend").get<std::string>() == "cpu");
    REQUIRE(scored7.value().preview().at("latin_prefix").is_string());
    REQUIRE_FALSE(scored7.value().digests().at("output_indices_sha256").is_null());

    // Promote best lane; reject the other (hypothesis_set_status contract).
    REQUIRE(scored7.value().set_status(HypothesisStatus::Promoted).ok());
    scored7.value().set_updated_utc("2026-09-22T12:02:00Z");
    REQUIRE(scored7.value().store(root).ok());

    StatusOr<HypothesisRecord> scored3 = HypothesisRecord::load(root, "f22-ws", id3);
    REQUIRE(scored3.ok());
    REQUIRE(scored3.value().status() == HypothesisStatus::Scored);
    REQUIRE(scored3.value().set_status(HypothesisStatus::Rejected).ok());
    scored3.value().set_updated_utc("2026-09-22T12:02:00Z");
    REQUIRE(scored3.value().store(root).ok());

    StatusOr<SearchPrior> prior =
        SearchPrior::from_workspace(root, "f22-ws", "2026-09-22T12:03:00Z");
    REQUIRE(prior.ok());
    REQUIRE(prior.value().seeds().size() == 1);
    REQUIRE(prior.value().seeds()[0].hypothesis_id() == id7);
    REQUIRE(prior.value().exclusions().size() == 1);
    REQUIRE(prior.value().exclusions()[0].hypothesis_id() == id3);

    // Bridge must not auto-promote on re-ingest.
    REQUIRE(HypothesisBridge::ingest(root, art.value()).ok());
    StatusOr<HypothesisRecord> still = HypothesisRecord::load(root, "f22-ws", id7);
    REQUIRE(still.ok());
    REQUIRE(still.value().status() == HypothesisStatus::Promoted);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

#if defined(PARCAE_HAS_CLI_GOLDENS)

TEST_CASE("CLI hypothesis_score + set-status after bridge ingest", "[search][bridge][score][cli]") {
    const auto root = make_sandbox("parcae_hypothesis_bridge_f22_cli");

    StatusOr<WorkspaceManifest> ws =
        WorkspaceManifest::make("f22-cli-ws", "2026-09-22T12:00:00Z", "f22-cli");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(root).ok());

    const std::vector<Index29> plain = {Index29{19}, Index29{7},  Index29{4},
                                        Index29{0},  Index29{13}, Index29{6}};
    StatusOr<std::vector<Index29>> cipher =
        CaesarTransform{}.apply(plain, nlohmann::json{{"shift", 5}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());
    StatusOr<std::vector<Index29>> out = CaesarTransform{}.apply(
        cipher.value(), nlohmann::json{{"shift", 5}}, TransformDirection::Decrypt);
    REQUIRE(out.ok());

    // Write indices input for CLI --indices.
    const auto cipher_path = root / "cipher.indices.txt";
    {
        std::ofstream out_file(cipher_path);
        REQUIRE(out_file);
        for (std::size_t i = 0; i < cipher.value().size(); ++i) {
            if (i != 0) {
                out_file << ',';
            }
            out_file << static_cast<int>(cipher.value()[i].value());
        }
    }

    std::vector<nlohmann::json> lines;
    lines.push_back(BatchArtifact::candidate_wire(
        caesar_candidate(5, out.value()), "chi2_english_gp_v0", "v0", 1.0, Backend::Cpu, 0));
    StatusOr<BatchArtifact> art = make_batch("f22-cli-ws", "b-f22-cli", std::move(lines));
    REQUIRE(art.ok());
    REQUIRE(HypothesisBridge::ingest(root, art.value()).ok());

    const std::string hid =
        HypothesisBridge::hypothesis_id_for("f22-cli-ws", "b-f22-cli", "caesar:shift=5");

    const CliSpawnResult score_run =
        run_hypothesis_cli({"--data-dir", root.string(), "score", "--workspace", "f22-cli-ws",
                            "--id", hid, "--indices", "--input", cipher_path.string(), "--score-id",
                            "chi2_english_gp_v0", "--json"});
    INFO(score_run.stdout_text);
    REQUIRE(score_run.exit_code == 0);

    StatusOr<HypothesisRecord> scored = HypothesisRecord::load(root, "f22-cli-ws", hid);
    REQUIRE(scored.ok());
    REQUIRE(scored.value().status() == HypothesisStatus::Scored);
    REQUIRE(scored.value().scores().size() == 1);

    const CliSpawnResult status_run =
        run_hypothesis_cli({"--data-dir", root.string(), "set-status", "--workspace", "f22-cli-ws",
                            "--id", hid, "--status", "rejected", "--json"});
    INFO(status_run.stdout_text);
    REQUIRE(status_run.exit_code == 0);

    StatusOr<HypothesisRecord> rejected = HypothesisRecord::load(root, "f22-cli-ws", hid);
    REQUIRE(rejected.ok());
    REQUIRE(rejected.value().status() == HypothesisStatus::Rejected);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

#else

TEST_CASE("CLI hypothesis_score after ingest skipped (no CLI goldens)",
          "[search][bridge][score][cli]") {
    SUCCEED("PARCAE_HAS_CLI_GOLDENS unset — library score path covered above");
}

#endif
