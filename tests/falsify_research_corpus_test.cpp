#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <parcae/core/sha256.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "cli_spawn.hpp"

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_EXE
#error "PARCAE_PYTHON_EXE must be defined"
#endif
#ifndef PARCAE_SOURCE_DIR
#error "PARCAE_SOURCE_DIR must be defined"
#endif

#if defined(PARCAE_HAS_CLI_GOLDENS)
#include "parcae_cli_paths.h"
#endif

namespace {

constexpr std::string_view kWorkspaceId = "_falsify_schema_check";
constexpr std::string_view kBatchId = "b_falsify_schema_v0";
constexpr const char* kHypothesisIds[] = {
    "h_nt_stream_a_phi",
    "h_nt_stream_b_nth_prime",
    "h_nt_stream_c_prime_gaps",
};

[[nodiscard]] std::filesystem::path make_sandbox(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    std::filesystem::create_directories(root / "fixtures" / "solved" / "a-warning", ec);
    {
        std::ofstream out(root / "fixtures" / "solved" / "a-warning" / "ciphertext.txt");
        out << "placeholder\n";
    }
    return root;
}

[[nodiscard]] std::string file_sha256(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return Sha256::hex_digest(bytes);
}

[[nodiscard]] CliSpawnResult run_falsify_schema_self_check(const std::filesystem::path& data_dir,
                                                           const std::string& tmp_tag) {
    const auto script =
        std::filesystem::path(PARCAE_SOURCE_DIR) / "scripts" / "research" /
        "falsify_nt_keystreams_lp2.py";
    REQUIRE(std::filesystem::is_regular_file(script));

    std::vector<std::string> args{
        script.string(),
        "--schema-self-check",
        "--data-dir",
        data_dir.string(),
    };
#if defined(PARCAE_HAS_CLI_GOLDENS) && defined(PARCAE_CLI_HYPOTHESIS)
    args.push_back("--hypothesis-exe");
    args.push_back(PARCAE_CLI_HYPOTHESIS);
#endif
    return run_cli_capture(std::filesystem::path(PARCAE_PYTHON_EXE), args, tmp_tag);
}

[[nodiscard]] nlohmann::json stable_snapshot(const std::filesystem::path& data_dir) {
    StatusOr<BatchArtifact> batch = BatchArtifact::load(data_dir, kWorkspaceId, kBatchId);
    REQUIRE(batch.ok());
    REQUIRE(batch.value().family() == "vigenere");
    REQUIRE(batch.value().backend() == Backend::Cpu);
    REQUIRE(batch.value().candidate_count() == 2);
    REQUIRE(batch.value().manifest_to_json().at("ordering").get<std::string>() ==
            BatchArtifact::ordering_id);

    nlohmann::json hyp_digests = nlohmann::json::object();
    for (const char* hid : kHypothesisIds) {
        StatusOr<HypothesisRecord> hyp = HypothesisRecord::load(data_dir, kWorkspaceId, hid);
        REQUIRE(hyp.ok());
        REQUIRE(hyp.value().status() == HypothesisStatus::Rejected);
        REQUIRE(hyp.value().method().at("transform_id").get<std::string>() == "vigenere_key");
        REQUIRE(hyp.value().source().at("family").get<std::string>() == "vigenere");
        REQUIRE_FALSE(hyp.value().scores().empty());
        REQUIRE(hyp.value().scores()[0].at("backend").get<std::string>() == "cpu");
        REQUIRE(hyp.value().scores()[0].contains("scored_utc"));

        StatusOr<std::filesystem::path> path =
            WorkspacePaths::hypothesis_file(data_dir, kWorkspaceId, hid);
        REQUIRE(path.ok());
        hyp_digests[hid] = file_sha256(path.value());
    }

    StatusOr<std::filesystem::path> batch_dir =
        WorkspacePaths::batch_dir(data_dir, kWorkspaceId, kBatchId);
    REQUIRE(batch_dir.ok());
    return nlohmann::json{
        {"job_digest", batch.value().job_digest_sha256()},
        {"prior_digest", batch.value().prior_digest_sha256()},
        {"candidate_count", batch.value().candidate_count()},
        {"manifest_sha256", file_sha256(batch_dir.value() / "manifest.json")},
        {"candidates_sha256", file_sha256(batch_dir.value() / "candidates.jsonl")},
        {"hypothesis_sha256", std::move(hyp_digests)},
    };
}

} // namespace

TEST_CASE("falsify research corpus: C++ loads BatchArtifact + HypothesisRecords",
          "[research][falsify][load]") {
    const auto root = make_sandbox("parcae_falsify_research_load");
    const CliSpawnResult run = run_falsify_schema_self_check(root, "falsify_load");
    INFO(run.stderr_text);
    INFO(run.stdout_text);
    REQUIRE(run.exit_code == 0);

    const nlohmann::json snap = stable_snapshot(root);
    REQUIRE(snap.at("candidate_count").get<std::size_t>() == 2);
    REQUIRE(snap.at("hypothesis_sha256").size() == 3);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("falsify research corpus: two deterministic runs yield identical digests",
          "[research][falsify][determinism]") {
    auto run_once = [](std::string_view sandbox, const std::string& tag) {
        const auto root = make_sandbox(sandbox);
        const CliSpawnResult run = run_falsify_schema_self_check(root, tag);
        INFO(run.stderr_text);
        INFO(run.stdout_text);
        REQUIRE(run.exit_code == 0);
        nlohmann::json snap = stable_snapshot(root);
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        return snap;
    };

    const nlohmann::json a = run_once("parcae_falsify_research_det_a", "falsify_det_a");
    const nlohmann::json b = run_once("parcae_falsify_research_det_b", "falsify_det_b");

    REQUIRE(a.at("job_digest") == b.at("job_digest"));
    REQUIRE(a.at("prior_digest") == b.at("prior_digest"));
    REQUIRE(a.at("manifest_sha256") == b.at("manifest_sha256"));
    REQUIRE(a.at("candidates_sha256") == b.at("candidates_sha256"));
    REQUIRE(a.at("hypothesis_sha256") == b.at("hypothesis_sha256"));
}

TEST_CASE("falsify research corpus: second ensure leaves writer digests unchanged",
          "[research][falsify][determinism]") {
    const auto root = make_sandbox("parcae_falsify_research_idempotent");
    const CliSpawnResult first = run_falsify_schema_self_check(root, "falsify_idemp1");
    INFO(first.stderr_text);
    REQUIRE(first.exit_code == 0);
    const nlohmann::json before = stable_snapshot(root);

    const CliSpawnResult second = run_falsify_schema_self_check(root, "falsify_idemp2");
    INFO(second.stderr_text);
    REQUIRE(second.exit_code == 0);
    const nlohmann::json after = stable_snapshot(root);

    REQUIRE(before == after);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}
