#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <parcae/core/index29.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

} // namespace

TEST_CASE("HypothesisStatus transitions v0", "[hypothesis][status]") {
    REQUIRE(
        HypothesisStatusUtil::can_transition(HypothesisStatus::Draft, HypothesisStatus::Proposed));
    REQUIRE(
        HypothesisStatusUtil::can_transition(HypothesisStatus::Proposed, HypothesisStatus::Scored));
    REQUIRE(HypothesisStatusUtil::can_transition(HypothesisStatus::Proposed,
                                                 HypothesisStatus::Promoted));
    REQUIRE(
        HypothesisStatusUtil::can_transition(HypothesisStatus::Scored, HypothesisStatus::Rejected));
    REQUIRE_FALSE(
        HypothesisStatusUtil::can_transition(HypothesisStatus::Draft, HypothesisStatus::Scored));
    REQUIRE_FALSE(HypothesisStatusUtil::can_transition(HypothesisStatus::Rejected,
                                                       HypothesisStatus::Proposed));
    REQUIRE(
        HypothesisStatusUtil::require_transition(HypothesisStatus::Draft, HypothesisStatus::Draft)
            .ok());
}

TEST_CASE("WorkspacePaths validate id and reject traversal", "[hypothesis][paths]") {
    REQUIRE(WorkspacePaths::validate_id("_example").ok());
    REQUIRE(WorkspacePaths::validate_id("h-atbash-example").ok());
    REQUIRE_FALSE(WorkspacePaths::validate_id("").ok());
    REQUIRE_FALSE(WorkspacePaths::validate_id("BadId").ok());
    REQUIRE_FALSE(WorkspacePaths::validate_id("1leading").ok());

    StatusOr<std::filesystem::path> ws = WorkspacePaths::workspace_root(data_root(), "_example");
    REQUIRE(ws.ok());

    REQUIRE_FALSE(WorkspacePaths::resolve_under(ws.value(), "../fixtures/solved").ok());
    REQUIRE_FALSE(WorkspacePaths::resolve_under(ws.value(), "hypotheses/../../etc/passwd").ok());
    REQUIRE(WorkspacePaths::resolve_under(ws.value(), "hypotheses/h-atbash-example.json").ok());

    const auto fixtures_target = data_root() / "fixtures" / "solved" / "a-warning" / "evil.json";
    REQUIRE_FALSE(WorkspacePaths::deny_fixtures_write(data_root(), fixtures_target).ok());
}

TEST_CASE("WorkspacePaths research layout helpers", "[hypothesis][paths][research]") {
    StatusOr<std::filesystem::path> pages = WorkspacePaths::pages_dir(data_root(), "_example");
    REQUIRE(pages.ok());
    REQUIRE(pages.value() == data_root() / "workspaces" / "_example" / "pages");

    StatusOr<std::filesystem::path> research =
        WorkspacePaths::research_dir(data_root(), "_example");
    REQUIRE(research.ok());
    REQUIRE(research.value() == data_root() / "workspaces" / "_example" / "research");

    StatusOr<std::filesystem::path> readme = WorkspacePaths::readme_path(data_root(), "_example");
    REQUIRE(readme.ok());
    REQUIRE(readme.value() == data_root() / "workspaces" / "_example" / "README.md");

    StatusOr<std::filesystem::path> source = WorkspacePaths::source_path(data_root(), "_example");
    REQUIRE(source.ok());
    REQUIRE(source.value() == data_root() / "workspaces" / "_example" / "SOURCE.txt");

    StatusOr<std::filesystem::path> inputs = WorkspacePaths::inputs_dir(data_root(), "_example");
    REQUIRE(inputs.ok());
    REQUIRE(inputs.value() == data_root() / "workspaces" / "_example" / "inputs");

    StatusOr<std::filesystem::path> transcripts =
        WorkspacePaths::transcripts_dir(data_root(), "_example");
    REQUIRE(transcripts.ok());
    REQUIRE(transcripts.value() == data_root() / "workspaces" / "_example" / "transcripts");

    REQUIRE_FALSE(WorkspacePaths::pages_dir(data_root(), "BadId").ok());
    REQUIRE_FALSE(WorkspacePaths::research_dir(data_root(), "").ok());
    REQUIRE_FALSE(WorkspacePaths::readme_path(data_root(), "../evil").ok());
    REQUIRE_FALSE(WorkspacePaths::source_path(data_root(), "1leading").ok());

    REQUIRE(WorkspacePaths::page_relative_path(0).value() == "pages/00.txt");
    REQUIRE(WorkspacePaths::page_relative_path(7).value() == "pages/07.txt");
    REQUIRE(WorkspacePaths::page_relative_path(55).value() == "pages/55.txt");
    REQUIRE(WorkspacePaths::page_relative_path(7, 3).value() == "pages/007.txt");
    REQUIRE_FALSE(WorkspacePaths::page_relative_path(100).ok());
    REQUIRE_FALSE(WorkspacePaths::page_relative_path(0, 0).ok());

    StatusOr<std::filesystem::path> page0 =
        WorkspacePaths::page_file(data_root(), "_example", 0);
    REQUIRE(page0.ok());
    REQUIRE(page0.value() == data_root() / "workspaces" / "_example" / "pages" / "00.txt");
}

TEST_CASE("HypothesisRecord loads committed _example", "[hypothesis][load]") {
    StatusOr<HypothesisRecord> record =
        HypothesisRecord::load(data_root(), "_example", "h-atbash-example");
    REQUIRE(record.ok());
    REQUIRE(record.value().id() == "h-atbash-example");
    REQUIRE(record.value().workspace_id() == "_example");
    REQUIRE(record.value().status() == HypothesisStatus::Proposed);
    REQUIRE(record.value().method().at("transform_id").get<std::string>() == "atbash");

    StatusOr<HypothesisRecord> again = HypothesisRecord::from_json(record.value().to_json());
    REQUIRE(again.ok());
    REQUIRE(again.value().to_json() == record.value().to_json());
}

TEST_CASE("HypothesisRecord digest is key-order stable", "[hypothesis][digest]") {
    const nlohmann::json a{
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", 3}}},
    };
    const nlohmann::json b{
        {"params", {{"shift", 3}}},
        {"direction", "decrypt"},
        {"transform_id", "caesar"},
    };
    REQUIRE(HypothesisRecord::method_sha256(a) == HypothesisRecord::method_sha256(b));
    REQUIRE(HypothesisRecord::output_indices_sha256(std::vector<Index29>{I(0), I(1), I(2)}) ==
            HypothesisRecord::output_indices_sha256(std::vector<Index29>{I(0), I(1), I(2)}));
}

TEST_CASE("HypothesisRecord store round-trip under temp workspace", "[hypothesis][store]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_hypothesis_store_test";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    StatusOr<HypothesisRecord> draft =
        HypothesisRecord::make_draft("tmp-ws", "h-draft-1", "2026-09-19T21:00:00Z", "temp draft",
                                     nlohmann::json{
                                         {"transform_id", "atbash"},
                                         {"direction", "decrypt"},
                                         {"params", nlohmann::json::object()},
                                     });
    REQUIRE(draft.ok());
    draft.value().recompute_method_digest();
    REQUIRE(draft.value().digests().at("method_sha256").is_string());

    Status stored = draft.value().store(tmp);
    REQUIRE(stored.ok());

    StatusOr<HypothesisRecord> loaded = HypothesisRecord::load(tmp, "tmp-ws", "h-draft-1");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().title() == "temp draft");
    REQUIRE(loaded.value().status() == HypothesisStatus::Draft);
    REQUIRE(loaded.value().digests().at("method_sha256").get<std::string>() ==
            draft.value().digests().at("method_sha256").get<std::string>());

    REQUIRE(loaded.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE_FALSE(loaded.value().set_status(HypothesisStatus::Draft).ok());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("HypothesisRecord source validates extended provenance fields", "[hypothesis][source]") {
    REQUIRE(HypothesisRecord::validate_source(nullptr).ok());
    REQUIRE(HypothesisRecord::validate_source(HypothesisRecord::empty_source()).ok());

    nlohmann::json batch_source = {
        {"generator_id", "gen_caesar"}, {"candidate_id", "caesar:shift=3"},
        {"agent_run_id", nullptr},      {"batch_id", "b-caesar-0001"},
        {"family", "caesar"},           {"rank", 0},
    };
    REQUIRE(HypothesisRecord::validate_source(batch_source).ok());

    REQUIRE_FALSE(
        HypothesisRecord::validate_source(nlohmann::json{{"generator_id", "gen_x"}, {"extra", 1}})
            .ok());
    REQUIRE_FALSE(HypothesisRecord::validate_source(
                      nlohmann::json{{"batch_id", "BadId"}, {"candidate_id", "x"}})
                      .ok());
    REQUIRE_FALSE(HypothesisRecord::validate_source(
                      nlohmann::json{{"family", "rot13"}, {"candidate_id", "x"}})
                      .ok());
    REQUIRE_FALSE(
        HypothesisRecord::validate_source(nlohmann::json{{"rank", -1}, {"candidate_id", "x"}})
            .ok());
    REQUIRE_FALSE(HypothesisRecord::validate_source(nlohmann::json{{"candidate_id", ""}}).ok());

    StatusOr<HypothesisRecord> draft =
        HypothesisRecord::make_draft("src-ws", "h-src-1", "2026-09-22T12:00:00Z");
    REQUIRE(draft.ok());
    REQUIRE(draft.value().source().contains("batch_id"));
    REQUIRE(draft.value().source().at("batch_id").is_null());
    REQUIRE(draft.value().source().contains("family"));
    REQUIRE(draft.value().source().contains("rank"));

    REQUIRE(draft.value().set_source(batch_source).ok());
    REQUIRE_FALSE(
        draft.value().set_source(nlohmann::json{{"batch_id", "!!"}, {"candidate_id", "x"}}).ok());

    // Round-trip with extended source.
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_hypothesis_source_f21";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);
    REQUIRE(draft.value().store(tmp).ok());
    StatusOr<HypothesisRecord> loaded = HypothesisRecord::load(tmp, "src-ws", "h-src-1");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().source().at("batch_id").get<std::string>() == "b-caesar-0001");
    REQUIRE(loaded.value().source().at("family").get<std::string>() == "caesar");
    REQUIRE(loaded.value().source().at("rank").get<int>() == 0);
    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("HypothesisRecord load rejects id mismatch", "[hypothesis][load]") {
    const auto path =
        data_root() / "workspaces" / "_example" / "hypotheses" / "h-atbash-example.json";
    REQUIRE_FALSE(HypothesisRecord::load_file(path, "_example", "wrong-id").ok());
    REQUIRE_FALSE(HypothesisRecord::load_file(path, "other-ws", "h-atbash-example").ok());
}

TEST_CASE("HypothesisRecord loads falsify-shaped research writer JSON",
          "[hypothesis][research][load]") {
    // Mirrors scripts/research/falsify_nt_keystreams_lp2.py make_hypothesis_record().
    nlohmann::json root = {
        {"schema", "parcae.hypothesis.v0"},
        {"id", "h_nt_stream_a_phi"},
        {"workspace_id", "_falsify_schema_check"},
        {"created_utc", "2026-09-27T00:00:00Z"},
        {"updated_utc", "2026-09-27T00:00:00Z"},
        {"status", "rejected"},
        {"title", "Keystream A: S_t = phi(t) mod 29"},
        {"rationale", "schema self-check"},
        {"method",
         {{"transform_id", "vigenere_key"},
          {"direction", "decrypt"},
          {"params",
           {{"stream_id", "A_phi_pos"},
            {"formula", "S_t = EulerTotient(t) mod 29"},
            {"mixer_ops", {"sub", "add"}},
            {"domains", {"index", "prime_mod_as_index"}},
            {"pos_bases", {1, 0}},
            {"modulus", 29},
            {"research", "nt_keystream_falsify"}}}}},
        {"source",
         {{"generator_id", "research_falsify_nt_keystreams_lp2"},
          {"batch_id", "b_falsify_schema_v0"},
          {"candidate_id", "a_phi_pos"},
          {"family", "vigenere"},
          {"agent_run_id", nullptr},
          {"rank", nullptr}}},
        {"preview", {{"latin_prefix", "def"}, {"max_chars", 96}}},
        {"scores",
         {{{"score_id", "chi2_english_gp_v0"},
           {"score_version", "v0"},
           {"value", 90.0},
           {"backend", "cpu"},
           {"scored_utc", "2026-09-27T00:00:00Z"}}}},
        {"digests", {{"method_sha256", nullptr}, {"output_indices_sha256", nullptr}}},
        {"promotion",
         {{"notes", "Research falsification only. Tools MUST NOT auto-write fixtures."},
          {"target_fixture_id", nullptr}}},
    };

    StatusOr<HypothesisRecord> parsed = HypothesisRecord::from_json(root);
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value().status() == HypothesisStatus::Rejected);
    REQUIRE(parsed.value().method().at("transform_id").get<std::string>() == "vigenere_key");
    REQUIRE(parsed.value().source().at("family").get<std::string>() == "vigenere");
    REQUIRE(parsed.value().scores().size() == 1);
    REQUIRE(parsed.value().scores()[0].at("backend").get<std::string>() == "cpu");
    REQUIRE(parsed.value().scores()[0].at("scored_utc").get<std::string>() ==
            "2026-09-27T00:00:00Z");

    const auto tmp = std::filesystem::temp_directory_path() / "parcae_falsify_schema_hyp";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);
    REQUIRE(parsed.value().store(tmp).ok());
    StatusOr<HypothesisRecord> loaded =
        HypothesisRecord::load(tmp, "_falsify_schema_check", "h_nt_stream_a_phi");
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().id() == "h_nt_stream_a_phi");
    std::filesystem::remove_all(tmp, ec);
}
