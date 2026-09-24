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
