#include <parcae/core/status_or.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/search_prior.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

[[nodiscard]] nlohmann::json caesar_method(int shift) {
    return nlohmann::json{
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", shift}}},
    };
}

}  // namespace

TEST_CASE("SearchPrior param_hash is key-order stable", "[search][prior]") {
    const nlohmann::json a{{"shift", 3}, {"note", "x"}};
    const nlohmann::json b{{"note", "x"}, {"shift", 3}};
    REQUIRE(SearchPrior::param_hash_of(a) == SearchPrior::param_hash_of(b));
    REQUIRE(SearchPrior::param_hash_of(a).rfind("sha256:", 0) == 0);
    REQUIRE(SearchPrior::param_hash_of(a).size() == 7 + 64);
}

TEST_CASE("SearchPrior make / to_json round-trip", "[search][prior]") {
    const std::string hash = SearchPrior::param_hash_of(nlohmann::json{{"shift", 7}});
    StatusOr<SearchPrior> prior = SearchPrior::make(
        "_example",
        {
            SearchPrior::Seed(
                "h-caesar-3",
                nlohmann::json{
                    {"transform_id", "caesar"},
                    {"direction", "decrypt"},
                    {"params", {{"shift", 3}}},
                }),
        },
        {
            SearchPrior::Exclusion(hash, "h-caesar-7", "rejected"),
        },
        "2026-09-21T18:00:00Z");
    REQUIRE(prior.ok());
    REQUIRE(prior.value().workspace_id() == "_example");
    REQUIRE(prior.value().seeds().size() == 1);
    REQUIRE(prior.value().exclusions().size() == 1);
    REQUIRE(prior.value().seeds()[0].hypothesis_id() == "h-caesar-3");
    REQUIRE(prior.value().excludes_params(nlohmann::json{{"shift", 7}}));
    REQUIRE_FALSE(prior.value().excludes_params(nlohmann::json{{"shift", 3}}));

    StatusOr<SearchPrior> again = SearchPrior::from_json(prior.value().to_json());
    REQUIRE(again.ok());
    REQUIRE(again.value().to_json() == prior.value().to_json());
    REQUIRE(again.value().prior_digest_sha256() == prior.value().prior_digest_sha256());
}

TEST_CASE("SearchPrior from_json rejects soft weights and bad schema", "[search][prior]") {
    nlohmann::json valid{
        {"schema", "parcae.search_prior.v0"},
        {"workspace_id", "_example"},
        {"seeds", nlohmann::json::array()},
        {"exclusions", nlohmann::json::array()},
        {"built_utc", "2026-09-21T18:00:00Z"},
    };
    REQUIRE(SearchPrior::from_json(valid).ok());

    {
        nlohmann::json j = valid;
        j["schema"] = "nope";
        REQUIRE_FALSE(SearchPrior::from_json(j).ok());
    }
    {
        nlohmann::json j = valid;
        j["weights"] = nlohmann::json::object();
        REQUIRE_FALSE(SearchPrior::from_json(j).ok());
    }
    {
        nlohmann::json j = valid;
        j["exclusions"] = nlohmann::json::array({
            {
                {"param_hash", "not-a-hash"},
                {"hypothesis_id", "h-x"},
                {"reason", "rejected"},
            },
        });
        REQUIRE_FALSE(SearchPrior::from_json(j).ok());
    }
}

TEST_CASE("SearchPrior from_workspace maps promoted and rejected", "[search][prior]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_search_prior_b6";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    StatusOr<HypothesisRecord> promoted = HypothesisRecord::make_draft(
        "prior-ws",
        "h-promoted",
        "2026-09-21T18:00:00Z",
        "promoted seed",
        caesar_method(3));
    REQUIRE(promoted.ok());
    REQUIRE(promoted.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(promoted.value().set_status(HypothesisStatus::Promoted).ok());
    REQUIRE(promoted.value().store(tmp).ok());

    StatusOr<HypothesisRecord> rejected = HypothesisRecord::make_draft(
        "prior-ws",
        "h-rejected",
        "2026-09-21T18:00:00Z",
        "rejected exclusion",
        caesar_method(7));
    REQUIRE(rejected.ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(rejected.value().set_status(HypothesisStatus::Rejected).ok());
    REQUIRE(rejected.value().store(tmp).ok());

    StatusOr<HypothesisRecord> proposed = HypothesisRecord::make_draft(
        "prior-ws",
        "h-proposed",
        "2026-09-21T18:00:00Z",
        "ignored proposed",
        caesar_method(5));
    REQUIRE(proposed.ok());
    REQUIRE(proposed.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(proposed.value().store(tmp).ok());

    StatusOr<SearchPrior> prior =
        SearchPrior::from_workspace(tmp, "prior-ws", "2026-09-21T18:01:00Z");
    REQUIRE(prior.ok());
    REQUIRE(prior.value().seeds().size() == 1);
    REQUIRE(prior.value().seeds()[0].hypothesis_id() == "h-promoted");
    REQUIRE(prior.value().exclusions().size() == 1);
    REQUIRE(prior.value().exclusions()[0].hypothesis_id() == "h-rejected");
    REQUIRE(prior.value().exclusions()[0].reason() == "rejected");
    REQUIRE(prior.value().excludes_params(nlohmann::json{{"shift", 7}}));
    REQUIRE_FALSE(prior.value().excludes_params(nlohmann::json{{"shift", 5}}));

    StatusOr<HypothesisRecord> scored = HypothesisRecord::make_draft(
        "prior-ws",
        "h-scored",
        "2026-09-21T18:00:00Z",
        "optional scored seed",
        caesar_method(11));
    REQUIRE(scored.ok());
    REQUIRE(scored.value().set_status(HypothesisStatus::Proposed).ok());
    REQUIRE(scored.value().set_status(HypothesisStatus::Scored).ok());
    REQUIRE(scored.value().store(tmp).ok());

    SearchPrior::BuildOptions opts;
    opts.include_scored_as_seeds = true;
    StatusOr<SearchPrior> with_scored =
        SearchPrior::from_workspace(tmp, "prior-ws", "2026-09-21T18:02:00Z", opts);
    REQUIRE(with_scored.ok());
    REQUIRE(with_scored.value().seeds().size() == 2);
    REQUIRE(with_scored.value().seeds()[0].hypothesis_id() == "h-promoted");
    REQUIRE(with_scored.value().seeds()[1].hypothesis_id() == "h-scored");

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("SearchPrior from_workspace on _example has no seeds", "[search][prior]") {
    StatusOr<SearchPrior> prior =
        SearchPrior::from_workspace(data_root(), "_example", "2026-09-21T18:00:00Z");
    REQUIRE(prior.ok());
    REQUIRE(prior.value().seeds().empty());
    REQUIRE(prior.value().exclusions().empty());
    REQUIRE(prior.value().workspace_id() == "_example");
}

TEST_CASE("SearchPrior load_file round-trip", "[search][prior]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_search_prior_file_b6";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp, ec);

    StatusOr<SearchPrior> prior = SearchPrior::make(
        "_example",
        {},
        {},
        "2026-09-21T18:00:00Z");
    REQUIRE(prior.ok());
    const auto path = tmp / "prior.json";
    {
        std::ofstream out(path);
        REQUIRE(out);
        out << prior.value().to_json().dump(2);
    }
    StatusOr<SearchPrior> loaded = SearchPrior::load_file(path);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().prior_digest_sha256() == prior.value().prior_digest_sha256());

    std::filesystem::remove_all(tmp, ec);
}
