#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <parcae/core/index29.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/hypothesis/hypothesis_record.hpp>
#include <parcae/hypothesis/hypothesis_status.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_prior.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <string>
#include <vector>

namespace {

[[nodiscard]] nlohmann::json caesar_method(int shift) {
    return nlohmann::json{
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", shift}}},
    };
}

[[nodiscard]] TransformCandidate caesar_candidate(std::uint8_t shift) {
    StatusOr<TransformId> id = TransformId::from_string("caesar");
    REQUIRE(id.ok());
    return TransformCandidate("caesar-shift-" + std::to_string(shift), id.value(),
                              TransformDirection::Decrypt, nlohmann::json{{"shift", shift}},
                              {Index29{19}, Index29{7}, Index29{4}});
}

[[nodiscard]] StatusOr<HypothesisRecord> store_status(const std::filesystem::path& data_root,
                                                      std::string_view workspace_id,
                                                      std::string_view hypothesis_id, int shift,
                                                      HypothesisStatus terminal) {
    StatusOr<HypothesisRecord> draft =
        HypothesisRecord::make_draft(workspace_id, hypothesis_id, "2026-09-21T19:00:00Z",
                                     std::string(hypothesis_id), caesar_method(shift));
    if (!draft.ok()) {
        return draft.status();
    }
    Status proposed = draft.value().set_status(HypothesisStatus::Proposed);
    if (!proposed.ok()) {
        return proposed;
    }
    if (terminal == HypothesisStatus::Scored || terminal == HypothesisStatus::Rejected ||
        terminal == HypothesisStatus::Promoted) {
        // Proposed → Scored → Rejected/Promoted, or Proposed → Rejected/Promoted.
        if (terminal == HypothesisStatus::Scored) {
            Status scored = draft.value().set_status(HypothesisStatus::Scored);
            if (!scored.ok()) {
                return scored;
            }
        } else {
            Status done = draft.value().set_status(terminal);
            if (!done.ok()) {
                return done;
            }
        }
    }
    Status stored = draft.value().store(data_root);
    if (!stored.ok()) {
        return stored;
    }
    return draft;
}

} // namespace

TEST_CASE("SearchJob + SearchPrior + BatchArtifact end-to-end round-trip", "[search][roundtrip]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_search_roundtrip_b8";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "workspaces", ec);

    REQUIRE(store_status(tmp, "rt-ws", "h-seed", 3, HypothesisStatus::Promoted).ok());
    REQUIRE(store_status(tmp, "rt-ws", "h-excl", 7, HypothesisStatus::Rejected).ok());
    REQUIRE(store_status(tmp, "rt-ws", "h-noise", 5, HypothesisStatus::Proposed).ok());

    StatusOr<SearchPrior> prior = SearchPrior::from_workspace(tmp, "rt-ws", "2026-09-21T19:30:00Z");
    REQUIRE(prior.ok());
    REQUIRE(prior.value().seeds().size() == 1);
    REQUIRE(prior.value().seeds()[0].hypothesis_id() == "h-seed");
    REQUIRE(prior.value().exclusions().size() == 1);
    REQUIRE(prior.value().excludes_params(nlohmann::json{{"shift", 7}}));
    REQUIRE_FALSE(prior.value().excludes_params(nlohmann::json{{"shift", 3}}));

    // Prior JSON file round-trip (workspace-derived → disk → parse).
    const auto prior_path = tmp / "workspaces" / "rt-ws" / "prior.json";
    {
        std::ofstream out(prior_path);
        REQUIRE(out);
        out << prior.value().to_json().dump(2);
    }
    StatusOr<SearchPrior> prior_loaded = SearchPrior::load_file(prior_path);
    REQUIRE(prior_loaded.ok());
    REQUIRE(prior_loaded.value().prior_digest_sha256() == prior.value().prior_digest_sha256());

    StatusOr<SearchJob> job =
        SearchJob::make("rt-ws", "caesar", "chi2_english_gp_v0", 2, 42, Backend::Cpu, 128,
                        TransformDirection::Decrypt, nlohmann::json::object(),
                        prior.value().to_json(), "v0", false);
    REQUIRE(job.ok());
    REQUIRE(job.value().prior().has_value());
    REQUIRE(job.value().require_workspace_dir(tmp).ok());

    // Job JSON file round-trip preserves digests and inline prior.
    const auto job_path = tmp / "workspaces" / "rt-ws" / "job.json";
    {
        std::ofstream out(job_path);
        REQUIRE(out);
        out << job.value().to_json().dump(2);
    }
    StatusOr<SearchJob> job_loaded = SearchJob::load_file(job_path);
    REQUIRE(job_loaded.ok());
    REQUIRE(job_loaded.value().job_digest_sha256() == job.value().job_digest_sha256());
    REQUIRE(job_loaded.value().prior().has_value());
    StatusOr<SearchPrior> prior_from_job =
        SearchPrior::from_json(job_loaded.value().prior().value());
    REQUIRE(prior_from_job.ok());
    REQUIRE(prior_from_job.value().prior_digest_sha256() == prior.value().prior_digest_sha256());

    // Candidate list respects prior: keep seed shift=3, drop excluded shift=7.
    std::vector<nlohmann::json> lines;
    const auto seed_cand = caesar_candidate(3);
    const auto excl_cand = caesar_candidate(7);
    const auto other_cand = caesar_candidate(11);
    REQUIRE_FALSE(prior.value().excludes_envelope(seed_cand.envelope()));
    REQUIRE(prior.value().excludes_envelope(excl_cand.envelope()));
    REQUIRE_FALSE(prior.value().excludes_envelope(other_cand.envelope()));

    lines.push_back(BatchArtifact::candidate_wire(seed_cand, job.value().score_id(),
                                                  job.value().score_version(), 9.5,
                                                  job.value().backend(), 0));
    lines.push_back(BatchArtifact::candidate_wire(other_cand, job.value().score_id(),
                                                  job.value().score_version(), 15.0,
                                                  job.value().backend(), 1));

    StatusOr<BatchArtifact> art = BatchArtifact::make(
        job.value().workspace_id(), "b-rt-20260921-0001", "2026-09-21T19:31:00Z",
        job.value().job_digest_sha256(), prior.value().prior_digest_sha256(), job.value().family(),
        job.value().score_id(), job.value().score_version(), job.value().backend(), job.value().k(),
        job.value().seed(), lines,
        nlohmann::json{
            {"schema", "parcae.batch_report.v0"},
            {"seed_hypothesis_id", "h-seed"},
            {"excluded_hypothesis_id", "h-excl"},
        });
    REQUIRE(art.ok());
    REQUIRE(art.value().store(tmp).ok());

    StatusOr<BatchArtifact> art_loaded = BatchArtifact::load(tmp, "rt-ws", "b-rt-20260921-0001");
    REQUIRE(art_loaded.ok());
    REQUIRE(art_loaded.value().job_digest_sha256() == job.value().job_digest_sha256());
    REQUIRE(art_loaded.value().prior_digest_sha256() == prior.value().prior_digest_sha256());
    REQUIRE(art_loaded.value().candidate_count() == 2);
    REQUIRE(art_loaded.value().candidates()[0].at("candidate_id") == "caesar-shift-3");
    REQUIRE(art_loaded.value().candidates()[1].at("candidate_id") == "caesar-shift-11");
    REQUIRE(art_loaded.value().manifest_to_json() == art.value().manifest_to_json());

    // Re-parse job from digest-bearing artifact fields still matches.
    StatusOr<SearchJob> job_again = SearchJob::from_json(job_loaded.value().to_json());
    REQUIRE(job_again.ok());
    REQUIRE(job_again.value().job_digest_sha256() == art_loaded.value().job_digest_sha256());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("SearchJob inline prior rejects soft weights; digests stay key-order stable",
          "[search][roundtrip]") {
    StatusOr<SearchPrior> prior =
        SearchPrior::make("_example",
                          {
                              SearchPrior::Seed("h-caesar-3",
                                                nlohmann::json{
                                                    {"transform_id", "caesar"},
                                                    {"direction", "decrypt"},
                                                    {"params", {{"shift", 3}}},
                                                }),
                          },
                          {}, "2026-09-21T19:00:00Z");
    REQUIRE(prior.ok());

    nlohmann::json prior_a = prior.value().to_json();
    nlohmann::json prior_b{
        {"built_utc", "2026-09-21T19:00:00Z"}, {"exclusions", nlohmann::json::array()},
        {"seeds", prior_a.at("seeds")},        {"workspace_id", "_example"},
        {"schema", "parcae.search_prior.v0"},
    };
    StatusOr<SearchPrior> pa = SearchPrior::from_json(prior_a);
    StatusOr<SearchPrior> pb = SearchPrior::from_json(prior_b);
    REQUIRE(pa.ok());
    REQUIRE(pb.ok());
    REQUIRE(pa.value().prior_digest_sha256() == pb.value().prior_digest_sha256());

    nlohmann::json bad_prior = prior_a;
    bad_prior["weights"] = nlohmann::json{{"h-caesar-3", 0.5}};
    REQUIRE_FALSE(SearchJob::make("_example", "caesar", "chi2_english_gp_v0", 1, 1, Backend::Cpu, 1,
                                  TransformDirection::Decrypt, nlohmann::json::object(), bad_prior)
                      .ok());

    StatusOr<SearchJob> job_a =
        SearchJob::make("_example", "caesar", "chi2_english_gp_v0", 1, 1, Backend::Cpu, 1,
                        TransformDirection::Decrypt, nlohmann::json::object(), prior_a);
    StatusOr<SearchJob> job_b =
        SearchJob::make("_example", "caesar", "chi2_english_gp_v0", 1, 1, Backend::Cpu, 1,
                        TransformDirection::Decrypt, nlohmann::json::object(), prior_b);
    REQUIRE(job_a.ok());
    REQUIRE(job_b.ok());
    REQUIRE(job_a.value().job_digest_sha256() == job_b.value().job_digest_sha256());
}
