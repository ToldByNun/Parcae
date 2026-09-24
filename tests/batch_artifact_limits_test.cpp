#include <parcae/batch/batch_hit.hpp>
#include <parcae/batch/batch_ordering.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/sha256.hpp>
#include <parcae/generate/transform_candidate.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/score/score_order.hpp>
#include <parcae/search/batch_artifact.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

constexpr std::uint32_t k_ordering_fuzz_seed = 0x00B47Cu;

[[nodiscard]] std::string fake_digest(std::string_view label) {
    return Sha256::hex_digest(label);
}

[[nodiscard]] TransformCandidate caesar_candidate(std::uint8_t shift) {
    StatusOr<TransformId> id = TransformId::from_string("caesar");
    REQUIRE(id.ok());
    return TransformCandidate(
        "caesar-shift-" + std::to_string(shift),
        id.value(),
        TransformDirection::Decrypt,
        nlohmann::json{{"shift", shift}},
        {Index29{19}, Index29{7}, Index29{4}});
}

[[nodiscard]] nlohmann::json wire_line(
    std::uint8_t shift,
    double score,
    std::size_t rank,
    std::string_view score_id = "chi2_english_gp_v0") {
    return BatchArtifact::candidate_wire(
        caesar_candidate(shift),
        score_id,
        "v0",
        score,
        Backend::Cpu,
        rank);
}

[[nodiscard]] StatusOr<BatchArtifact> make_batch(
    std::vector<nlohmann::json> lines,
    std::size_t k,
    std::string_view score_id = "chi2_english_gp_v0") {
    return BatchArtifact::make(
        "batch-ws",
        "b-l45-limits",
        "2026-09-23T07:00:00Z",
        fake_digest("job-l45"),
        fake_digest("prior-l45"),
        "caesar",
        score_id,
        "v0",
        Backend::Cpu,
        k,
        1,
        std::move(lines));
}

}  // namespace

TEST_CASE(
    "L45 BatchArtifact rejects count > k and oversized JSONL lines",
    "[search][batch][limits]") {
    {
        // Three lines but k=2.
        std::vector<nlohmann::json> lines = {
            wire_line(1, 1.0, 0),
            wire_line(2, 2.0, 1),
            wire_line(3, 3.0, 2),
        };
        REQUIRE_FALSE(make_batch(std::move(lines), 2).ok());
    }
    {
        // Oversized line: inflate output_indices past kMaxCandidateLineBytes.
        nlohmann::json fat = wire_line(1, 1.0, 0);
        nlohmann::json indices = nlohmann::json::array();
        // Each "0," ~2 bytes; need dump > 256 KiB. Use many indices.
        const std::size_t n =
            (BatchArtifact::kMaxCandidateLineBytes / 2) + 64;
        indices.get_ptr<nlohmann::json::array_t*>()->reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            indices.push_back(static_cast<int>(i % 29));
        }
        fat["output_indices"] = std::move(indices);
        REQUIRE(fat.dump().size() > BatchArtifact::kMaxCandidateLineBytes);
        REQUIRE_FALSE(make_batch({fat}, 1).ok());
    }
    {
        // Load path also rejects oversized lines before parse.
        const auto tmp =
            std::filesystem::temp_directory_path() / "parcae_batch_l45_fat_line";
        std::error_code ec;
        std::filesystem::remove_all(tmp, ec);
        std::filesystem::create_directories(tmp / "workspaces", ec);

        nlohmann::json ok_line = wire_line(1, 1.0, 0);
        StatusOr<BatchArtifact> art = make_batch({ok_line}, 1);
        REQUIRE(art.ok());
        REQUIRE(art.value().store(tmp).ok());

        StatusOr<std::filesystem::path> dir =
            WorkspacePaths::batch_dir(tmp, "batch-ws", "b-l45-limits");
        REQUIRE(dir.ok());
        {
            std::ofstream out(
                dir.value() / "candidates.jsonl", std::ios::binary | std::ios::trunc);
            REQUIRE(out);
            out << std::string(BatchArtifact::kMaxCandidateLineBytes + 8, 'x') << '\n';
        }
        REQUIRE_FALSE(BatchArtifact::load(tmp, "batch-ws", "b-l45-limits").ok());
        std::filesystem::remove_all(tmp, ec);
    }
}

TEST_CASE(
    "L45 BatchArtifact rejects inverted best-first score order",
    "[search][batch][limits][ordering]") {
    // chi2 is Asc — lower score must rank first.
    {
        std::vector<nlohmann::json> bad = {
            wire_line(1, 50.0, 0),
            wire_line(2, 10.0, 1),
        };
        REQUIRE_FALSE(make_batch(std::move(bad), 2).ok());
    }
    {
        std::vector<nlohmann::json> good = {
            wire_line(1, 10.0, 0),
            wire_line(2, 50.0, 1),
        };
        REQUIRE(make_batch(std::move(good), 2).ok());
    }
    // ic_mod29 is Desc — higher score must rank first.
    {
        std::vector<nlohmann::json> bad = {
            wire_line(1, 0.1, 0, "ic_mod29"),
            wire_line(2, 0.9, 1, "ic_mod29"),
        };
        REQUIRE_FALSE(make_batch(std::move(bad), 2, "ic_mod29").ok());
    }
    {
        std::vector<nlohmann::json> good = {
            wire_line(1, 0.9, 0, "ic_mod29"),
            wire_line(2, 0.1, 1, "ic_mod29"),
        };
        REQUIRE(make_batch(std::move(good), 2, "ic_mod29").ok());
    }
    // Equal scores: lexicographically smaller candidate_id wins (Asc/Desc same).
    {
        // caesar-shift-1 < caesar-shift-2
        std::vector<nlohmann::json> good = {
            wire_line(1, 5.0, 0),
            wire_line(2, 5.0, 1),
        };
        REQUIRE(make_batch(std::move(good), 2).ok());
    }
    {
        std::vector<nlohmann::json> bad = {
            wire_line(2, 5.0, 0),
            wire_line(1, 5.0, 1),
        };
        REQUIRE_FALSE(make_batch(std::move(bad), 2).ok());
    }
}

TEST_CASE(
    "L45 BatchOrdering shuffle+sort is deterministic (fuzz)",
    "[search][batch][fuzz][ordering]") {
    std::mt19937 rng(k_ordering_fuzz_seed);
    std::uniform_real_distribution<double> score_dist(0.0, 100.0);
    std::uniform_int_distribution<int> id_dist(0, 40);
    std::uniform_int_distribution<int> idx_dist(0, 200);

    for (ScoreOrder order : {ScoreOrder::Asc, ScoreOrder::Desc}) {
        for (int trial = 0; trial < 48; ++trial) {
            std::vector<BatchHit> hits;
            hits.reserve(64);
            for (int i = 0; i < 64; ++i) {
                hits.emplace_back(
                    "id-" + std::to_string(id_dist(rng)),
                    score_dist(rng),
                    static_cast<std::size_t>(idx_dist(rng)));
            }

            std::vector<BatchHit> a = hits;
            std::vector<BatchHit> b = hits;
            std::shuffle(a.begin(), a.end(), rng);
            std::shuffle(b.begin(), b.end(), rng);

            std::sort(a.begin(), a.end(), BatchOrdering::BestFirst(order));
            std::sort(b.begin(), b.end(), BatchOrdering::BestFirst(order));

            REQUIRE(a.size() == b.size());
            for (std::size_t i = 0; i < a.size(); ++i) {
                REQUIRE(a[i].candidate_id() == b[i].candidate_id());
                REQUIRE(a[i].score() == b[i].score());
                REQUIRE(a[i].source_index() == b[i].source_index());
            }

            // Adjacent pairs must not invert under the total order.
            for (std::size_t i = 1; i < a.size(); ++i) {
                REQUIRE_FALSE(BatchOrdering::better(a[i], a[i - 1], order));
            }
        }
    }
}

TEST_CASE(
    "L45 BatchArtifact lines from BatchOrdering sort round-trip",
    "[search][batch][fuzz]") {
    std::mt19937 rng(k_ordering_fuzz_seed ^ 0x5EED);
    std::uniform_real_distribution<double> score_dist(0.0, 40.0);

    std::vector<BatchHit> hits;
    for (std::uint8_t shift = 0; shift < 16; ++shift) {
        hits.emplace_back(
            "caesar-shift-" + std::to_string(shift),
            score_dist(rng),
            static_cast<std::size_t>(shift));
    }
    std::shuffle(hits.begin(), hits.end(), rng);
    std::sort(hits.begin(), hits.end(), BatchOrdering::BestFirst(ScoreOrder::Asc));

    std::vector<nlohmann::json> lines;
    lines.reserve(hits.size());
    for (std::size_t i = 0; i < hits.size(); ++i) {
        // Recover shift from source_index used above.
        const auto shift = static_cast<std::uint8_t>(hits[i].source_index());
        lines.push_back(wire_line(shift, hits[i].score(), i));
    }
    StatusOr<BatchArtifact> art = make_batch(std::move(lines), hits.size());
    REQUIRE(art.ok());
    REQUIRE(art.value().candidate_count() == hits.size());

    // Corrupt: swap two score-adjacent ranks' wire order without fixing ranks.
    if (hits.size() >= 2) {
        std::vector<nlohmann::json> inverted;
        inverted.push_back(wire_line(
            static_cast<std::uint8_t>(hits[1].source_index()), hits[1].score(), 0));
        inverted.push_back(wire_line(
            static_cast<std::uint8_t>(hits[0].source_index()), hits[0].score(), 1));
        // Only fail when the swap actually violates Asc order.
        if (BatchOrdering::better(
                BatchHit{hits[1].candidate_id(), hits[1].score(), 0},
                BatchHit{hits[0].candidate_id(), hits[0].score(), 1},
                ScoreOrder::Asc)) {
            REQUIRE_FALSE(make_batch(std::move(inverted), 2).ok());
        }
    }
}
