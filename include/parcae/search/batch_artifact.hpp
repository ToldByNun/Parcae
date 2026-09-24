#ifndef BATCH_ARTIFACT_HPP
#define BATCH_ARTIFACT_HPP

#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/search/search_job.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/tool/transform_envelope.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

/// On-disk top-k batch under `workspaces/<id>/batches/<batch_id>/`
/// (`parcae.batch_artifact.v0` — docs/spec/search-loop.md).
class BatchArtifact {
public:
    static constexpr std::string_view schema_id = "parcae.batch_artifact.v0";
    static constexpr std::string_view ordering_id = "batch_ordering_v0";
    static constexpr std::string_view default_candidates_file = "candidates.jsonl";
    static constexpr std::string_view default_report_file = "report.json";

    /// Absolute ceiling on `candidates.jsonl` lines (in addition to `count <= k`).
    static constexpr std::size_t kMaxCandidatesPerBatch = 65536;
    /// Max UTF-8 bytes per JSONL line (search-loop.md § Limits).
    static constexpr std::size_t kMaxCandidateLineBytes = 256 * 1024;

    BatchArtifact() = default;

    [[nodiscard]] const std::string& batch_id() const noexcept { return batch_id_; }

    [[nodiscard]] const std::string& workspace_id() const noexcept { return workspace_id_; }

    [[nodiscard]] const std::string& created_utc() const noexcept { return created_utc_; }

    [[nodiscard]] const std::string& job_digest_sha256() const noexcept {
        return job_digest_sha256_;
    }

    [[nodiscard]] const std::string& prior_digest_sha256() const noexcept {
        return prior_digest_sha256_;
    }

    [[nodiscard]] const std::string& family() const noexcept { return family_; }

    [[nodiscard]] const std::string& score_id() const noexcept { return score_id_; }

    [[nodiscard]] const std::string& score_version() const noexcept { return score_version_; }

    [[nodiscard]] Backend backend() const noexcept { return backend_; }

    [[nodiscard]] std::size_t k() const noexcept { return k_; }

    [[nodiscard]] std::size_t candidate_count() const noexcept { return candidates_.size(); }

    [[nodiscard]] std::uint32_t seed() const noexcept { return seed_; }

    [[nodiscard]] const std::string& candidates_relpath() const noexcept {
        return candidates_relpath_;
    }

    /// Empty when report omitted (`paths.report` null).
    [[nodiscard]] const std::optional<std::string>& report_relpath() const noexcept {
        return report_relpath_;
    }

    [[nodiscard]] const std::vector<nlohmann::json>& candidates() const noexcept {
        return candidates_;
    }

    [[nodiscard]] const std::optional<nlohmann::json>& report() const noexcept { return report_; }

    /// Wire object for one `candidates.jsonl` line (best-first `rank`).
    [[nodiscard]] static nlohmann::json candidate_wire(const TransformCandidate& candidate,
                                                       std::string_view score_id,
                                                       std::string_view score_version,
                                                       double score_value, Backend backend,
                                                       std::size_t rank) {
        nlohmann::json base = candidate.to_json();
        base["rank"] = rank;
        base["score"] = nlohmann::json{
            {"score_id", std::string(score_id)},
            {"score_version", std::string(score_version)},
            {"value", score_value},
            {"backend", std::string(BackendUtil::to_string(backend))},
        };
        return base;
    }

    [[nodiscard]] static StatusOr<BatchArtifact>
    make(std::string_view workspace_id, std::string_view batch_id, std::string_view created_utc,
         std::string_view job_digest_sha256, std::string_view prior_digest_sha256,
         std::string_view family, std::string_view score_id, std::string_view score_version,
         Backend backend, std::size_t k, std::uint32_t seed, std::vector<nlohmann::json> candidates,
         std::optional<nlohmann::json> report = std::nullopt) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        StatusOr<std::string> bid = WorkspacePaths::validate_id(batch_id);
        if (!bid.ok()) {
            return bid.status();
        }
        if (created_utc.empty()) {
            return Status::error("BatchArtifact.created_utc must be non-empty");
        }
        Status digest_ok = require_sha256_hex("job_digest_sha256", job_digest_sha256);
        if (!digest_ok.ok()) {
            return digest_ok;
        }
        digest_ok = require_sha256_hex("prior_digest_sha256", prior_digest_sha256);
        if (!digest_ok.ok()) {
            return digest_ok;
        }
        if (!SearchJob::is_v0_family(family) && !SearchJob::is_extended_family(family) &&
            !SearchJob::is_theory_family(family)) {
            return Status::error(
                "BatchArtifact.family unknown (expected caesar|atbash|atbash_caesar|"
                "affine|vigenere|compose|beaufort|totient|theory): " +
                std::string(family));
        }
        StatusOr<std::string> sid = SearchJob::validate_score_id(score_id);
        if (!sid.ok()) {
            return sid.status();
        }
        if (score_version.empty() || score_version.size() > 32) {
            return Status::error("BatchArtifact.score_version length must be 1..32");
        }
        if (k == 0) {
            return Status::error("BatchArtifact.k must be >= 1");
        }
        if (candidates.size() > k) {
            return Status::error("BatchArtifact.candidate_count must be <= k");
        }
        if (candidates.size() > kMaxCandidatesPerBatch) {
            return Status::error("BatchArtifact.candidate_count exceeds kMaxCandidatesPerBatch (" +
                                 std::to_string(kMaxCandidatesPerBatch) + ")");
        }
        if (report.has_value()) {
            Status report_ok = validate_report(*report);
            if (!report_ok.ok()) {
                return report_ok;
            }
        }

        Status candidates_ok = validate_candidates(candidates, sid.value(), score_version, backend);
        if (!candidates_ok.ok()) {
            return candidates_ok;
        }

        BatchArtifact art;
        art.workspace_id_ = std::move(wid.value());
        art.batch_id_ = std::move(bid.value());
        art.created_utc_ = std::string(created_utc);
        art.job_digest_sha256_ = std::string(job_digest_sha256);
        art.prior_digest_sha256_ = std::string(prior_digest_sha256);
        art.family_ = std::string(family);
        art.score_id_ = std::move(sid.value());
        art.score_version_ = std::string(score_version);
        art.backend_ = backend;
        art.k_ = k;
        art.seed_ = seed;
        art.candidates_relpath_ = std::string(default_candidates_file);
        if (report.has_value()) {
            art.report_relpath_ = std::string(default_report_file);
            art.report_ = std::move(report);
        }
        art.candidates_ = std::move(candidates);
        return art;
    }

    [[nodiscard]] static StatusOr<BatchArtifact>
    from_manifest_json(const nlohmann::json& root, std::vector<nlohmann::json> candidates,
                       std::optional<nlohmann::json> report = std::nullopt) {
        if (!root.is_object()) {
            return Status::error("BatchArtifact manifest must be an object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error("BatchArtifact.schema must be \"" + std::string(schema_id) + "\"");
        }
        if (!root.contains("ordering") || !root.at("ordering").is_string() ||
            root.at("ordering").get<std::string>() != ordering_id) {
            return Status::error("BatchArtifact.ordering must be \"" + std::string(ordering_id) +
                                 "\"");
        }

        StatusOr<std::string> workspace_id = require_string(root, "workspace_id");
        if (!workspace_id.ok()) {
            return workspace_id.status();
        }
        StatusOr<std::string> batch_id = require_string(root, "batch_id");
        if (!batch_id.ok()) {
            return batch_id.status();
        }
        StatusOr<std::string> created_utc = require_string(root, "created_utc");
        if (!created_utc.ok()) {
            return created_utc.status();
        }
        StatusOr<std::string> job_digest = require_string(root, "job_digest_sha256");
        if (!job_digest.ok()) {
            return job_digest.status();
        }
        StatusOr<std::string> prior_digest = require_string(root, "prior_digest_sha256");
        if (!prior_digest.ok()) {
            return prior_digest.status();
        }
        StatusOr<std::string> family = require_string(root, "family");
        if (!family.ok()) {
            return family.status();
        }
        StatusOr<std::string> score_id = require_string(root, "score_id");
        if (!score_id.ok()) {
            return score_id.status();
        }
        StatusOr<std::string> score_version = require_string(root, "score_version");
        if (!score_version.ok()) {
            return score_version.status();
        }
        if (!root.contains("backend") || !root.at("backend").is_string()) {
            return Status::error("BatchArtifact.backend must be a string");
        }
        StatusOr<Backend> backend = BackendUtil::from_string(root.at("backend").get<std::string>());
        if (!backend.ok()) {
            return backend.status();
        }

        if (!root.contains("k") || !root.at("k").is_number_integer()) {
            return Status::error("BatchArtifact.k must be an integer");
        }
        if (!root.contains("seed") || !root.at("seed").is_number_integer()) {
            return Status::error("BatchArtifact.seed must be an integer");
        }
        if (!root.contains("candidate_count") || !root.at("candidate_count").is_number_integer()) {
            return Status::error("BatchArtifact.candidate_count must be an integer");
        }
        const auto k_i = root.at("k").get<std::int64_t>();
        const auto seed_i = root.at("seed").get<std::int64_t>();
        const auto count_i = root.at("candidate_count").get<std::int64_t>();
        if (k_i < 1) {
            return Status::error("BatchArtifact.k must be >= 1");
        }
        if (seed_i < 0 || seed_i > 0xFFFFFFFFll) {
            return Status::error("BatchArtifact.seed must fit in uint32");
        }
        if (count_i < 0) {
            return Status::error("BatchArtifact.candidate_count must be >= 0");
        }
        if (static_cast<std::size_t>(count_i) != candidates.size()) {
            return Status::error(
                "BatchArtifact.candidate_count does not match candidates.jsonl line count");
        }

        if (!root.contains("paths") || !root.at("paths").is_object()) {
            return Status::error("BatchArtifact.paths must be an object");
        }
        const nlohmann::json& paths = root.at("paths");
        if (!paths.contains("candidates") || !paths.at("candidates").is_string()) {
            return Status::error("BatchArtifact.paths.candidates must be a string");
        }
        StatusOr<std::string> candidates_rel =
            validate_rel_filename(paths.at("candidates").get<std::string>(), "candidates");
        if (!candidates_rel.ok()) {
            return candidates_rel.status();
        }

        std::optional<std::string> report_rel;
        if (!paths.contains("report") || paths.at("report").is_null()) {
            if (report.has_value()) {
                return Status::error(
                    "BatchArtifact.paths.report is null but report.json was provided");
            }
        } else {
            if (!paths.at("report").is_string()) {
                return Status::error("BatchArtifact.paths.report must be a string or null");
            }
            StatusOr<std::string> rel =
                validate_rel_filename(paths.at("report").get<std::string>(), "report");
            if (!rel.ok()) {
                return rel.status();
            }
            report_rel = std::move(rel.value());
            if (!report.has_value()) {
                return Status::error("BatchArtifact.paths.report set but report.json is missing");
            }
        }

        StatusOr<BatchArtifact> art =
            make(workspace_id.value(), batch_id.value(), created_utc.value(), job_digest.value(),
                 prior_digest.value(), family.value(), score_id.value(), score_version.value(),
                 backend.value(), static_cast<std::size_t>(k_i), static_cast<std::uint32_t>(seed_i),
                 std::move(candidates), std::move(report));
        if (!art.ok()) {
            return art.status();
        }
        art.value().candidates_relpath_ = std::move(candidates_rel.value());
        art.value().report_relpath_ = std::move(report_rel);
        return art;
    }

    [[nodiscard]] nlohmann::json manifest_to_json() const {
        nlohmann::json paths{
            {"candidates", candidates_relpath_},
            {"report", report_relpath_.has_value() ? nlohmann::json(*report_relpath_)
                                                   : nlohmann::json(nullptr)},
        };
        return nlohmann::json{
            {"schema", std::string(schema_id)},
            {"batch_id", batch_id_},
            {"workspace_id", workspace_id_},
            {"created_utc", created_utc_},
            {"job_digest_sha256", job_digest_sha256_},
            {"prior_digest_sha256", prior_digest_sha256_},
            {"family", family_},
            {"score_id", score_id_},
            {"score_version", score_version_},
            {"backend", std::string(BackendUtil::to_string(backend_))},
            {"k", k_},
            {"candidate_count", candidates_.size()},
            {"seed", seed_},
            {"ordering", std::string(ordering_id)},
            {"paths", std::move(paths)},
        };
    }

    [[nodiscard]] Status store(const std::filesystem::path& data_root) const {
        StatusOr<std::filesystem::path> dir =
            WorkspacePaths::batch_dir(data_root, workspace_id_, batch_id_);
        if (!dir.ok()) {
            return dir.status();
        }
        Status fixtures = WorkspacePaths::deny_fixtures_write(data_root, dir.value());
        if (!fixtures.ok()) {
            return fixtures;
        }

        std::error_code ec;
        std::filesystem::create_directories(dir.value(), ec);
        if (ec) {
            return Status::error("Failed to create batch directory: " + ec.message());
        }

        StatusOr<std::filesystem::path> candidates_path =
            WorkspacePaths::resolve_under(dir.value(), std::filesystem::path(candidates_relpath_));
        if (!candidates_path.ok()) {
            return candidates_path.status();
        }
        {
            std::ofstream out(candidates_path.value(), std::ios::binary | std::ios::trunc);
            if (!out) {
                return Status::error("Failed to write candidates.jsonl: " +
                                     candidates_path.value().string());
            }
            for (const nlohmann::json& line : candidates_) {
                out << line.dump() << '\n';
            }
            if (!out) {
                return Status::error("Failed while writing candidates.jsonl: " +
                                     candidates_path.value().string());
            }
        }

        if (report_relpath_.has_value()) {
            if (!report_.has_value()) {
                return Status::error("BatchArtifact report path set but report body missing");
            }
            StatusOr<std::filesystem::path> report_path =
                WorkspacePaths::resolve_under(dir.value(), std::filesystem::path(*report_relpath_));
            if (!report_path.ok()) {
                return report_path.status();
            }
            std::ofstream out(report_path.value(), std::ios::binary | std::ios::trunc);
            if (!out) {
                return Status::error("Failed to write report.json: " +
                                     report_path.value().string());
            }
            out << report_.value().dump(2);
            if (!out) {
                return Status::error("Failed while writing report.json: " +
                                     report_path.value().string());
            }
        }

        const std::filesystem::path manifest_path = dir.value() / "manifest.json";
        {
            std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                return Status::error("Failed to write manifest.json: " + manifest_path.string());
            }
            out << manifest_to_json().dump(2);
            if (!out) {
                return Status::error("Failed while writing manifest.json: " +
                                     manifest_path.string());
            }
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<BatchArtifact> load(const std::filesystem::path& data_root,
                                                      std::string_view workspace_id,
                                                      std::string_view batch_id) {
        StatusOr<std::filesystem::path> dir =
            WorkspacePaths::batch_dir(data_root, workspace_id, batch_id);
        if (!dir.ok()) {
            return dir.status();
        }
        if (!std::filesystem::is_directory(dir.value())) {
            return Status::error("BatchArtifact directory missing: " + dir.value().string());
        }

        const std::filesystem::path manifest_path = dir.value() / "manifest.json";
        std::ifstream manifest_in(manifest_path, std::ios::binary);
        if (!manifest_in) {
            return Status::error("Failed to open manifest.json: " + manifest_path.string());
        }
        nlohmann::json manifest;
        try {
            manifest_in >> manifest;
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid BatchArtifact manifest.json: ") + ex.what());
        }

        if (!manifest.contains("paths") || !manifest.at("paths").is_object()) {
            return Status::error("BatchArtifact.paths must be an object");
        }
        const nlohmann::json& paths = manifest.at("paths");
        if (!paths.contains("candidates") || !paths.at("candidates").is_string()) {
            return Status::error("BatchArtifact.paths.candidates must be a string");
        }
        StatusOr<std::filesystem::path> candidates_path = WorkspacePaths::resolve_under(
            dir.value(), std::filesystem::path(paths.at("candidates").get<std::string>()));
        if (!candidates_path.ok()) {
            return candidates_path.status();
        }
        StatusOr<std::vector<nlohmann::json>> candidates =
            load_candidates_jsonl(candidates_path.value());
        if (!candidates.ok()) {
            return candidates.status();
        }

        std::optional<nlohmann::json> report;
        if (paths.contains("report") && !paths.at("report").is_null()) {
            if (!paths.at("report").is_string()) {
                return Status::error("BatchArtifact.paths.report must be a string or null");
            }
            StatusOr<std::filesystem::path> report_path = WorkspacePaths::resolve_under(
                dir.value(), std::filesystem::path(paths.at("report").get<std::string>()));
            if (!report_path.ok()) {
                return report_path.status();
            }
            std::ifstream report_in(report_path.value(), std::ios::binary);
            if (!report_in) {
                return Status::error("Failed to open report.json: " + report_path.value().string());
            }
            try {
                report_in >> report.emplace();
            } catch (const nlohmann::json::exception& ex) {
                return Status::error(std::string("Invalid report.json: ") + ex.what());
            }
        }

        StatusOr<BatchArtifact> art =
            from_manifest_json(manifest, std::move(candidates.value()), std::move(report));
        if (!art.ok()) {
            return art.status();
        }
        if (art.value().workspace_id() != workspace_id) {
            return Status::error("BatchArtifact.workspace_id does not match load path");
        }
        if (art.value().batch_id() != batch_id) {
            return Status::error("BatchArtifact.batch_id does not match directory name");
        }
        return art;
    }

private:
    [[nodiscard]] static StatusOr<std::string> require_string(const nlohmann::json& root,
                                                              std::string_view key) {
        if (!root.contains(key) || !root.at(std::string(key)).is_string()) {
            return Status::error("BatchArtifact." + std::string(key) + " must be a string");
        }
        return root.at(std::string(key)).get<std::string>();
    }

    [[nodiscard]] static Status require_sha256_hex(std::string_view field, std::string_view hex) {
        if (hex.size() != 64) {
            return Status::error("BatchArtifact." + std::string(field) +
                                 " must be 64 lowercase hex digits");
        }
        for (char ch : hex) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if (!std::isxdigit(c) || (std::isalpha(c) && !std::islower(c))) {
                return Status::error("BatchArtifact." + std::string(field) +
                                     " must be 64 lowercase hex digits");
            }
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::string> validate_rel_filename(std::string_view name,
                                                                     std::string_view label) {
        if (name.empty() || name.find('/') != std::string_view::npos ||
            name.find('\\') != std::string_view::npos || name == "." || name == "..") {
            return Status::error("BatchArtifact.paths." + std::string(label) +
                                 " must be a single relative filename");
        }
        return std::string(name);
    }

    [[nodiscard]] static Status validate_report(const nlohmann::json& report) {
        if (!report.is_object()) {
            return Status::error("BatchArtifact report must be an object");
        }
        static constexpr std::string_view kForbidden[] = {
            "tok_per_sec", "tokens_per_sec", "wall_ms", "wall_s",  "duration_ms",  "duration_s",
            "elapsed_ms",  "device_ms",      "gpu_ms",  "cuda_ms", "device_clock",
        };
        for (std::string_view key : kForbidden) {
            if (report.contains(key)) {
                return Status::error("BatchArtifact report must not include timing field: " +
                                     std::string(key));
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status validate_candidates(const std::vector<nlohmann::json>& candidates,
                                                    std::string_view expected_score_id,
                                                    std::string_view expected_score_version,
                                                    Backend expected_backend) {
        std::unordered_set<std::string> seen_ids;
        seen_ids.reserve(candidates.size());

        ScoreOrder score_order = ScoreOrder::Asc;
        StatusOr<ScoreId> score_id = ScoreId::from_string(expected_score_id);
        if (score_id.ok()) {
            score_order = ScoreOrderUtil::for_score_id(score_id.value());
        }

        std::optional<BatchHit> prev_hit;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const nlohmann::json& row = candidates[i];
            if (!row.is_object()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "] must be an object");
            }
            const std::string dumped = row.dump();
            if (dumped.size() > kMaxCandidateLineBytes) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "] line exceeds kMaxCandidateLineBytes (" +
                                     std::to_string(kMaxCandidateLineBytes) + ")");
            }
            if (!row.contains("candidate_id") || !row.at("candidate_id").is_string()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].candidate_id must be a string");
            }
            const std::string cid = row.at("candidate_id").get<std::string>();
            if (cid.empty()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].candidate_id must be non-empty");
            }
            if (!seen_ids.insert(cid).second) {
                return Status::error("BatchArtifact candidate_id not unique within batch: " + cid);
            }
            if (!row.contains("envelope") || !row.at("envelope").is_object()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].envelope must be an object");
            }
            StatusOr<TransformEnvelope> envelope = TransformEnvelope::from_json(row.at("envelope"));
            if (!envelope.ok()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].envelope invalid: " + envelope.status().message());
            }
            if (!row.contains("output_indices") || !row.at("output_indices").is_array()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].output_indices must be an array");
            }
            for (std::size_t j = 0; j < row.at("output_indices").size(); ++j) {
                const auto& idx = row.at("output_indices")[j];
                if (!idx.is_number_integer()) {
                    return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                         "].output_indices must be integers");
                }
                const auto v = idx.get<std::int64_t>();
                if (v < 0 || v > 28) {
                    return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                         "].output_indices must be in 0..28");
                }
            }
            if (!row.contains("score") || !row.at("score").is_object()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].score must be an object");
            }
            const nlohmann::json& score = row.at("score");
            if (!score.contains("score_id") || !score.at("score_id").is_string() ||
                score.at("score_id").get<std::string>() != expected_score_id) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].score.score_id must match manifest");
            }
            if (!score.contains("score_version") || !score.at("score_version").is_string() ||
                score.at("score_version").get<std::string>() != expected_score_version) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].score.score_version must match manifest");
            }
            if (!score.contains("value") || !score.at("value").is_number()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].score.value must be a number");
            }
            if (!score.contains("backend") || !score.at("backend").is_string()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].score.backend must be a string");
            }
            StatusOr<Backend> backend =
                BackendUtil::from_string(score.at("backend").get<std::string>());
            if (!backend.ok() || backend.value() != expected_backend) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].score.backend must match manifest");
            }
            if (!row.contains("rank") || !row.at("rank").is_number_integer()) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].rank must be an integer");
            }
            const auto rank = row.at("rank").get<std::int64_t>();
            if (rank < 0 || static_cast<std::size_t>(rank) != i) {
                return Status::error("BatchArtifact candidates[" + std::to_string(i) +
                                     "].rank must equal line index (best-first)");
            }

            const double value = score.at("value").get<double>();
            const BatchHit hit{cid, value, i};
            if (prev_hit.has_value() && BatchOrdering::better(hit, *prev_hit, score_order)) {
                return Status::error(
                    "BatchArtifact candidates[" + std::to_string(i) +
                    "] is not best-first under batch_ordering_v0 (score/id/index)");
            }
            prev_hit = hit;
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::vector<nlohmann::json>>
    load_candidates_jsonl(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("Failed to open candidates.jsonl: " + path.string());
        }
        std::vector<nlohmann::json> out;
        std::string line;
        std::size_t line_no = 0;
        while (std::getline(in, line)) {
            ++line_no;
            if (line.empty()) {
                continue;
            }
            // Strip optional CR from Windows CRLF.
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            if (line.size() > kMaxCandidateLineBytes) {
                return Status::error("candidates.jsonl line " + std::to_string(line_no) +
                                     " exceeds kMaxCandidateLineBytes (" +
                                     std::to_string(kMaxCandidateLineBytes) + ")");
            }
            if (out.size() >= kMaxCandidatesPerBatch) {
                return Status::error("candidates.jsonl exceeds kMaxCandidatesPerBatch (" +
                                     std::to_string(kMaxCandidatesPerBatch) + ")");
            }
            try {
                out.push_back(nlohmann::json::parse(line));
            } catch (const nlohmann::json::exception& ex) {
                return Status::error("Invalid candidates.jsonl line " + std::to_string(line_no) +
                                     ": " + ex.what());
            }
        }
        return out;
    }

    std::string batch_id_;
    std::string workspace_id_;
    std::string created_utc_;
    std::string job_digest_sha256_;
    std::string prior_digest_sha256_;
    std::string family_;
    std::string score_id_;
    std::string score_version_ = "v0";
    Backend backend_ = Backend::Cpu;
    std::size_t k_ = 1;
    std::uint32_t seed_ = 1;
    std::string candidates_relpath_ = std::string(default_candidates_file);
    std::optional<std::string> report_relpath_;
    std::vector<nlohmann::json> candidates_;
    std::optional<nlohmann::json> report_;
};

#endif // BATCH_ARTIFACT_HPP
