#ifndef HYPOTHESIS_RECORD_HPP
#define HYPOTHESIS_RECORD_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/hypothesis_status.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/tool/transform_envelope.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// One scored/proposed transform hypothesis (`parcae.hypothesis.v0`).
class HypothesisRecord {
public:
    static constexpr std::string_view schema_id = "parcae.hypothesis.v0";

    HypothesisRecord() = default;

    [[nodiscard]] const std::string& id() const noexcept {
        return id_;
    }

    [[nodiscard]] const std::string& workspace_id() const noexcept {
        return workspace_id_;
    }

    [[nodiscard]] HypothesisStatus status() const noexcept {
        return status_;
    }

    [[nodiscard]] const std::string& title() const noexcept {
        return title_;
    }

    [[nodiscard]] const std::string& rationale() const noexcept {
        return rationale_;
    }

    [[nodiscard]] const nlohmann::json& method() const noexcept {
        return method_;
    }

    [[nodiscard]] const nlohmann::json& source() const noexcept {
        return source_;
    }

    [[nodiscard]] const nlohmann::json& scores() const noexcept {
        return scores_;
    }

    [[nodiscard]] const nlohmann::json& preview() const noexcept {
        return preview_;
    }

    [[nodiscard]] const nlohmann::json& digests() const noexcept {
        return digests_;
    }

    [[nodiscard]] const nlohmann::json& promotion() const noexcept {
        return promotion_;
    }

    [[nodiscard]] const std::string& created_utc() const noexcept {
        return created_utc_;
    }

    [[nodiscard]] const std::string& updated_utc() const noexcept {
        return updated_utc_;
    }

    /// Recursively sort object keys; compact UTF-8 dump for digests.
    [[nodiscard]] static nlohmann::json canonicalize_json(const nlohmann::json& value) {
        if (value.is_object()) {
            std::vector<std::string> keys;
            keys.reserve(value.size());
            for (auto it = value.begin(); it != value.end(); ++it) {
                keys.push_back(it.key());
            }
            std::sort(keys.begin(), keys.end());
            nlohmann::json out = nlohmann::json::object();
            for (const std::string& key : keys) {
                out[key] = canonicalize_json(value.at(key));
            }
            return out;
        }
        if (value.is_array()) {
            nlohmann::json out = nlohmann::json::array();
            for (const auto& item : value) {
                out.push_back(canonicalize_json(item));
            }
            return out;
        }
        return value;
    }

    [[nodiscard]] static std::string canonical_dump(const nlohmann::json& value) {
        return canonicalize_json(value).dump();
    }

    [[nodiscard]] static std::string method_sha256(const nlohmann::json& method) {
        return Sha256::hex_digest(canonical_dump(method));
    }

    [[nodiscard]] static std::string output_indices_sha256(std::span<const Index29> indices) {
        nlohmann::json arr = nlohmann::json::array();
        for (const Index29 idx : indices) {
            arr.push_back(idx.value());
        }
        return Sha256::hex_digest(canonical_dump(arr));
    }

    [[nodiscard]] Status set_status(HypothesisStatus next) {
        Status ok = HypothesisStatusUtil::require_transition(status_, next);
        if (!ok.ok()) {
            return ok;
        }
        status_ = next;
        return Status::success();
    }

    void set_title(std::string title) {
        title_ = std::move(title);
    }

    void set_rationale(std::string rationale) {
        rationale_ = std::move(rationale);
    }

    void set_updated_utc(std::string utc) {
        updated_utc_ = std::move(utc);
    }

    [[nodiscard]] Status set_source(nlohmann::json source) {
        Status ok = validate_source(source);
        if (!ok.ok()) {
            return ok;
        }
        source_ = std::move(source);
        return Status::success();
    }

    /// Validate `source` per hypothesis-workspace.md (null or object with known keys).
    [[nodiscard]] static Status validate_source(const nlohmann::json& source) {
        if (source.is_null()) {
            return Status::success();
        }
        if (!source.is_object()) {
            return Status::error("HypothesisRecord.source must be an object or null");
        }
        for (auto it = source.begin(); it != source.end(); ++it) {
            const std::string& key = it.key();
            if (key != "generator_id" && key != "candidate_id" && key != "agent_run_id" &&
                key != "batch_id" && key != "family" && key != "rank") {
                return Status::error("HypothesisRecord.source unknown key: " + key);
            }
        }
        Status gen = validate_optional_id_string(source, "generator_id", /*allow_empty=*/false);
        if (!gen.ok()) {
            return gen;
        }
        Status cand = validate_optional_id_string(source, "candidate_id", /*allow_empty=*/false);
        if (!cand.ok()) {
            return cand;
        }
        Status agent = validate_optional_id_string(source, "agent_run_id", /*allow_empty=*/false);
        if (!agent.ok()) {
            return agent;
        }
        if (source.contains("batch_id") && !source.at("batch_id").is_null()) {
            if (!source.at("batch_id").is_string()) {
                return Status::error("HypothesisRecord.source.batch_id must be a string or null");
            }
            const std::string batch_id = source.at("batch_id").get<std::string>();
            if (batch_id.empty()) {
                return Status::error("HypothesisRecord.source.batch_id must be non-empty when set");
            }
            StatusOr<std::string> ok = WorkspacePaths::validate_id(batch_id);
            if (!ok.ok()) {
                return Status::error(
                    "HypothesisRecord.source.batch_id invalid: " + ok.status().message());
            }
        }
        if (source.contains("family") && !source.at("family").is_null()) {
            if (!source.at("family").is_string()) {
                return Status::error("HypothesisRecord.source.family must be a string or null");
            }
            const std::string family = source.at("family").get<std::string>();
            if (!is_known_family(family)) {
                return Status::error(
                    "HypothesisRecord.source.family unknown: " + family);
            }
        }
        if (source.contains("rank") && !source.at("rank").is_null()) {
            if (!source.at("rank").is_number_integer()) {
                return Status::error("HypothesisRecord.source.rank must be an integer or null");
            }
            if (source.at("rank").get<std::int64_t>() < 0) {
                return Status::error("HypothesisRecord.source.rank must be >= 0");
            }
        }
        return Status::success();
    }

    /// Empty provenance stub used by `make_draft` / CLI init.
    [[nodiscard]] static nlohmann::json empty_source() {
        return nlohmann::json{
            {"generator_id", nullptr},
            {"candidate_id", nullptr},
            {"agent_run_id", nullptr},
            {"batch_id", nullptr},
            {"family", nullptr},
            {"rank", nullptr},
        };
    }

    void set_preview(nlohmann::json preview) {
        preview_ = std::move(preview);
    }

    [[nodiscard]] Status set_method(nlohmann::json method) {
        StatusOr<TransformEnvelope> envelope =
            TransformEnvelope::from_json(method);
        if (!envelope.ok()) {
            return envelope.status();
        }
        method_ = envelope.value().to_json();
        return Status::success();
    }

    [[nodiscard]] Status append_score(nlohmann::json entry) {
        nlohmann::json arr = nlohmann::json::array();
        arr.push_back(entry);
        Status ok = validate_scores(arr);
        if (!ok.ok()) {
            return ok;
        }
        if (!scores_.is_array()) {
            scores_ = nlohmann::json::array();
        }
        scores_.push_back(std::move(entry));
        return Status::success();
    }

    /// Refresh `digests.method_sha256` from current `method`.
    void recompute_method_digest() {
        if (!digests_.is_object()) {
            digests_ = nlohmann::json::object();
        }
        digests_["method_sha256"] = method_sha256(method_);
    }

    void set_output_indices_digest(std::span<const Index29> indices) {
        if (!digests_.is_object()) {
            digests_ = nlohmann::json::object();
        }
        digests_["output_indices_sha256"] = output_indices_sha256(indices);
    }

    /// Return error when a non-null stored digest does not match current method.
    [[nodiscard]] Status verify_method_digest() const {
        if (!digests_.is_object() || !digests_.contains("method_sha256") ||
            digests_.at("method_sha256").is_null()) {
            return Status::success();
        }
        if (!digests_.at("method_sha256").is_string()) {
            return Status::error("digests.method_sha256 must be a string or null");
        }
        const std::string expected = digests_.at("method_sha256").get<std::string>();
        const std::string actual = method_sha256(method_);
        if (expected != actual) {
            return Status::error("digests.method_sha256 does not match method JSON");
        }
        return Status::success();
    }

    /// Return error when a non-null stored digest does not match `indices`.
    [[nodiscard]] Status verify_output_indices_digest(std::span<const Index29> indices) const {
        if (!digests_.is_object() || !digests_.contains("output_indices_sha256") ||
            digests_.at("output_indices_sha256").is_null()) {
            return Status::success();
        }
        if (!digests_.at("output_indices_sha256").is_string()) {
            return Status::error("digests.output_indices_sha256 must be a string or null");
        }
        const std::string expected = digests_.at("output_indices_sha256").get<std::string>();
        const std::string actual = output_indices_sha256(indices);
        if (expected != actual) {
            return Status::error("digests.output_indices_sha256 does not match indices");
        }
        return Status::success();
    }

    [[nodiscard]] nlohmann::json to_json() const {
        return nlohmann::json{
            {"schema", std::string(schema_id)},
            {"id", id_},
            {"workspace_id", workspace_id_},
            {"created_utc", created_utc_},
            {"updated_utc", updated_utc_},
            {"status", std::string(HypothesisStatusUtil::to_string(status_))},
            {"title", title_},
            {"rationale", rationale_},
            {"method", method_},
            {"source", source_},
            {"scores", scores_},
            {"preview", preview_},
            {"digests", digests_},
            {"promotion", promotion_},
        };
    }

    [[nodiscard]] static StatusOr<HypothesisRecord> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("HypothesisRecord must be a JSON object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error("HypothesisRecord.schema must be parcae.hypothesis.v0");
        }

        HypothesisRecord record;
        StatusOr<std::string> id = require_string(root, "id");
        if (!id.ok()) {
            return id.status();
        }
        StatusOr<std::string> valid_id = WorkspacePaths::validate_id(id.value());
        if (!valid_id.ok()) {
            return valid_id.status();
        }
        record.id_ = std::move(valid_id.value());

        StatusOr<std::string> workspace_id = require_string(root, "workspace_id");
        if (!workspace_id.ok()) {
            return workspace_id.status();
        }
        StatusOr<std::string> valid_ws = WorkspacePaths::validate_id(workspace_id.value());
        if (!valid_ws.ok()) {
            return valid_ws.status();
        }
        record.workspace_id_ = std::move(valid_ws.value());

        StatusOr<std::string> created = require_string(root, "created_utc");
        if (!created.ok()) {
            return created.status();
        }
        record.created_utc_ = std::move(created.value());
        StatusOr<std::string> updated = require_string(root, "updated_utc");
        if (!updated.ok()) {
            return updated.status();
        }
        record.updated_utc_ = std::move(updated.value());

        StatusOr<std::string> status_text = require_string(root, "status");
        if (!status_text.ok()) {
            return status_text.status();
        }
        StatusOr<HypothesisStatus> status = HypothesisStatusUtil::from_string(status_text.value());
        if (!status.ok()) {
            return status.status();
        }
        record.status_ = status.value();

        record.title_ = optional_string(root, "title");
        record.rationale_ = optional_string(root, "rationale");

        if (!root.contains("method") || !root.at("method").is_object()) {
            return Status::error("HypothesisRecord.method must be an object");
        }
        StatusOr<TransformEnvelope> envelope =
            TransformEnvelope::from_json(root.at("method"));
        if (!envelope.ok()) {
            return Status::error(
                "HypothesisRecord.method is not a valid TransformEnvelope: " +
                envelope.status().message());
        }
        record.method_ = envelope.value().to_json();

        if (root.contains("source")) {
            if (!root.at("source").is_object() && !root.at("source").is_null()) {
                return Status::error("HypothesisRecord.source must be an object or null");
            }
            if (root.at("source").is_object()) {
                Status source_ok = validate_source(root.at("source"));
                if (!source_ok.ok()) {
                    return source_ok;
                }
                record.source_ = root.at("source");
            }
        }

        if (!root.contains("scores") || !root.at("scores").is_array()) {
            return Status::error("HypothesisRecord.scores must be an array");
        }
        Status scores_ok = validate_scores(root.at("scores"));
        if (!scores_ok.ok()) {
            return scores_ok;
        }
        record.scores_ = root.at("scores");

        if (root.contains("preview") && root.at("preview").is_object()) {
            record.preview_ = root.at("preview");
        }
        if (!root.contains("digests") || !root.at("digests").is_object()) {
            return Status::error("HypothesisRecord.digests must be an object");
        }
        record.digests_ = root.at("digests");
        if (root.contains("promotion") && root.at("promotion").is_object()) {
            record.promotion_ = root.at("promotion");
        }

        return record;
    }

    [[nodiscard]] static StatusOr<HypothesisRecord> from_string(const std::string& text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid HypothesisRecord JSON: ") + ex.what());
        }
        return from_json(root);
    }

    /// Load `data_root/workspaces/<workspace_id>/hypotheses/<hypothesis_id>.json`.
    [[nodiscard]] static StatusOr<HypothesisRecord> load(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view hypothesis_id) {
        StatusOr<std::filesystem::path> path =
            WorkspacePaths::hypothesis_file(data_root, workspace_id, hypothesis_id);
        if (!path.ok()) {
            return path.status();
        }
        return load_file(path.value(), workspace_id, hypothesis_id);
    }

    /// Load from an explicit path; still enforces id / workspace_id match.
    [[nodiscard]] static StatusOr<HypothesisRecord> load_file(
        const std::filesystem::path& path,
        std::string_view expected_workspace_id,
        std::string_view expected_hypothesis_id) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("Failed to open hypothesis file: " + path.string());
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        StatusOr<HypothesisRecord> record = from_string(buf.str());
        if (!record.ok()) {
            return record.status();
        }
        if (record.value().workspace_id() != expected_workspace_id) {
            return Status::error(
                "HypothesisRecord.workspace_id does not match workspace directory");
        }
        if (record.value().id() != expected_hypothesis_id) {
            return Status::error("HypothesisRecord.id does not match filename stem");
        }
        const std::string stem = path.stem().string();
        if (stem != expected_hypothesis_id) {
            return Status::error("hypothesis filename stem must equal hypothesis id");
        }
        return record;
    }

    /// Path-safe write under `data_root/workspaces/<workspace_id>/hypotheses/`.
    [[nodiscard]] Status store(const std::filesystem::path& data_root) const {
        StatusOr<std::filesystem::path> path =
            WorkspacePaths::hypothesis_file(data_root, workspace_id_, id_);
        if (!path.ok()) {
            return path.status();
        }

        StatusOr<std::filesystem::path> ws_root =
            WorkspacePaths::workspace_root(data_root, workspace_id_);
        if (!ws_root.ok()) {
            return ws_root.status();
        }

        // Ensure the write target stays under the workspace and not fixtures.
        StatusOr<std::filesystem::path> resolved = WorkspacePaths::resolve_under(
            ws_root.value(),
            std::filesystem::path("hypotheses") / (id_ + ".json"));
        if (!resolved.ok()) {
            return resolved.status();
        }
        Status fixtures = WorkspacePaths::deny_fixtures_write(data_root, resolved.value());
        if (!fixtures.ok()) {
            return fixtures;
        }

        std::error_code ec;
        std::filesystem::create_directories(resolved.value().parent_path(), ec);
        if (ec) {
            return Status::error("Failed to create hypotheses directory: " + ec.message());
        }

        const std::string body = to_json().dump(2) + '\n';
        std::ofstream out(resolved.value(), std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::error("Failed to write hypothesis file: " + resolved.value().string());
        }
        out << body;
        if (!out) {
            return Status::error("Failed while writing hypothesis file: " + resolved.value().string());
        }
        return Status::success();
    }

    /// List hypothesis ids in `workspaces/<id>/hypotheses/*.json` (sorted).
    [[nodiscard]] static StatusOr<std::vector<std::string>> list_ids(
        const std::filesystem::path& data_root,
        std::string_view workspace_id) {
        StatusOr<std::filesystem::path> dir =
            WorkspacePaths::hypotheses_dir(data_root, workspace_id);
        if (!dir.ok()) {
            return dir.status();
        }
        if (!std::filesystem::is_directory(dir.value())) {
            return std::vector<std::string>{};
        }
        std::vector<std::string> ids;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(dir.value(), ec)) {
            if (ec) {
                return Status::error("Failed to list hypotheses: " + ec.message());
            }
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().extension() != ".json") {
                continue;
            }
            const std::string stem = entry.path().stem().string();
            StatusOr<std::string> id = WorkspacePaths::validate_id(stem);
            if (!id.ok()) {
                continue;
            }
            ids.push_back(id.value());
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    /// Build a minimal draft stub (CLI `init` / tests).
    [[nodiscard]] static StatusOr<HypothesisRecord> make_draft(
        std::string_view workspace_id,
        std::string_view hypothesis_id,
        std::string_view created_utc,
        std::string_view title = "",
        nlohmann::json method = nlohmann::json{
            {"transform_id", "identity"},
            {"direction", "decrypt"},
            {"params", nlohmann::json::object()},
        }) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        StatusOr<std::string> hid = WorkspacePaths::validate_id(hypothesis_id);
        if (!hid.ok()) {
            return hid.status();
        }
        StatusOr<TransformEnvelope> envelope =
            TransformEnvelope::from_json(method);
        if (!envelope.ok()) {
            return envelope.status();
        }

        HypothesisRecord record;
        record.id_ = std::move(hid.value());
        record.workspace_id_ = std::move(wid.value());
        record.created_utc_ = std::string(created_utc);
        record.updated_utc_ = std::string(created_utc);
        record.status_ = HypothesisStatus::Draft;
        record.title_ = std::string(title);
        record.method_ = envelope.value().to_json();
        record.source_ = empty_source();
        record.scores_ = nlohmann::json::array();
        record.preview_ = nlohmann::json{{"latin_prefix", nullptr}, {"max_chars", 64}};
        record.digests_ = nlohmann::json{
            {"method_sha256", nullptr},
            {"output_indices_sha256", nullptr},
        };
        record.promotion_ = nlohmann::json{
            {"target_fixture_id", nullptr},
            {"notes", "Tools MUST NOT auto-write fixtures."},
        };
        return record;
    }

private:
    [[nodiscard]] static bool is_known_family(std::string_view family) noexcept {
        return family == "caesar" || family == "atbash" || family == "atbash_caesar" ||
               family == "affine" || family == "vigenere" || family == "beaufort" ||
               family == "totient";
    }

    [[nodiscard]] static Status validate_optional_id_string(
        const nlohmann::json& source,
        const char* key,
        bool allow_empty) {
        if (!source.contains(key) || source.at(key).is_null()) {
            return Status::success();
        }
        if (!source.at(key).is_string()) {
            return Status::error(
                std::string("HypothesisRecord.source.") + key + " must be a string or null");
        }
        const std::string value = source.at(key).get<std::string>();
        if (!allow_empty && value.empty()) {
            return Status::error(
                std::string("HypothesisRecord.source.") + key + " must be non-empty when set");
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::string> require_string(
        const nlohmann::json& root,
        const char* key) {
        if (!root.contains(key) || !root.at(key).is_string()) {
            return Status::error(std::string("HypothesisRecord.") + key + " must be a string");
        }
        return root.at(key).get<std::string>();
    }

    [[nodiscard]] static std::string optional_string(const nlohmann::json& root, const char* key) {
        if (!root.contains(key) || root.at(key).is_null()) {
            return {};
        }
        if (root.at(key).is_string()) {
            return root.at(key).get<std::string>();
        }
        return {};
    }

    [[nodiscard]] static Status validate_scores(const nlohmann::json& scores) {
        for (std::size_t i = 0; i < scores.size(); ++i) {
            const auto& entry = scores[i];
            if (!entry.is_object()) {
                return Status::error("scores[" + std::to_string(i) + "] must be an object");
            }
            for (const char* key : {"score_id", "score_version", "backend", "scored_utc"}) {
                if (!entry.contains(key) || !entry.at(key).is_string()) {
                    return Status::error(
                        "scores[" + std::to_string(i) + "]." + key + " must be a string");
                }
            }
            if (!entry.contains("value") || !entry.at("value").is_number()) {
                return Status::error("scores[" + std::to_string(i) + "].value must be a number");
            }
            const double value = entry.at("value").get<double>();
            if (!std::isfinite(value)) {
                return Status::error("scores[" + std::to_string(i) + "].value must be finite");
            }
            const std::string backend = entry.at("backend").get<std::string>();
            if (backend != "cpu" && backend != "cuda") {
                return Status::error("scores[" + std::to_string(i) + "].backend must be cpu|cuda");
            }
        }
        return Status::success();
    }

    std::string id_;
    std::string workspace_id_;
    std::string created_utc_;
    std::string updated_utc_;
    HypothesisStatus status_ = HypothesisStatus::Draft;
    std::string title_;
    std::string rationale_;
    nlohmann::json method_ = nlohmann::json::object();
    nlohmann::json source_ = nlohmann::json::object();
    nlohmann::json scores_ = nlohmann::json::array();
    nlohmann::json preview_ = nlohmann::json::object();
    nlohmann::json digests_ = nlohmann::json::object();
    nlohmann::json promotion_ = nlohmann::json::object();
};

#endif  // HYPOTHESIS_RECORD_HPP
