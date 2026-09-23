#ifndef HYPOTHESIS_BRIDGE_HPP
#define HYPOTHESIS_BRIDGE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/beaufort_explicit_key_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/totient_offset_candidate_generator.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/hypothesis/hypothesis_record.hpp"
#include "parcae/hypothesis/hypothesis_status.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/search/batch_artifact.hpp"
#include "parcae/tool/transform_envelope.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Ingest `BatchArtifact` top-k lines into workspace `HypothesisRecord`s
/// (`docs/spec/search-loop.md` Hypothesis bridge).
///
/// Creates `proposed` records when the candidate envelope is complete. Does
/// **not** auto-set `promoted` and MUST NOT write under `data/fixtures/`.
///
/// Idempotent ids: `hypothesis_id_for(workspace_id, batch_id, candidate_id)` is
/// a pure function — re-ingesting the same triple updates or no-ops the same
/// file and never duplicates.
class HypothesisBridge {
public:
    /// Hex chars taken from SHA-256 for the id suffix (`h` + N hex ≤ 64).
    static constexpr std::size_t id_digest_hex_len = 32;
    /// UTF-8 field separator in the id preimage (not allowed in validated ids).
    static constexpr char id_preimage_sep = '\n';

    /// Options for `ingest`.
    class Options {
    public:
        /// When true, overwrite an existing file for the same hypothesis id.
        /// When false, leave existing records untouched (count as skipped).
        bool overwrite_existing = true;
    };

    /// Outcome of one ingest pass.
    class Result {
    public:
        Result() = default;

        [[nodiscard]] const std::vector<std::string>& created_ids() const noexcept {
            return created_ids_;
        }

        [[nodiscard]] const std::vector<std::string>& updated_ids() const noexcept {
            return updated_ids_;
        }

        [[nodiscard]] const std::vector<std::string>& skipped_ids() const noexcept {
            return skipped_ids_;
        }

        [[nodiscard]] std::size_t written_count() const noexcept {
            return created_ids_.size() + updated_ids_.size();
        }

    private:
        friend class HypothesisBridge;
        std::vector<std::string> created_ids_;
        std::vector<std::string> updated_ids_;
        std::vector<std::string> skipped_ids_;
    };

    /// Canonical preimage for `hypothesis_id_for` (documented in search-loop.md):
    /// `workspace_id + "\\n" + batch_id + "\\n" + candidate_id`.
    [[nodiscard]] static std::string id_preimage(
        std::string_view workspace_id,
        std::string_view batch_id,
        std::string_view candidate_id) {
        std::string material;
        material.reserve(
            workspace_id.size() + batch_id.size() + candidate_id.size() + 2);
        material.append(workspace_id);
        material.push_back(id_preimage_sep);
        material.append(batch_id);
        material.push_back(id_preimage_sep);
        material.append(candidate_id);
        return material;
    }

    /// Deterministic id from `(workspace_id, batch_id, candidate_id)`.
    /// Format: `h` + first `id_digest_hex_len` lowercase hex of SHA-256(preimage).
    [[nodiscard]] static std::string hypothesis_id_for(
        std::string_view workspace_id,
        std::string_view batch_id,
        std::string_view candidate_id) {
        const std::string digest = Sha256::hex_digest(
            id_preimage(workspace_id, batch_id, candidate_id));
        return std::string("h") + digest.substr(0, id_digest_hex_len);
    }

    /// Load the hypothesis that would be written for this provenance triple.
    [[nodiscard]] static StatusOr<HypothesisRecord> load_for_candidate(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view batch_id,
        std::string_view candidate_id) {
        return HypothesisRecord::load(
            data_root,
            workspace_id,
            hypothesis_id_for(workspace_id, batch_id, candidate_id));
    }

    /// Canonical `source` object for batch ingest (search-loop.md extended fields).
    [[nodiscard]] static nlohmann::json make_batch_source(
        std::string_view batch_id,
        std::string_view candidate_id,
        std::string_view family,
        std::string_view generator_id,
        std::size_t rank,
        std::optional<std::string> agent_run_id = std::nullopt) {
        nlohmann::json source{
            {"batch_id", std::string(batch_id)},
            {"candidate_id", std::string(candidate_id)},
            {"family", std::string(family)},
            {"generator_id", std::string(generator_id)},
            {"rank", rank},
            {"agent_run_id", nullptr},
        };
        if (agent_run_id.has_value()) {
            source["agent_run_id"] = std::move(*agent_run_id);
        }
        return source;
    }

    /// Require `batch_id` + `candidate_id` on a hypothesis `source` object.
    [[nodiscard]] static Status require_batch_provenance(const nlohmann::json& source) {
        if (!source.is_object()) {
            return Status::error("HypothesisBridge: source must be an object");
        }
        if (!source.contains("batch_id") || !source.at("batch_id").is_string() ||
            source.at("batch_id").get<std::string>().empty()) {
            return Status::error("HypothesisBridge: source.batch_id is required");
        }
        if (!source.contains("candidate_id") || !source.at("candidate_id").is_string() ||
            source.at("candidate_id").get<std::string>().empty()) {
            return Status::error("HypothesisBridge: source.candidate_id is required");
        }
        StatusOr<std::string> batch_ok =
            WorkspacePaths::validate_id(source.at("batch_id").get<std::string>());
        if (!batch_ok.ok()) {
            return Status::error(
                "HypothesisBridge: source.batch_id invalid: " + batch_ok.status().message());
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::string_view> generator_id_for_family(
        std::string_view family) {
        if (family == "caesar") {
            return CaesarCandidateGenerator::generator_id;
        }
        if (family == "atbash") {
            return AtbashCandidateGenerator::generator_id;
        }
        if (family == "atbash_caesar") {
            return AtbashCaesarCandidateGenerator::generator_id;
        }
        if (family == "affine") {
            return AffineCandidateGenerator::generator_id;
        }
        if (family == "vigenere") {
            return VigenereExplicitKeyCandidateGenerator::generator_id;
        }
        if (family == "beaufort") {
            return BeaufortExplicitKeyCandidateGenerator::generator_id;
        }
        if (family == "totient") {
            return TotientOffsetCandidateGenerator::generator_id;
        }
        return Status::error(
            "HypothesisBridge: unsupported family for generator_id mapping: " +
            std::string(family));
    }

    /// Write one hypothesis per candidate line in `artifact` (best-first order).
    /// Leaves `scores` empty; callers MAY attach values via `hypothesis_score`.
    [[nodiscard]] static StatusOr<Result> ingest(
        const std::filesystem::path& data_root,
        const BatchArtifact& artifact,
        const Options& options = {}) {
        StatusOr<std::string_view> generator_id =
            generator_id_for_family(artifact.family());
        if (!generator_id.ok()) {
            return generator_id.status();
        }

        Result out;
        for (const nlohmann::json& line : artifact.candidates()) {
            StatusOr<std::string> written = ingest_one(
                data_root, artifact, line, generator_id.value(), options, out);
            if (!written.ok()) {
                return written.status();
            }
        }
        return out;
    }

    /// Load batch from disk then ingest.
    [[nodiscard]] static StatusOr<Result> ingest_batch_dir(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view batch_id,
        const Options& options = {}) {
        StatusOr<BatchArtifact> artifact =
            BatchArtifact::load(data_root, workspace_id, batch_id);
        if (!artifact.ok()) {
            return artifact.status();
        }
        return ingest(data_root, artifact.value(), options);
    }

private:
    HypothesisBridge() = delete;

    [[nodiscard]] static StatusOr<std::string> ingest_one(
        const std::filesystem::path& data_root,
        const BatchArtifact& artifact,
        const nlohmann::json& line,
        std::string_view generator_id,
        const Options& options,
        Result& out) {
        if (!line.is_object()) {
            return Status::error("HypothesisBridge: candidate line must be an object");
        }
        if (!line.contains("candidate_id") || !line.at("candidate_id").is_string()) {
            return Status::error("HypothesisBridge: candidate_id is required");
        }
        if (!line.contains("envelope") || !line.at("envelope").is_object()) {
            return Status::error("HypothesisBridge: envelope is required");
        }

        const std::string candidate_id = line.at("candidate_id").get<std::string>();
        const std::string hypothesis_id = hypothesis_id_for(
            artifact.workspace_id(), artifact.batch_id(), candidate_id);

        StatusOr<std::filesystem::path> path = WorkspacePaths::hypothesis_file(
            data_root, artifact.workspace_id(), hypothesis_id);
        if (!path.ok()) {
            return path.status();
        }

        const bool existed = std::filesystem::is_regular_file(path.value());
        if (existed && !options.overwrite_existing) {
            out.skipped_ids_.push_back(hypothesis_id);
            return hypothesis_id;
        }

        StatusOr<parcae::tool::TransformEnvelope> envelope =
            parcae::tool::TransformEnvelope::from_json(line.at("envelope"));
        if (!envelope.ok()) {
            return envelope.status();
        }

        std::size_t rank = 0;
        if (line.contains("rank") && line.at("rank").is_number_integer()) {
            const std::int64_t r = line.at("rank").get<std::int64_t>();
            if (r >= 0) {
                rank = static_cast<std::size_t>(r);
            }
        }

        nlohmann::json source = make_batch_source(
            artifact.batch_id(),
            candidate_id,
            artifact.family(),
            generator_id,
            rank);
        Status provenance = require_batch_provenance(source);
        if (!provenance.ok()) {
            return provenance;
        }

        StatusOr<HypothesisRecord> record = HypothesisRecord::make_draft(
            artifact.workspace_id(),
            hypothesis_id,
            artifact.created_utc(),
            title_for(artifact.batch_id(), rank, candidate_id),
            envelope.value().to_json());
        if (!record.ok()) {
            return record.status();
        }

        // Method is complete → proposed (search-loop.md SHOULD).
        Status proposed = record.value().set_status(HypothesisStatus::Proposed);
        if (!proposed.ok()) {
            return proposed;
        }

        Status source_ok = record.value().set_source(source);
        if (!source_ok.ok()) {
            return source_ok;
        }
        record.value().set_rationale(
            "Auto-ingested from batch " + artifact.batch_id() + " by HypothesisBridge");
        record.value().recompute_method_digest();

        if (line.contains("output_indices") && line.at("output_indices").is_array()) {
            StatusOr<std::vector<Index29>> indices =
                parse_output_indices(line.at("output_indices"));
            if (!indices.ok()) {
                return indices.status();
            }
            record.value().set_output_indices_digest(indices.value());
        }

        // Preserve lifecycle if an advanced status already exists and we overwrite.
        if (existed) {
            StatusOr<HypothesisRecord> previous =
                HypothesisRecord::load(data_root, artifact.workspace_id(), hypothesis_id);
            if (previous.ok()) {
                const HypothesisStatus prev = previous.value().status();
                if (prev == HypothesisStatus::Scored || prev == HypothesisStatus::Rejected ||
                    prev == HypothesisStatus::Promoted) {
                    Status method_ok = previous.value().set_method(envelope.value().to_json());
                    if (!method_ok.ok()) {
                        return method_ok;
                    }
                    // Refresh provenance (batch_id / candidate_id) even when status is kept.
                    Status refresh_source = previous.value().set_source(std::move(source));
                    if (!refresh_source.ok()) {
                        return refresh_source;
                    }
                    previous.value().set_title(record.value().title());
                    previous.value().set_rationale(record.value().rationale());
                    previous.value().set_updated_utc(artifact.created_utc());
                    previous.value().recompute_method_digest();
                    if (line.contains("output_indices") &&
                        line.at("output_indices").is_array()) {
                        StatusOr<std::vector<Index29>> indices =
                            parse_output_indices(line.at("output_indices"));
                        if (!indices.ok()) {
                            return indices.status();
                        }
                        previous.value().set_output_indices_digest(indices.value());
                    }
                    Status stored = previous.value().store(data_root);
                    if (!stored.ok()) {
                        return stored;
                    }
                    out.updated_ids_.push_back(hypothesis_id);
                    return hypothesis_id;
                }
            }
        }

        Status stored = record.value().store(data_root);
        if (!stored.ok()) {
            return stored;
        }
        if (existed) {
            out.updated_ids_.push_back(hypothesis_id);
        } else {
            out.created_ids_.push_back(hypothesis_id);
        }
        return hypothesis_id;
    }

    [[nodiscard]] static std::string title_for(
        std::string_view batch_id,
        std::size_t rank,
        std::string_view candidate_id) {
        return std::string("batch ") + std::string(batch_id) + " rank " +
               std::to_string(rank) + " " + std::string(candidate_id);
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> parse_output_indices(
        const nlohmann::json& arr) {
        std::vector<Index29> out;
        out.reserve(arr.size());
        for (const auto& item : arr) {
            if (!item.is_number_integer()) {
                return Status::error("HypothesisBridge: output_indices entries must be integers");
            }
            const std::int64_t value = item.get<std::int64_t>();
            if (value < 0 || value >= static_cast<std::int64_t>(Index29::modulus)) {
                return Status::error("HypothesisBridge: output_indices out of range [0,28]");
            }
            out.push_back(Index29{static_cast<std::uint8_t>(value)});
        }
        return out;
    }
};

#endif  // HYPOTHESIS_BRIDGE_HPP
