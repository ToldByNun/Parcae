#ifndef SEARCH_PRIOR_HPP
#define SEARCH_PRIOR_HPP

#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/hypothesis_record.hpp"
#include "parcae/hypothesis/hypothesis_status.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/tool/transform_envelope.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Seeds + exclusions derived from workspace hypotheses (`parcae.search_prior.v0`).
class SearchPrior {
public:
    static constexpr std::string_view schema_id = "parcae.search_prior.v0";

    /// One promoted (or optionally scored) hypothesis used as a seed envelope.
    class Seed {
    public:
        Seed() = default;

        Seed(std::string hypothesis_id, nlohmann::json envelope)
            : hypothesis_id_(std::move(hypothesis_id)), envelope_(std::move(envelope)) {}

        [[nodiscard]] const std::string& hypothesis_id() const noexcept {
            return hypothesis_id_;
        }

        [[nodiscard]] const nlohmann::json& envelope() const noexcept {
            return envelope_;
        }

        [[nodiscard]] nlohmann::json to_json() const {
            return nlohmann::json{
                {"hypothesis_id", hypothesis_id_},
                {"envelope", envelope_},
            };
        }

    private:
        friend class SearchPrior;
        std::string hypothesis_id_;
        nlohmann::json envelope_ = nlohmann::json::object();
    };

    /// One rejected hypothesis keyed by canonical params digest.
    class Exclusion {
    public:
        Exclusion() = default;

        Exclusion(std::string param_hash, std::string hypothesis_id, std::string reason)
            : param_hash_(std::move(param_hash)),
              hypothesis_id_(std::move(hypothesis_id)),
              reason_(std::move(reason)) {}

        [[nodiscard]] const std::string& param_hash() const noexcept {
            return param_hash_;
        }

        [[nodiscard]] const std::string& hypothesis_id() const noexcept {
            return hypothesis_id_;
        }

        [[nodiscard]] const std::string& reason() const noexcept {
            return reason_;
        }

        [[nodiscard]] nlohmann::json to_json() const {
            return nlohmann::json{
                {"param_hash", param_hash_},
                {"hypothesis_id", hypothesis_id_},
                {"reason", reason_},
            };
        }

    private:
        friend class SearchPrior;
        std::string param_hash_;
        std::string hypothesis_id_;
        std::string reason_;
    };

    /// Options for `from_workspace`.
    class BuildOptions {
    public:
        /// When true, status `scored` is also added to `seeds` (spec MAY).
        bool include_scored_as_seeds = false;
    };

    SearchPrior() = default;

    [[nodiscard]] const std::string& workspace_id() const noexcept {
        return workspace_id_;
    }

    [[nodiscard]] const std::vector<Seed>& seeds() const noexcept {
        return seeds_;
    }

    [[nodiscard]] const std::vector<Exclusion>& exclusions() const noexcept {
        return exclusions_;
    }

    [[nodiscard]] const std::string& built_utc() const noexcept {
        return built_utc_;
    }

    /// Canonical SHA-256 hex of replay-relevant prior JSON.
    [[nodiscard]] std::string prior_digest_sha256() const {
        return Sha256::hex_digest(canonical_dump(to_json()));
    }

    /// `sha256:` + lowercase hex of canonical params JSON (search-loop.md).
    [[nodiscard]] static std::string param_hash_of(const nlohmann::json& params) {
        const nlohmann::json object = params.is_null() ? nlohmann::json::object() : params;
        return std::string("sha256:") + Sha256::hex_digest(canonical_dump(object));
    }

    [[nodiscard]] static std::string param_hash_of_envelope(const nlohmann::json& envelope) {
        if (!envelope.is_object()) {
            return param_hash_of(nlohmann::json::object());
        }
        if (!envelope.contains("params") || envelope.at("params").is_null()) {
            return param_hash_of(nlohmann::json::object());
        }
        return param_hash_of(envelope.at("params"));
    }

    [[nodiscard]] bool excludes_params(const nlohmann::json& params) const {
        const std::string hash = param_hash_of(params);
        for (const Exclusion& ex : exclusions_) {
            if (ex.param_hash() == hash) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool excludes_envelope(const nlohmann::json& envelope) const {
        return excludes_params(
            envelope.is_object() && envelope.contains("params") && !envelope.at("params").is_null()
                ? envelope.at("params")
                : nlohmann::json::object());
    }

    [[nodiscard]] static StatusOr<SearchPrior> make(
        std::string_view workspace_id,
        std::vector<Seed> seeds,
        std::vector<Exclusion> exclusions,
        std::string_view built_utc) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        if (built_utc.empty()) {
            return Status::error("SearchPrior.built_utc must be non-empty");
        }
        for (const Seed& seed : seeds) {
            Status ok = validate_seed(seed);
            if (!ok.ok()) {
                return ok;
            }
        }
        for (const Exclusion& ex : exclusions) {
            Status ok = validate_exclusion(ex);
            if (!ok.ok()) {
                return ok;
            }
        }
        sort_seeds(seeds);
        sort_exclusions(exclusions);

        SearchPrior prior;
        prior.workspace_id_ = std::move(wid.value());
        prior.seeds_ = std::move(seeds);
        prior.exclusions_ = std::move(exclusions);
        prior.built_utc_ = std::string(built_utc);
        return prior;
    }

    /// Load promoted / rejected (and optionally scored) hypotheses from a workspace.
    [[nodiscard]] static StatusOr<SearchPrior> from_workspace(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view built_utc) {
        return from_workspace(data_root, workspace_id, built_utc, BuildOptions{});
    }

    [[nodiscard]] static StatusOr<SearchPrior> from_workspace(
        const std::filesystem::path& data_root,
        std::string_view workspace_id,
        std::string_view built_utc,
        BuildOptions options) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        StatusOr<std::vector<std::string>> ids =
            HypothesisRecord::list_ids(data_root, wid.value());
        if (!ids.ok()) {
            return ids.status();
        }

        std::vector<Seed> seeds;
        std::vector<Exclusion> exclusions;
        for (const std::string& hid : ids.value()) {
            StatusOr<HypothesisRecord> record =
                HypothesisRecord::load(data_root, wid.value(), hid);
            if (!record.ok()) {
                return record.status();
            }
            const HypothesisStatus status = record.value().status();
            if (status == HypothesisStatus::Promoted ||
                (options.include_scored_as_seeds && status == HypothesisStatus::Scored)) {
                StatusOr<TransformEnvelope> envelope =
                    TransformEnvelope::from_json(record.value().method());
                if (!envelope.ok()) {
                    return Status::error(
                        "SearchPrior seed envelope invalid for " + hid + ": " +
                        envelope.status().message());
                }
                seeds.emplace_back(hid, envelope.value().to_json());
            } else if (status == HypothesisStatus::Rejected) {
                StatusOr<TransformEnvelope> envelope =
                    TransformEnvelope::from_json(record.value().method());
                if (!envelope.ok()) {
                    return Status::error(
                        "SearchPrior exclusion envelope invalid for " + hid + ": " +
                        envelope.status().message());
                }
                exclusions.emplace_back(
                    param_hash_of(envelope.value().params()), hid, "rejected");
            }
        }

        return make(wid.value(), std::move(seeds), std::move(exclusions), built_utc);
    }

    [[nodiscard]] static StatusOr<SearchPrior> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("SearchPrior JSON must be an object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error(
                "SearchPrior.schema must be \"" + std::string(schema_id) + "\"");
        }
        // Soft weights are forbidden in v0.
        if (root.contains("weights") || root.contains("soft_weights") || root.contains("weight")) {
            return Status::error("SearchPrior soft weights are not allowed in v0");
        }

        StatusOr<std::string> workspace_id = require_string(root, "workspace_id");
        if (!workspace_id.ok()) {
            return workspace_id.status();
        }
        StatusOr<std::string> built_utc = require_string(root, "built_utc");
        if (!built_utc.ok()) {
            return built_utc.status();
        }
        if (!root.contains("seeds") || !root.at("seeds").is_array()) {
            return Status::error("SearchPrior.seeds must be an array");
        }
        if (!root.contains("exclusions") || !root.at("exclusions").is_array()) {
            return Status::error("SearchPrior.exclusions must be an array");
        }

        std::vector<Seed> seeds;
        seeds.reserve(root.at("seeds").size());
        for (std::size_t i = 0; i < root.at("seeds").size(); ++i) {
            StatusOr<Seed> seed = seed_from_json(root.at("seeds")[i], i);
            if (!seed.ok()) {
                return seed.status();
            }
            seeds.push_back(std::move(seed.value()));
        }

        std::vector<Exclusion> exclusions;
        exclusions.reserve(root.at("exclusions").size());
        for (std::size_t i = 0; i < root.at("exclusions").size(); ++i) {
            StatusOr<Exclusion> ex = exclusion_from_json(root.at("exclusions")[i], i);
            if (!ex.ok()) {
                return ex.status();
            }
            exclusions.push_back(std::move(ex.value()));
        }

        return make(
            workspace_id.value(),
            std::move(seeds),
            std::move(exclusions),
            built_utc.value());
    }

    [[nodiscard]] static StatusOr<SearchPrior> parse(std::string_view text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("SearchPrior JSON parse error: ") + ex.what());
        }
        return from_json(root);
    }

    [[nodiscard]] static StatusOr<SearchPrior> load_file(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("failed to open SearchPrior file: " + path.string());
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return parse(ss.str());
    }

    [[nodiscard]] nlohmann::json to_json() const {
        nlohmann::json seeds = nlohmann::json::array();
        for (const Seed& seed : seeds_) {
            seeds.push_back(seed.to_json());
        }
        nlohmann::json exclusions = nlohmann::json::array();
        for (const Exclusion& ex : exclusions_) {
            exclusions.push_back(ex.to_json());
        }
        return nlohmann::json{
            {"schema", std::string(schema_id)},
            {"workspace_id", workspace_id_},
            {"seeds", std::move(seeds)},
            {"exclusions", std::move(exclusions)},
            {"built_utc", built_utc_},
        };
    }

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

private:
    [[nodiscard]] static StatusOr<std::string> require_string(
        const nlohmann::json& root,
        std::string_view key) {
        if (!root.contains(key) || !root.at(std::string(key)).is_string()) {
            return Status::error("SearchPrior." + std::string(key) + " must be a string");
        }
        return root.at(std::string(key)).get<std::string>();
    }

    [[nodiscard]] static Status validate_param_hash(std::string_view hash) {
        constexpr std::string_view kPrefix = "sha256:";
        if (hash.size() != kPrefix.size() + 64 || hash.substr(0, kPrefix.size()) != kPrefix) {
            return Status::error(
                "SearchPrior.exclusions.param_hash must be \"sha256:\" + 64 hex digits");
        }
        for (std::size_t i = kPrefix.size(); i < hash.size(); ++i) {
            const unsigned char c = static_cast<unsigned char>(hash[i]);
            if (!std::isxdigit(c) || (std::isalpha(c) && !std::islower(c))) {
                return Status::error(
                    "SearchPrior.exclusions.param_hash hex must be lowercase");
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status validate_seed(const Seed& seed) {
        StatusOr<std::string> hid = WorkspacePaths::validate_id(seed.hypothesis_id_);
        if (!hid.ok()) {
            return hid.status();
        }
        StatusOr<TransformEnvelope> envelope =
            TransformEnvelope::from_json(seed.envelope_);
        if (!envelope.ok()) {
            return Status::error(
                "SearchPrior.seeds.envelope invalid: " + envelope.status().message());
        }
        return Status::success();
    }

    [[nodiscard]] static Status validate_exclusion(const Exclusion& ex) {
        StatusOr<std::string> hid = WorkspacePaths::validate_id(ex.hypothesis_id_);
        if (!hid.ok()) {
            return hid.status();
        }
        if (ex.reason_.empty()) {
            return Status::error("SearchPrior.exclusions.reason must be non-empty");
        }
        return validate_param_hash(ex.param_hash_);
    }

    [[nodiscard]] static StatusOr<Seed> seed_from_json(
        const nlohmann::json& item,
        std::size_t index) {
        if (!item.is_object()) {
            return Status::error(
                "SearchPrior.seeds[" + std::to_string(index) + "] must be an object");
        }
        StatusOr<std::string> hid = require_string(item, "hypothesis_id");
        if (!hid.ok()) {
            return Status::error(
                "SearchPrior.seeds[" + std::to_string(index) + "].hypothesis_id must be a string");
        }
        if (!item.contains("envelope") || !item.at("envelope").is_object()) {
            return Status::error(
                "SearchPrior.seeds[" + std::to_string(index) + "].envelope must be an object");
        }
        StatusOr<TransformEnvelope> envelope =
            TransformEnvelope::from_json(item.at("envelope"));
        if (!envelope.ok()) {
            return Status::error(
                "SearchPrior.seeds[" + std::to_string(index) + "].envelope invalid: " +
                envelope.status().message());
        }
        Seed seed(hid.value(), envelope.value().to_json());
        Status ok = validate_seed(seed);
        if (!ok.ok()) {
            return ok;
        }
        return seed;
    }

    [[nodiscard]] static StatusOr<Exclusion> exclusion_from_json(
        const nlohmann::json& item,
        std::size_t index) {
        if (!item.is_object()) {
            return Status::error(
                "SearchPrior.exclusions[" + std::to_string(index) + "] must be an object");
        }
        StatusOr<std::string> hash = require_string(item, "param_hash");
        if (!hash.ok()) {
            return Status::error(
                "SearchPrior.exclusions[" + std::to_string(index) +
                "].param_hash must be a string");
        }
        StatusOr<std::string> hid = require_string(item, "hypothesis_id");
        if (!hid.ok()) {
            return Status::error(
                "SearchPrior.exclusions[" + std::to_string(index) +
                "].hypothesis_id must be a string");
        }
        std::string reason = "rejected";
        if (item.contains("reason") && !item.at("reason").is_null()) {
            if (!item.at("reason").is_string()) {
                return Status::error(
                    "SearchPrior.exclusions[" + std::to_string(index) +
                    "].reason must be a string");
            }
            reason = item.at("reason").get<std::string>();
        }
        Exclusion ex(hash.value(), hid.value(), std::move(reason));
        Status ok = validate_exclusion(ex);
        if (!ok.ok()) {
            return ok;
        }
        return ex;
    }

    static void sort_seeds(std::vector<Seed>& seeds) {
        std::sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
            return a.hypothesis_id_ < b.hypothesis_id_;
        });
    }

    static void sort_exclusions(std::vector<Exclusion>& exclusions) {
        std::sort(
            exclusions.begin(),
            exclusions.end(),
            [](const Exclusion& a, const Exclusion& b) {
                if (a.param_hash_ != b.param_hash_) {
                    return a.param_hash_ < b.param_hash_;
                }
                return a.hypothesis_id_ < b.hypothesis_id_;
            });
    }

    std::string workspace_id_;
    std::vector<Seed> seeds_;
    std::vector<Exclusion> exclusions_;
    std::string built_utc_;
};

#endif // SEARCH_PRIOR_HPP
