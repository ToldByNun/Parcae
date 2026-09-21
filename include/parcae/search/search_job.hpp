#ifndef SEARCH_JOB_HPP
#define SEARCH_JOB_HPP

#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/search/search_prior.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Workspace search request (`parcae.search_job.v0` — docs/spec/search-loop.md).
class SearchJob {
public:
    static constexpr std::string_view schema_id = "parcae.search_job.v0";

    SearchJob() = default;

    [[nodiscard]] const std::string& workspace_id() const noexcept {
        return workspace_id_;
    }

    [[nodiscard]] const std::string& family() const noexcept {
        return family_;
    }

    [[nodiscard]] const std::string& score_id() const noexcept {
        return score_id_;
    }

    [[nodiscard]] const std::string& score_version() const noexcept {
        return score_version_;
    }

    [[nodiscard]] std::size_t k() const noexcept {
        return k_;
    }

    [[nodiscard]] std::uint32_t seed() const noexcept {
        return seed_;
    }

    [[nodiscard]] parcae::tool::Backend backend() const noexcept {
        return backend_;
    }

    [[nodiscard]] std::size_t max_candidates() const noexcept {
        return max_candidates_;
    }

    [[nodiscard]] TransformDirection direction() const noexcept {
        return direction_;
    }

    /// Family-specific bounds; empty object means “use family defaults”.
    [[nodiscard]] const nlohmann::json& param_grid() const noexcept {
        return param_grid_;
    }

    /// Inline `parcae.search_prior.v0` when present; otherwise nullopt (load from workspace).
    [[nodiscard]] const std::optional<nlohmann::json>& prior() const noexcept {
        return prior_;
    }

    /// When true, `beaufort` / `totient` families are accepted (opt-in).
    [[nodiscard]] bool allow_extended_families() const noexcept {
        return allow_extended_families_;
    }

    [[nodiscard]] static bool is_v0_family(std::string_view family) noexcept {
        return family == "caesar" || family == "atbash" || family == "atbash_caesar" ||
               family == "affine" || family == "vigenere";
    }

    [[nodiscard]] static bool is_extended_family(std::string_view family) noexcept {
        return family == "beaufort" || family == "totient";
    }

    [[nodiscard]] static StatusOr<std::string> validate_family(
        std::string_view family,
        bool allow_extended) {
        if (family.empty()) {
            return Status::error("SearchJob.family must be non-empty");
        }
        if (is_v0_family(family)) {
            return std::string(family);
        }
        if (allow_extended && is_extended_family(family)) {
            return std::string(family);
        }
        if (is_extended_family(family)) {
            return Status::error(
                "SearchJob.family requires allow_extended_families: " + std::string(family));
        }
        return Status::error(
            "SearchJob.family unknown (expected caesar|atbash|atbash_caesar|affine|vigenere"
            " or opt-in beaufort|totient): " +
            std::string(family));
    }

    [[nodiscard]] static StatusOr<std::string> validate_score_id(std::string_view score_id) {
        if (score_id.empty() || score_id.size() > 128) {
            return Status::error("SearchJob.score_id length must be 1..128");
        }
        return std::string(score_id);
    }

    /// Build a validated job (does not check that the workspace directory exists).
    [[nodiscard]] static StatusOr<SearchJob> make(
        std::string_view workspace_id,
        std::string_view family,
        std::string_view score_id,
        std::size_t k,
        std::uint32_t seed,
        parcae::tool::Backend backend,
        std::size_t max_candidates,
        TransformDirection direction = TransformDirection::Decrypt,
        nlohmann::json param_grid = nlohmann::json::object(),
        std::optional<nlohmann::json> prior = std::nullopt,
        std::string_view score_version = "v0",
        bool allow_extended_families = false) {
        StatusOr<std::string> wid = WorkspacePaths::validate_id(workspace_id);
        if (!wid.ok()) {
            return wid.status();
        }
        StatusOr<std::string> fam = validate_family(family, allow_extended_families);
        if (!fam.ok()) {
            return fam.status();
        }
        StatusOr<std::string> sid = validate_score_id(score_id);
        if (!sid.ok()) {
            return sid.status();
        }
        if (k == 0) {
            return Status::error("SearchJob.k must be >= 1");
        }
        if (max_candidates < k) {
            return Status::error("SearchJob.max_candidates must be >= k");
        }
        if (score_version.empty() || score_version.size() > 32) {
            return Status::error("SearchJob.score_version length must be 1..32");
        }
        if (!param_grid.is_object() && !param_grid.is_null()) {
            return Status::error("SearchJob.param_grid must be an object or null");
        }
        if (prior.has_value()) {
            Status prior_ok = validate_inline_prior(*prior);
            if (!prior_ok.ok()) {
                return prior_ok;
            }
        }

        SearchJob job;
        job.workspace_id_ = std::move(wid.value());
        job.family_ = std::move(fam.value());
        job.score_id_ = std::move(sid.value());
        job.score_version_ = std::string(score_version);
        job.k_ = k;
        job.seed_ = seed;
        job.backend_ = backend;
        job.max_candidates_ = max_candidates;
        job.direction_ = direction;
        job.param_grid_ = param_grid.is_null() ? nlohmann::json::object() : std::move(param_grid);
        job.prior_ = std::move(prior);
        job.allow_extended_families_ = allow_extended_families;
        return job;
    }

    [[nodiscard]] static StatusOr<SearchJob> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("SearchJob JSON must be an object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error(
                "SearchJob.schema must be \"" + std::string(schema_id) + "\"");
        }

        StatusOr<std::string> workspace_id = require_string(root, "workspace_id");
        if (!workspace_id.ok()) {
            return workspace_id.status();
        }
        StatusOr<std::string> family = require_string(root, "family");
        if (!family.ok()) {
            return family.status();
        }
        StatusOr<std::string> score_id = require_string(root, "score_id");
        if (!score_id.ok()) {
            return score_id.status();
        }

        // nlohmann stores small integers as signed; accept any integer >= 0.
        if (!root.contains("k") || !root.at("k").is_number_integer()) {
            return Status::error("SearchJob.k must be an integer");
        }
        if (!root.contains("seed") || !root.at("seed").is_number_integer()) {
            return Status::error("SearchJob.seed must be an integer");
        }
        if (!root.contains("backend") || !root.at("backend").is_string()) {
            return Status::error("SearchJob.backend must be a string");
        }
        if (!root.contains("max_candidates") || !root.at("max_candidates").is_number_integer()) {
            return Status::error("SearchJob.max_candidates must be an integer");
        }

        const auto k_i = root.at("k").get<std::int64_t>();
        const auto seed_i = root.at("seed").get<std::int64_t>();
        const auto max_i = root.at("max_candidates").get<std::int64_t>();
        if (k_i < 1 || static_cast<std::uint64_t>(k_i) > static_cast<std::uint64_t>(SIZE_MAX)) {
            return Status::error("SearchJob.k out of range");
        }
        if (seed_i < 0 || seed_i > 0xFFFFFFFFll) {
            return Status::error("SearchJob.seed must fit in uint32");
        }
        if (max_i < 1 || static_cast<std::uint64_t>(max_i) > static_cast<std::uint64_t>(SIZE_MAX)) {
            return Status::error("SearchJob.max_candidates out of range");
        }
        const std::uint64_t k_u = static_cast<std::uint64_t>(k_i);
        const std::uint64_t seed_u = static_cast<std::uint64_t>(seed_i);
        const std::uint64_t max_u = static_cast<std::uint64_t>(max_i);

        StatusOr<parcae::tool::Backend> backend =
            parcae::tool::BackendUtil::from_string(root.at("backend").get<std::string>());
        if (!backend.ok()) {
            return backend.status();
        }

        TransformDirection direction = TransformDirection::Decrypt;
        if (root.contains("direction") && !root.at("direction").is_null()) {
            if (!root.at("direction").is_string()) {
                return Status::error("SearchJob.direction must be a string");
            }
            StatusOr<TransformDirection> dir =
                TransformDirectionUtil::from_string(root.at("direction").get<std::string>());
            if (!dir.ok()) {
                return dir.status();
            }
            direction = dir.value();
        }

        std::string score_version = "v0";
        if (root.contains("score_version") && !root.at("score_version").is_null()) {
            if (!root.at("score_version").is_string()) {
                return Status::error("SearchJob.score_version must be a string");
            }
            score_version = root.at("score_version").get<std::string>();
        }

        nlohmann::json param_grid = nlohmann::json::object();
        if (root.contains("param_grid") && !root.at("param_grid").is_null()) {
            if (!root.at("param_grid").is_object()) {
                return Status::error("SearchJob.param_grid must be an object or null");
            }
            param_grid = root.at("param_grid");
        }

        std::optional<nlohmann::json> prior;
        if (root.contains("prior") && !root.at("prior").is_null()) {
            prior = root.at("prior");
        }

        bool allow_extended = false;
        if (root.contains("allow_extended_families") &&
            !root.at("allow_extended_families").is_null()) {
            if (!root.at("allow_extended_families").is_boolean()) {
                return Status::error("SearchJob.allow_extended_families must be a boolean");
            }
            allow_extended = root.at("allow_extended_families").get<bool>();
        }

        return make(
            workspace_id.value(),
            family.value(),
            score_id.value(),
            static_cast<std::size_t>(k_u),
            static_cast<std::uint32_t>(seed_u),
            backend.value(),
            static_cast<std::size_t>(max_u),
            direction,
            std::move(param_grid),
            std::move(prior),
            score_version,
            allow_extended);
    }

    [[nodiscard]] static StatusOr<SearchJob> parse(std::string_view text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("SearchJob JSON parse error: ") + ex.what());
        }
        return from_json(root);
    }

    [[nodiscard]] static StatusOr<SearchJob> load_file(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("failed to open SearchJob file: " + path.string());
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return parse(ss.str());
    }

    /// Fail if `data_root/workspaces/<workspace_id>/` is missing.
    [[nodiscard]] Status require_workspace_dir(const std::filesystem::path& data_root) const {
        StatusOr<std::filesystem::path> root =
            WorkspacePaths::workspace_root(data_root, workspace_id_);
        if (!root.ok()) {
            return root.status();
        }
        std::error_code ec;
        if (!std::filesystem::is_directory(root.value(), ec) || ec) {
            return Status::error(
                "SearchJob workspace directory missing: " + root.value().string());
        }
        return Status::success();
    }

    [[nodiscard]] nlohmann::json to_json() const {
        nlohmann::json j{
            {"schema", std::string(schema_id)},
            {"workspace_id", workspace_id_},
            {"family", family_},
            {"score_id", score_id_},
            {"score_version", score_version_},
            {"k", k_},
            {"seed", seed_},
            {"backend", std::string(parcae::tool::BackendUtil::to_string(backend_))},
            {"max_candidates", max_candidates_},
            {"direction", std::string(TransformDirectionUtil::to_string(direction_))},
            {"param_grid", param_grid_.empty() ? nlohmann::json(nullptr) : param_grid_},
            {"prior", prior_.has_value() ? *prior_ : nlohmann::json(nullptr)},
        };
        if (allow_extended_families_) {
            j["allow_extended_families"] = true;
        }
        return j;
    }

    /// Canonical SHA-256 of replay-relevant fields (excludes timing; includes prior).
    [[nodiscard]] std::string job_digest_sha256() const {
        return Sha256::hex_digest(canonical_dump(to_json()));
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
            return Status::error(
                "SearchJob." + std::string(key) + " must be a string");
        }
        return root.at(std::string(key)).get<std::string>();
    }

    [[nodiscard]] static Status validate_inline_prior(const nlohmann::json& prior) {
        StatusOr<SearchPrior> parsed = SearchPrior::from_json(prior);
        if (!parsed.ok()) {
            return parsed.status();
        }
        return Status::success();
    }

    std::string workspace_id_;
    std::string family_;
    std::string score_id_;
    std::string score_version_ = "v0";
    std::size_t k_ = 1;
    std::uint32_t seed_ = 1;
    parcae::tool::Backend backend_ = parcae::tool::Backend::Cpu;
    std::size_t max_candidates_ = 1;
    TransformDirection direction_ = TransformDirection::Decrypt;
    nlohmann::json param_grid_ = nlohmann::json::object();
    std::optional<nlohmann::json> prior_;
    bool allow_extended_families_ = false;
};

#endif // SEARCH_JOB_HPP
