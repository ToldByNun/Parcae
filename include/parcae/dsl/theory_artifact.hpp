#ifndef THEORY_ARTIFACT_HPP
#define THEORY_ARTIFACT_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/version.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"
#include "parcae/dsl/dsl_verifier.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_uri.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Compiled theory manifest `parcae.theory_artifact.v0` (docs/spec/theory-artifact.md).
/// Writers always stamp `dsl_spec_version` = `DslSpecVersion::current` and
/// `compiler_version` = toolkit `PARCAE_VERSION_STRING`. Ready artifacts require
/// `verification.passed == true`.
class TheoryArtifact {
public:
    static constexpr std::string_view schema_id = "parcae.theory_artifact.v0";

    enum class FusionStatus : std::uint8_t {
        NotApplicable = 0,
        Fused,
        FallbackStaged,
    };

    enum class InterruptMode : std::uint8_t {
        PolicyMethod = 0,
        NoneByDesign,
        ElementwiseDefault,
    };

    class Param {
    public:
        Param(std::string name, std::int64_t min, std::int64_t max)
            : name_(std::move(name)), min_(min), max_(max) {}

        [[nodiscard]] const std::string& name() const noexcept { return name_; }

        [[nodiscard]] std::int64_t min() const noexcept { return min_; }

        [[nodiscard]] std::int64_t max() const noexcept { return max_; }

        [[nodiscard]] nlohmann::json to_json() const {
            return nlohmann::json{{"name", name_}, {"min", min_}, {"max", max_}};
        }

    private:
        std::string name_;
        std::int64_t min_ = 0;
        std::int64_t max_ = 0;
    };

    class Verification {
    public:
        Verification(DslVerifier::Mode mode, bool passed, std::optional<std::uint32_t> seed,
                     std::string completed_utc)
            : mode_(mode), passed_(passed), seed_(seed), completed_utc_(std::move(completed_utc)) {}

        [[nodiscard]] DslVerifier::Mode mode() const noexcept { return mode_; }

        [[nodiscard]] bool passed() const noexcept { return passed_; }

        [[nodiscard]] const std::optional<std::uint32_t>& seed() const noexcept { return seed_; }

        [[nodiscard]] const std::string& completed_utc() const noexcept { return completed_utc_; }

        [[nodiscard]] nlohmann::json to_json() const {
            nlohmann::json seed = nullptr;
            if (seed_.has_value()) {
                seed = *seed_;
            }
            return nlohmann::json{
                {"mode", mode_ == DslVerifier::Mode::Exhaustive ? "exhaustive" : "fuzz"},
                {"passed", passed_},
                {"seed", std::move(seed)},
                {"completed_utc", completed_utc_},
            };
        }

    private:
        DslVerifier::Mode mode_ = DslVerifier::Mode::Exhaustive;
        bool passed_ = false;
        std::optional<std::uint32_t> seed_;
        std::string completed_utc_;
    };

    class Paths {
    public:
        Paths() = default;

        [[nodiscard]] const std::optional<std::string>& cpu_reference() const noexcept {
            return cpu_reference_;
        }

        [[nodiscard]] const std::optional<std::string>& cuda_header() const noexcept {
            return cuda_header_;
        }

        [[nodiscard]] const std::optional<std::string>& cuda_source() const noexcept {
            return cuda_source_;
        }

        [[nodiscard]] const std::optional<std::string>& envelope_template() const noexcept {
            return envelope_template_;
        }

        [[nodiscard]] const std::optional<std::string>& apply_ir() const noexcept {
            return apply_ir_;
        }

        [[nodiscard]] const std::optional<std::string>& verify_report() const noexcept {
            return verify_report_;
        }

        void set_cpu_reference(std::optional<std::string> path) {
            cpu_reference_ = std::move(path);
        }

        void set_cuda_header(std::optional<std::string> path) { cuda_header_ = std::move(path); }

        void set_cuda_source(std::optional<std::string> path) { cuda_source_ = std::move(path); }

        void set_envelope_template(std::optional<std::string> path) {
            envelope_template_ = std::move(path);
        }

        void set_apply_ir(std::optional<std::string> path) { apply_ir_ = std::move(path); }

        void set_verify_report(std::optional<std::string> path) {
            verify_report_ = std::move(path);
        }

        [[nodiscard]] nlohmann::json to_json() const {
            return nlohmann::json{
                {"cpu_reference", opt_path(cpu_reference_)},
                {"cuda_header", opt_path(cuda_header_)},
                {"cuda_source", opt_path(cuda_source_)},
                {"envelope_template", opt_path(envelope_template_)},
                {"apply_ir", opt_path(apply_ir_)},
                {"verify_report", opt_path(verify_report_)},
            };
        }

        [[nodiscard]] Status validate() const {
            Status s = check_rel(cpu_reference_, "cpu_reference");
            if (!s.ok()) {
                return s;
            }
            s = check_rel(cuda_header_, "cuda_header");
            if (!s.ok()) {
                return s;
            }
            s = check_rel(cuda_source_, "cuda_source");
            if (!s.ok()) {
                return s;
            }
            s = check_rel(envelope_template_, "envelope_template");
            if (!s.ok()) {
                return s;
            }
            s = check_rel(apply_ir_, "apply_ir");
            if (!s.ok()) {
                return s;
            }
            return check_rel(verify_report_, "verify_report");
        }

    private:
        [[nodiscard]] static nlohmann::json opt_path(const std::optional<std::string>& p) {
            return p.has_value() ? nlohmann::json(*p) : nlohmann::json(nullptr);
        }

        [[nodiscard]] static Status check_rel(const std::optional<std::string>& path,
                                              std::string_view field) {
            if (!path.has_value()) {
                return Status::success();
            }
            return check_relative_path(*path, field);
        }

        std::optional<std::string> cpu_reference_;
        std::optional<std::string> cuda_header_;
        std::optional<std::string> cuda_source_;
        std::optional<std::string> envelope_template_;
        std::optional<std::string> apply_ir_;
        std::optional<std::string> verify_report_;
    };

    [[nodiscard]] static constexpr std::string_view fusion_status_str(FusionStatus s) noexcept {
        switch (s) {
        case FusionStatus::NotApplicable:
            return "n/a";
        case FusionStatus::Fused:
            return "fused";
        case FusionStatus::FallbackStaged:
            return "fallback_staged";
        }
        return "n/a";
    }

    [[nodiscard]] static constexpr std::string_view interrupt_mode_str(InterruptMode m) noexcept {
        switch (m) {
        case InterruptMode::PolicyMethod:
            return "policy_method";
        case InterruptMode::NoneByDesign:
            return "none_by_design";
        case InterruptMode::ElementwiseDefault:
            return "elementwise_default";
        }
        return "unknown";
    }

    [[nodiscard]] static InterruptMode
    interrupt_mode_from_theory(TheoryIr::InterruptMode m) noexcept {
        switch (m) {
        case TheoryIr::InterruptMode::PolicyMethod:
            return InterruptMode::PolicyMethod;
        case TheoryIr::InterruptMode::NoneByDesign:
            return InterruptMode::NoneByDesign;
        case TheoryIr::InterruptMode::ElementwiseDefault:
            return InterruptMode::ElementwiseDefault;
        }
        return InterruptMode::ElementwiseDefault;
    }

    /// Build a ready artifact. Always embeds current `dsl_spec_version` + toolkit version.
    /// `verification.passed` MUST be true (compile MUST NOT write failed gates).
    [[nodiscard]] static StatusOr<TheoryArtifact>
    make(std::string name, std::uint32_t version, TheoryIr::Tier tier, TheoryIr::Family family,
         std::string source_sha256, Verification verification, FusionStatus fusion,
         InterruptMode interrupts, std::vector<Param> params = {},
         std::vector<std::string> primitives = {},
         std::optional<std::string> structural_claim = std::nullopt,
         std::optional<std::string> source_path = std::nullopt, Paths paths = {},
         nlohmann::json sweep = nullptr) {
        StatusOr<TheoryUri> uri = TheoryUri::make(name, version);
        if (!uri.ok()) {
            return uri.status();
        }
        Status sha = check_sha256(source_sha256);
        if (!sha.ok()) {
            return sha;
        }
        if (!verification.passed()) {
            return Status::error(
                "TheoryArtifact: verification.passed must be true for a ready artifact");
        }
        if (verification.mode() == DslVerifier::Mode::Exhaustive &&
            verification.seed().has_value()) {
            return Status::error("TheoryArtifact: exhaustive verification seed must be null");
        }
        if (verification.mode() == DslVerifier::Mode::Fuzz && !verification.seed().has_value()) {
            return Status::error("TheoryArtifact: fuzz verification requires a seed");
        }
        if (verification.completed_utc().empty()) {
            return Status::error("TheoryArtifact: verification.completed_utc must be non-empty");
        }
        if ((tier == TheoryIr::Tier::B || tier == TheoryIr::Tier::C) &&
            (!structural_claim.has_value() || structural_claim->empty())) {
            return Status::error("TheoryArtifact: structural_claim required for tier B/C");
        }
        Status path_ok = paths.validate();
        if (!path_ok.ok()) {
            return path_ok;
        }
        if (!sweep.is_null() && !sweep.is_object()) {
            return Status::error("TheoryArtifact: sweep must be null or an object");
        }
        if (!sweep.is_null() && (tier == TheoryIr::Tier::B || tier == TheoryIr::Tier::C) &&
            (!sweep.contains("compare_against") || sweep.at("compare_against").is_null())) {
            return Status::error(
                "TheoryArtifact: sweep.compare_against required for tier B/C when sweep is set");
        }

        TheoryArtifact a{std::move(uri.value())};
        a.dsl_spec_version_ = std::string(DslSpecVersion::current_string);
        a.compiler_version_ = std::string(PARCAE_VERSION_STRING);
        a.source_sha256_ = std::move(source_sha256);
        a.tier_ = tier;
        a.family_ = family;
        a.structural_claim_ = std::move(structural_claim);
        a.source_path_ = std::move(source_path);
        a.params_ = std::move(params);
        a.primitives_ = std::move(primitives);
        a.verification_ = std::move(verification);
        a.fusion_ = fusion;
        a.interrupts_ = interrupts;
        a.paths_ = std::move(paths);
        a.sweep_ = std::move(sweep);
        return a;
    }

    [[nodiscard]] const TheoryUri& uri() const noexcept { return uri_; }

    [[nodiscard]] const std::string& name() const noexcept { return uri_.name(); }

    [[nodiscard]] std::uint32_t version() const noexcept { return uri_.version(); }

    [[nodiscard]] const std::string& dsl_spec_version() const noexcept { return dsl_spec_version_; }

    [[nodiscard]] const std::string& compiler_version() const noexcept { return compiler_version_; }

    [[nodiscard]] const std::string& source_sha256() const noexcept { return source_sha256_; }

    [[nodiscard]] TheoryIr::Tier tier() const noexcept { return tier_; }

    [[nodiscard]] TheoryIr::Family family() const noexcept { return family_; }

    [[nodiscard]] const std::optional<std::string>& structural_claim() const noexcept {
        return structural_claim_;
    }

    [[nodiscard]] const std::optional<std::string>& source_path() const noexcept {
        return source_path_;
    }

    void set_dsl_ignores_applied(std::vector<std::string> flags) {
        dsl_ignores_applied_ = std::move(flags);
    }

    [[nodiscard]] const std::vector<std::string>& dsl_ignores_applied() const noexcept {
        return dsl_ignores_applied_;
    }

    [[nodiscard]] const std::vector<Param>& params() const noexcept { return params_; }

    [[nodiscard]] const std::vector<std::string>& primitives() const noexcept {
        return primitives_;
    }

    [[nodiscard]] const Verification& verification() const noexcept { return verification_; }

    [[nodiscard]] FusionStatus fusion() const noexcept { return fusion_; }

    [[nodiscard]] InterruptMode interrupts() const noexcept { return interrupts_; }

    [[nodiscard]] const Paths& paths() const noexcept { return paths_; }

    [[nodiscard]] const nlohmann::json& sweep() const noexcept { return sweep_; }

    [[nodiscard]] nlohmann::json to_json() const {
        nlohmann::json params = nlohmann::json::array();
        for (const Param& p : params_) {
            params.push_back(p.to_json());
        }
        nlohmann::json primitives = nlohmann::json::array();
        for (const std::string& name : primitives_) {
            primitives.push_back(name);
        }
        nlohmann::json root{
            {"schema", std::string(schema_id)},
            {"uri", uri_.to_string()},
            {"name", uri_.name()},
            {"version", uri_.version()},
            {"dsl_spec_version", dsl_spec_version_},
            {"compiler_version", compiler_version_},
            {"source_sha256", source_sha256_},
            {"tier", TheoryIr::tier_str(tier_)},
            {"family", TheoryIr::family_str(family_)},
            {"params", std::move(params)},
            {"primitives", std::move(primitives)},
            {"verification", verification_.to_json()},
            {"fusion", nlohmann::json{{"status", std::string(fusion_status_str(fusion_))}}},
            {"paths", paths_.to_json()},
            {"sweep", sweep_},
            {"interrupts", nlohmann::json{{"mode", std::string(interrupt_mode_str(interrupts_))}}},
        };
        if (source_path_.has_value()) {
            root["source_path"] = *source_path_;
        }
        if (structural_claim_.has_value()) {
            root["structural_claim"] = *structural_claim_;
        }
        if (!dsl_ignores_applied_.empty()) {
            root["dsl_ignores_applied"] = dsl_ignores_applied_;
        }
        return root;
    }

    [[nodiscard]] static StatusOr<TheoryArtifact> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("TheoryArtifact must be a JSON object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error("TheoryArtifact.schema must be parcae.theory_artifact.v0");
        }
        if (!root.contains("uri") || !root.at("uri").is_string()) {
            return Status::error("TheoryArtifact.uri must be a string");
        }
        if (!root.contains("name") || !root.at("name").is_string()) {
            return Status::error("TheoryArtifact.name must be a string");
        }
        if (!root.contains("version") || !root.at("version").is_number_unsigned()) {
            return Status::error("TheoryArtifact.version must be a positive integer");
        }
        StatusOr<TheoryUri> uri = TheoryUri::parse(root.at("uri").get<std::string>());
        if (!uri.ok()) {
            return uri.status();
        }
        const std::string name = root.at("name").get<std::string>();
        const auto version = root.at("version").get<std::uint32_t>();
        if (uri.value().name() != name || uri.value().version() != version) {
            return Status::error(
                "TheoryArtifact.uri must equal parcae://theories/{name}@{version}");
        }

        if (!root.contains("dsl_spec_version") || !root.at("dsl_spec_version").is_string()) {
            return Status::error("TheoryArtifact.dsl_spec_version must be a string");
        }
        if (!root.contains("compiler_version") || !root.at("compiler_version").is_string()) {
            return Status::error("TheoryArtifact.compiler_version must be a string");
        }
        if (!root.contains("source_sha256") || !root.at("source_sha256").is_string()) {
            return Status::error("TheoryArtifact.source_sha256 must be a string");
        }
        if (!root.contains("tier") || !root.at("tier").is_string()) {
            return Status::error("TheoryArtifact.tier must be a string");
        }
        if (!root.contains("family") || !root.at("family").is_string()) {
            return Status::error("TheoryArtifact.family must be a string");
        }

        StatusOr<TheoryIr::Tier> tier = TheoryIr::parse_tier(root.at("tier").get<std::string>());
        if (!tier.ok()) {
            return tier.status();
        }
        StatusOr<TheoryIr::Family> family =
            TheoryIr::parse_family(root.at("family").get<std::string>());
        if (!family.ok()) {
            return family.status();
        }

        StatusOr<Verification> verification = parse_verification(root);
        if (!verification.ok()) {
            return verification.status();
        }
        StatusOr<FusionStatus> fusion = parse_fusion(root);
        if (!fusion.ok()) {
            return fusion.status();
        }
        StatusOr<InterruptMode> interrupts = parse_interrupts(root);
        if (!interrupts.ok()) {
            return interrupts.status();
        }
        StatusOr<std::vector<Param>> params = parse_params(root);
        if (!params.ok()) {
            return params.status();
        }
        StatusOr<std::vector<std::string>> primitives = parse_primitives(root);
        if (!primitives.ok()) {
            return primitives.status();
        }
        StatusOr<Paths> paths = parse_paths(root);
        if (!paths.ok()) {
            return paths.status();
        }

        std::optional<std::string> structural_claim;
        if (root.contains("structural_claim") && root.at("structural_claim").is_string()) {
            structural_claim = root.at("structural_claim").get<std::string>();
        }
        std::optional<std::string> source_path;
        if (root.contains("source_path") && root.at("source_path").is_string()) {
            source_path = root.at("source_path").get<std::string>();
        }
        nlohmann::json sweep = nullptr;
        if (root.contains("sweep") && !root.at("sweep").is_null()) {
            if (!root.at("sweep").is_object()) {
                return Status::error("TheoryArtifact.sweep must be null or an object");
            }
            sweep = root.at("sweep");
        }

        // Preserve on-disk dsl_spec_version / compiler_version (registry checks later).
        TheoryArtifact a{std::move(uri.value())};
        a.dsl_spec_version_ = root.at("dsl_spec_version").get<std::string>();
        a.compiler_version_ = root.at("compiler_version").get<std::string>();
        a.source_sha256_ = root.at("source_sha256").get<std::string>();
        Status sha = check_sha256(a.source_sha256_);
        if (!sha.ok()) {
            return sha;
        }
        a.tier_ = tier.value();
        a.family_ = family.value();
        a.structural_claim_ = std::move(structural_claim);
        a.source_path_ = std::move(source_path);
        a.params_ = std::move(params.value());
        a.primitives_ = std::move(primitives.value());
        a.verification_ = std::move(verification.value());
        a.fusion_ = fusion.value();
        a.interrupts_ = interrupts.value();
        a.paths_ = std::move(paths.value());
        a.sweep_ = std::move(sweep);

        if (root.contains("dsl_ignores_applied")) {
            if (!root.at("dsl_ignores_applied").is_array()) {
                return Status::error("TheoryArtifact.dsl_ignores_applied must be an array");
            }
            for (const auto& item : root.at("dsl_ignores_applied")) {
                if (!item.is_string()) {
                    return Status::error(
                        "TheoryArtifact.dsl_ignores_applied entries must be strings");
                }
                a.dsl_ignores_applied_.push_back(item.get<std::string>());
            }
        }

        if ((a.tier_ == TheoryIr::Tier::B || a.tier_ == TheoryIr::Tier::C) &&
            (!a.structural_claim_.has_value() || a.structural_claim_->empty())) {
            return Status::error("TheoryArtifact: structural_claim required for tier B/C");
        }
        if (!a.sweep_.is_null() && (a.tier_ == TheoryIr::Tier::B || a.tier_ == TheoryIr::Tier::C) &&
            (!a.sweep_.contains("compare_against") || a.sweep_.at("compare_against").is_null())) {
            return Status::error(
                "TheoryArtifact: sweep.compare_against required for tier B/C when sweep is set");
        }
        if (!a.verification_.passed()) {
            return Status::error(
                "TheoryArtifact: verification.passed must be true for a ready artifact");
        }
        return a;
    }

    [[nodiscard]] static std::filesystem::path
    artifact_dir(const std::filesystem::path& theories_root, std::string_view name,
                 std::uint32_t version) {
        return theories_root / std::string(name) / std::to_string(version);
    }

    [[nodiscard]] std::filesystem::path
    artifact_dir(const std::filesystem::path& theories_root) const {
        return artifact_dir(theories_root, uri_.name(), uri_.version());
    }

    [[nodiscard]] std::filesystem::path
    manifest_path(const std::filesystem::path& theories_root) const {
        return artifact_dir(theories_root) / "manifest.json";
    }

    /// Write `manifest.json` under `theories_root/<name>/<version>/`.
    /// Refuses `verification.passed == false`. Always serializes the embedded
    /// `dsl_spec_version` (stamped at `make` as current).
    [[nodiscard]] Status store(const std::filesystem::path& theories_root) const {
        if (!verification_.passed()) {
            return Status::error("TheoryArtifact::store refuses verification.passed == false");
        }
        if (dsl_spec_version_.empty()) {
            return Status::error("TheoryArtifact::store: dsl_spec_version must be embedded");
        }
        const std::filesystem::path dir = artifact_dir(theories_root);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            return Status::error("Failed to create artifact directory: " + ec.message());
        }
        const std::filesystem::path path = dir / "manifest.json";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::error("Failed to write manifest.json: " + path.string());
        }
        out << to_json().dump(2) << '\n';
        if (!out) {
            return Status::error("Failed while writing manifest.json");
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<TheoryArtifact>
    load(const std::filesystem::path& theories_root, std::string_view name, std::uint32_t version) {
        StatusOr<TheoryUri> uri = TheoryUri::make(name, version);
        if (!uri.ok()) {
            return uri.status();
        }
        const std::filesystem::path path =
            artifact_dir(theories_root, uri.value().name(), uri.value().version()) /
            "manifest.json";
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("Failed to open manifest.json: " + path.string());
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        nlohmann::json json;
        try {
            json = nlohmann::json::parse(buf.str());
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid manifest.json: ") + ex.what());
        }
        StatusOr<TheoryArtifact> a = from_json(json);
        if (!a.ok()) {
            return a.status();
        }
        if (a.value().name() != uri.value().name() ||
            a.value().version() != uri.value().version()) {
            return Status::error("manifest name/version does not match artifact directory");
        }
        return a;
    }

    [[nodiscard]] static Status check_relative_path(std::string_view path, std::string_view field) {
        if (path.empty()) {
            return Status::error("TheoryArtifact.paths." + std::string(field) +
                                 " must be non-empty when set");
        }
        if (path.find(':') != std::string_view::npos || path.front() == '/' ||
            path.front() == '\\') {
            return Status::error("TheoryArtifact.paths." + std::string(field) +
                                 " must be relative (no absolute / drive paths)");
        }
        const std::filesystem::path p{std::string(path)};
        for (const std::filesystem::path& part : p) {
            if (part == "..") {
                return Status::error("TheoryArtifact.paths." + std::string(field) +
                                     " must not escape via '..'");
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status check_sha256(std::string_view hex) {
        if (hex.size() != 64) {
            return Status::error("source_sha256 must be 64 lowercase hex characters");
        }
        for (char ch : hex) {
            if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
                return Status::error("source_sha256 must be 64 lowercase hex characters");
            }
        }
        return Status::success();
    }

private:
    explicit TheoryArtifact(TheoryUri uri) : uri_(std::move(uri)) {}

    [[nodiscard]] static StatusOr<Verification> parse_verification(const nlohmann::json& root) {
        if (!root.contains("verification") || !root.at("verification").is_object()) {
            return Status::error("TheoryArtifact.verification must be an object");
        }
        const nlohmann::json& v = root.at("verification");
        if (!v.contains("mode") || !v.at("mode").is_string()) {
            return Status::error("TheoryArtifact.verification.mode must be a string");
        }
        if (!v.contains("passed") || !v.at("passed").is_boolean()) {
            return Status::error("TheoryArtifact.verification.passed must be a boolean");
        }
        if (!v.contains("completed_utc") || !v.at("completed_utc").is_string()) {
            return Status::error("TheoryArtifact.verification.completed_utc must be a string");
        }
        const std::string mode_s = v.at("mode").get<std::string>();
        DslVerifier::Mode mode = DslVerifier::Mode::Exhaustive;
        if (mode_s == "exhaustive") {
            mode = DslVerifier::Mode::Exhaustive;
        } else if (mode_s == "fuzz") {
            mode = DslVerifier::Mode::Fuzz;
        } else {
            return Status::error("TheoryArtifact.verification.mode must be exhaustive|fuzz");
        }
        std::optional<std::uint32_t> seed;
        if (v.contains("seed") && !v.at("seed").is_null()) {
            if (!v.at("seed").is_number_unsigned()) {
                return Status::error("TheoryArtifact.verification.seed must be null or unsigned");
            }
            seed = v.at("seed").get<std::uint32_t>();
        }
        if (mode == DslVerifier::Mode::Exhaustive && seed.has_value()) {
            return Status::error("TheoryArtifact: exhaustive verification seed must be null");
        }
        if (mode == DslVerifier::Mode::Fuzz && !seed.has_value()) {
            return Status::error("TheoryArtifact: fuzz verification requires a seed");
        }
        return Verification{
            mode,
            v.at("passed").get<bool>(),
            seed,
            v.at("completed_utc").get<std::string>(),
        };
    }

    [[nodiscard]] static StatusOr<FusionStatus> parse_fusion(const nlohmann::json& root) {
        if (!root.contains("fusion") || !root.at("fusion").is_object()) {
            return Status::error("TheoryArtifact.fusion must be an object");
        }
        const nlohmann::json& f = root.at("fusion");
        if (!f.contains("status") || !f.at("status").is_string()) {
            return Status::error("TheoryArtifact.fusion.status must be a string");
        }
        const std::string s = f.at("status").get<std::string>();
        if (s == "n/a") {
            return FusionStatus::NotApplicable;
        }
        if (s == "fused") {
            return FusionStatus::Fused;
        }
        if (s == "fallback_staged") {
            return FusionStatus::FallbackStaged;
        }
        return Status::error("TheoryArtifact.fusion.status must be n/a|fused|fallback_staged");
    }

    [[nodiscard]] static StatusOr<InterruptMode> parse_interrupts(const nlohmann::json& root) {
        if (!root.contains("interrupts") || !root.at("interrupts").is_object()) {
            return Status::error("TheoryArtifact.interrupts must be an object");
        }
        const nlohmann::json& i = root.at("interrupts");
        if (!i.contains("mode") || !i.at("mode").is_string()) {
            return Status::error("TheoryArtifact.interrupts.mode must be a string");
        }
        const std::string m = i.at("mode").get<std::string>();
        if (m == "policy_method") {
            return InterruptMode::PolicyMethod;
        }
        if (m == "none_by_design") {
            return InterruptMode::NoneByDesign;
        }
        if (m == "elementwise_default") {
            return InterruptMode::ElementwiseDefault;
        }
        return Status::error("TheoryArtifact.interrupts.mode must be "
                             "policy_method|none_by_design|elementwise_default");
    }

    [[nodiscard]] static StatusOr<std::vector<Param>> parse_params(const nlohmann::json& root) {
        if (!root.contains("params") || !root.at("params").is_array()) {
            return Status::error("TheoryArtifact.params must be an array");
        }
        std::vector<Param> out;
        for (const nlohmann::json& item : root.at("params")) {
            if (!item.is_object() || !item.contains("name") || !item.at("name").is_string() ||
                !item.contains("min") || !item.at("min").is_number_integer() ||
                !item.contains("max") || !item.at("max").is_number_integer()) {
                return Status::error("TheoryArtifact.params entries must be {name,min,max}");
            }
            out.emplace_back(item.at("name").get<std::string>(), item.at("min").get<std::int64_t>(),
                             item.at("max").get<std::int64_t>());
        }
        return out;
    }

    [[nodiscard]] static StatusOr<std::vector<std::string>>
    parse_primitives(const nlohmann::json& root) {
        if (!root.contains("primitives") || !root.at("primitives").is_array()) {
            return Status::error("TheoryArtifact.primitives must be an array");
        }
        std::vector<std::string> out;
        for (const nlohmann::json& item : root.at("primitives")) {
            if (!item.is_string()) {
                return Status::error("TheoryArtifact.primitives entries must be strings");
            }
            out.push_back(item.get<std::string>());
        }
        return out;
    }

    [[nodiscard]] static StatusOr<Paths> parse_paths(const nlohmann::json& root) {
        if (!root.contains("paths") || !root.at("paths").is_object()) {
            return Status::error("TheoryArtifact.paths must be an object");
        }
        const nlohmann::json& p = root.at("paths");
        Paths paths;

        auto read_opt = [&](const char* key) -> StatusOr<std::optional<std::string>> {
            if (!p.contains(key) || p.at(key).is_null()) {
                return std::optional<std::string>{};
            }
            if (!p.at(key).is_string()) {
                return Status::error(std::string("TheoryArtifact.paths.") + key +
                                     " must be string or null");
            }
            return std::optional<std::string>{p.at(key).get<std::string>()};
        };

        StatusOr<std::optional<std::string>> cpu = read_opt("cpu_reference");
        if (!cpu.ok()) {
            return cpu.status();
        }
        paths.set_cpu_reference(std::move(cpu.value()));

        StatusOr<std::optional<std::string>> hdr = read_opt("cuda_header");
        if (!hdr.ok()) {
            return hdr.status();
        }
        paths.set_cuda_header(std::move(hdr.value()));

        StatusOr<std::optional<std::string>> src = read_opt("cuda_source");
        if (!src.ok()) {
            return src.status();
        }
        paths.set_cuda_source(std::move(src.value()));

        StatusOr<std::optional<std::string>> env = read_opt("envelope_template");
        if (!env.ok()) {
            return env.status();
        }
        paths.set_envelope_template(std::move(env.value()));

        StatusOr<std::optional<std::string>> apply = read_opt("apply_ir");
        if (!apply.ok()) {
            return apply.status();
        }
        paths.set_apply_ir(std::move(apply.value()));

        StatusOr<std::optional<std::string>> rep = read_opt("verify_report");
        if (!rep.ok()) {
            return rep.status();
        }
        paths.set_verify_report(std::move(rep.value()));

        Status s = paths.validate();
        if (!s.ok()) {
            return s;
        }
        return paths;
    }

    TheoryUri uri_;
    std::string dsl_spec_version_;
    std::string compiler_version_;
    std::string source_sha256_;
    TheoryIr::Tier tier_ = TheoryIr::Tier::A;
    TheoryIr::Family family_ = TheoryIr::Family::Elementwise;
    std::optional<std::string> structural_claim_;
    std::optional<std::string> source_path_;
    std::vector<std::string> dsl_ignores_applied_;
    std::vector<Param> params_;
    std::vector<std::string> primitives_;
    Verification verification_{
        DslVerifier::Mode::Exhaustive,
        false,
        std::nullopt,
        {},
    };
    FusionStatus fusion_ = FusionStatus::NotApplicable;
    InterruptMode interrupts_ = InterruptMode::ElementwiseDefault;
    Paths paths_;
    nlohmann::json sweep_ = nullptr;
};

#endif // THEORY_ARTIFACT_HPP
