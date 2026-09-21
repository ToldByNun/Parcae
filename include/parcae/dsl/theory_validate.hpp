#ifndef THEORY_VALIDATE_HPP
#define THEORY_VALIDATE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_envelope_bridge.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"
#include "parcae/dsl/theory_uri.hpp"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Validate compiled theory artifacts (docs/spec/theory-artifact.md).
/// Checks: manifest load, `dsl_spec_version` compatibility (stale → fail),
/// `verification.passed`, declared `paths.*` files on disk, and when
/// `paths.envelope_template` is set that the file parses as a theory/catalog
/// envelope via TheoryEnvelopeBridge.
/// Does not re-run exhaustive/fuzz verify (that is `parcae-compile`).
class TheoryValidate {
public:
    class Check {
    public:
        Check(std::string name, bool ok, std::string message)
            : name_(std::move(name)), ok_(ok), message_(std::move(message)) {}

        [[nodiscard]] const std::string& name() const noexcept {
            return name_;
        }

        [[nodiscard]] bool ok() const noexcept {
            return ok_;
        }

        [[nodiscard]] const std::string& message() const noexcept {
            return message_;
        }

    private:
        std::string name_;
        bool ok_ = false;
        std::string message_;
    };

    class Report {
    public:
        [[nodiscard]] bool ok() const noexcept {
            return ok_;
        }

        [[nodiscard]] const std::string& uri() const noexcept {
            return uri_;
        }

        [[nodiscard]] const std::vector<Check>& checks() const noexcept {
            return checks_;
        }

        [[nodiscard]] const std::optional<std::string>& dsl_spec_version() const noexcept {
            return dsl_spec_version_;
        }

        void set_uri(std::string uri) {
            uri_ = std::move(uri);
        }

        void set_dsl_spec_version(std::string v) {
            dsl_spec_version_ = std::move(v);
        }

        void add_check(std::string name, bool passed, std::string message) {
            if (!passed) {
                ok_ = false;
            }
            checks_.emplace_back(std::move(name), passed, std::move(message));
        }

    private:
        bool ok_ = true;
        std::string uri_;
        std::optional<std::string> dsl_spec_version_;
        std::vector<Check> checks_;
    };

    /// Validate one artifact under `theories_root` by name + version.
    [[nodiscard]] static Report validate(
        const std::filesystem::path& theories_root,
        std::string_view name,
        std::uint32_t version) {
        Report report;
        StatusOr<TheoryUri> uri = TheoryUri::make(name, version);
        if (!uri.ok()) {
            report.set_uri(std::string(name) + "@" + std::to_string(version));
            report.add_check("uri", false, uri.status().message());
            return report;
        }
        report.set_uri(uri.value().to_string());

        StatusOr<TheoryArtifact> artifact =
            TheoryArtifact::load(theories_root, uri.value().name(), uri.value().version());
        if (!artifact.ok()) {
            report.add_check("manifest", false, artifact.status().message());
            return report;
        }
        report.add_check("manifest", true, "manifest.json loaded");
        return validate_loaded(theories_root, artifact.value(), report);
    }

    /// Validate by `parcae://theories/<name>@<ver>` or bare `<name>@<ver>`.
    [[nodiscard]] static Report validate_uri(
        const std::filesystem::path& theories_root,
        std::string_view uri_or_ref) {
        Report report;
        StatusOr<TheoryUri> uri = parse_ref(uri_or_ref);
        if (!uri.ok()) {
            report.set_uri(std::string(uri_or_ref));
            report.add_check("uri", false, uri.status().message());
            return report;
        }
        return validate(theories_root, uri.value().name(), uri.value().version());
    }

    /// Resolve a CLI target: URI, `name@version`, or filesystem path to a version
    /// directory / `manifest.json`. Relative paths are under `theories_root`.
    [[nodiscard]] static Report validate_target(
        const std::filesystem::path& theories_root,
        std::string_view target) {
        if (target.empty()) {
            Report report;
            report.set_uri("");
            report.add_check("target", false, "empty theory target");
            return report;
        }

        // URI / name@version first (no filesystem ambiguity).
        if (looks_like_uri_or_at_ref(target)) {
            return validate_uri(theories_root, target);
        }

        std::filesystem::path p{std::string(target)};
        std::error_code ec;
        if (!p.is_absolute()) {
            // Try as theories_root-relative path (e.g. name/1 or name/1/manifest.json).
            const std::filesystem::path under = theories_root / p;
            if (std::filesystem::exists(under, ec) && !ec) {
                p = under;
            }
        }
        if (std::filesystem::is_regular_file(p, ec) && !ec &&
            p.filename() == "manifest.json") {
            p = p.parent_path();
        }
        if (!std::filesystem::is_directory(p, ec) || ec) {
            // Fall back: treat as URI-ish if path does not exist.
            if (looks_like_uri_or_at_ref(target) || target.find('@') != std::string_view::npos) {
                return validate_uri(theories_root, target);
            }
            Report report;
            report.set_uri(std::string(target));
            report.add_check(
                "target",
                false,
                "theory target is not a URI, name@version, or artifact directory: " +
                    std::string(target));
            return report;
        }

        const std::string ver_text = p.filename().string();
        const std::string name = p.parent_path().filename().string();
        StatusOr<std::uint32_t> version = parse_version_dir(ver_text);
        if (!version.ok()) {
            Report report;
            report.set_uri(p.string());
            report.add_check(
                "target",
                false,
                "artifact directory must end with <name>/<version>: " + p.string());
            return report;
        }
        // Prefer theories_root-relative load when path is under it.
        return validate(theories_root, name, version.value());
    }

    /// Validate every catalogable artifact under `theories_root`.
    /// Empty root ⇒ empty ok list (not an error).
    [[nodiscard]] static StatusOr<std::vector<Report>> validate_all(
        const std::filesystem::path& theories_root) {
        StatusOr<std::vector<TheoryRegistry::CatalogEntry>> entries =
            TheoryRegistry::list(theories_root);
        if (!entries.ok()) {
            return entries.status();
        }
        std::vector<Report> out;
        out.reserve(entries.value().size());
        for (const TheoryRegistry::CatalogEntry& e : entries.value()) {
            out.push_back(validate(theories_root, e.uri().name(), e.uri().version()));
        }
        return out;
    }

private:
    TheoryValidate() = delete;

    [[nodiscard]] static bool looks_like_uri_or_at_ref(std::string_view target) {
        return target.rfind("parcae://", 0) == 0 ||
               target.find('@') != std::string_view::npos;
    }

    [[nodiscard]] static StatusOr<TheoryUri> parse_ref(std::string_view text) {
        if (text.rfind("parcae://", 0) == 0) {
            return TheoryUri::parse(text);
        }
        // Bare name@version
        const auto at = text.find('@');
        if (at == std::string_view::npos || at == 0 || at + 1 >= text.size()) {
            return Status::error(
                "theory ref must be parcae://theories/<name>@<ver> or <name>@<ver>");
        }
        const std::string_view name = text.substr(0, at);
        const std::string_view ver = text.substr(at + 1);
        StatusOr<std::uint32_t> version = parse_version_dir(ver);
        if (!version.ok()) {
            return Status::error("invalid theory version in ref: " + std::string(text));
        }
        return TheoryUri::make(std::string(name), version.value());
    }

    [[nodiscard]] static StatusOr<std::uint32_t> parse_version_dir(std::string_view text) {
        if (text.empty() || (text.size() > 1 && text[0] == '0') ||
            text.find_first_not_of("0123456789") != std::string_view::npos) {
            return Status::error("invalid version");
        }
        std::uint32_t v = 0;
        for (char c : text) {
            const std::uint32_t digit = static_cast<std::uint32_t>(c - '0');
            if (v > (std::numeric_limits<std::uint32_t>::max() - digit) / 10u) {
                return Status::error("version overflow");
            }
            v = v * 10u + digit;
        }
        if (v == 0) {
            return Status::error("version must be >= 1");
        }
        return v;
    }

    [[nodiscard]] static Report validate_loaded(
        const std::filesystem::path& theories_root,
        const TheoryArtifact& artifact,
        Report report) {
        report.set_dsl_spec_version(artifact.dsl_spec_version());

        const Status spec = TheoryRegistry::check_dsl_spec(artifact);
        if (spec.ok()) {
            report.add_check(
                "dsl_spec_version",
                true,
                "compatible with " + std::string(DslSpecVersion::current_string));
        } else {
            report.add_check("dsl_spec_version", false, spec.message());
        }

        if (artifact.verification().passed()) {
            report.add_check("verification.passed", true, "true");
        } else {
            report.add_check(
                "verification.passed",
                false,
                "manifest verification.passed is false");
        }

        const std::filesystem::path dir = artifact.artifact_dir(theories_root);
        auto check_file = [&](const std::optional<std::string>& rel, std::string_view field) {
            if (!rel.has_value()) {
                report.add_check(std::string(field), true, "not declared");
                return;
            }
            const std::filesystem::path full = dir / *rel;
            std::error_code ec;
            if (std::filesystem::is_regular_file(full, ec) && !ec) {
                report.add_check(std::string(field), true, "present: " + *rel);
            } else {
                report.add_check(
                    std::string(field),
                    false,
                    "missing file paths." + std::string(field) + "=" + *rel);
            }
        };
        check_file(artifact.paths().cpu_reference(), "paths.cpu_reference");
        check_file(artifact.paths().cuda_header(), "paths.cuda_header");
        check_file(artifact.paths().cuda_source(), "paths.cuda_source");
        check_file(artifact.paths().envelope_template(), "paths.envelope_template");
        check_file(artifact.paths().apply_ir(), "paths.apply_ir");
        check_file(artifact.paths().verify_report(), "paths.verify_report");

        if (artifact.paths().envelope_template().has_value()) {
            const std::filesystem::path env_path =
                dir / *artifact.paths().envelope_template();
            std::error_code ec;
            if (std::filesystem::is_regular_file(env_path, ec) && !ec) {
                StatusOr<TheoryEnvelopeBridge::Envelope> env =
                    TheoryEnvelopeBridge::load(env_path);
                if (!env.ok()) {
                    report.add_check(
                        "envelope_template.parse",
                        false,
                        env.status().message());
                } else {
                    Status against =
                        TheoryEnvelopeBridge::check_against_artifact(env.value(), artifact);
                    if (!against.ok()) {
                        report.add_check(
                            "envelope_template.content",
                            false,
                            against.message());
                    } else {
                        report.add_check(
                            "envelope_template.content",
                            true,
                            env.value().is_theory() ? "theory URI envelope"
                                                    : "catalog TransformEnvelope");
                    }
                }
            }
        }

        if (artifact.paths().apply_ir().has_value()) {
            const std::filesystem::path ir_path = dir / *artifact.paths().apply_ir();
            std::error_code ec;
            if (std::filesystem::is_regular_file(ir_path, ec) && !ec) {
                StatusOr<TheoryIr> ir = TheoryApplyIr::load(ir_path);
                if (!ir.ok()) {
                    report.add_check("apply_ir.parse", false, ir.status().message());
                } else if (ir.value().name() != artifact.name()) {
                    report.add_check(
                        "apply_ir.content",
                        false,
                        "apply_ir name does not match artifact name");
                } else {
                    report.add_check(
                        "apply_ir.content",
                        true,
                        "TheoryApplyIr ok for " + ir.value().name());
                }
            }
        }

        return report;
    }
};

#endif // THEORY_VALIDATE_HPP
