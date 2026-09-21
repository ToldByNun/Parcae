#ifndef THEORY_REGISTRY_HPP
#define THEORY_REGISTRY_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_uri.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Load / catalog compiled theory artifacts under a theories root.
/// `load` hard-rejects incompatible `dsl_spec_version` (docs/spec/dsl.md).
/// Catalog listings expose `stale_spec` for MAJOR mismatches without pretending
/// the artifact is ready-to-run.
class TheoryRegistry {
public:
    class CatalogEntry {
    public:
        CatalogEntry(
            TheoryUri uri,
            std::string dsl_spec_version,
            bool stale_spec,
            bool ready,
            std::string detail = {})
            : uri_(std::move(uri)),
              dsl_spec_version_(std::move(dsl_spec_version)),
              stale_spec_(stale_spec),
              ready_(ready),
              detail_(std::move(detail)) {}

        [[nodiscard]] const TheoryUri& uri() const noexcept {
            return uri_;
        }

        [[nodiscard]] const std::string& dsl_spec_version() const noexcept {
            return dsl_spec_version_;
        }

        /// True when MAJOR differs from running `DslSpecVersion::current`.
        [[nodiscard]] bool stale_spec() const noexcept {
            return stale_spec_;
        }

        /// Ready-to-run: dsl_spec_version compatible with current toolchain.
        [[nodiscard]] bool ready() const noexcept {
            return ready_;
        }

        [[nodiscard]] const std::string& detail() const noexcept {
            return detail_;
        }

        [[nodiscard]] nlohmann::json to_json() const {
            nlohmann::json out{
                {"uri", uri_.to_string()},
                {"dsl_spec_version", dsl_spec_version_},
                {"stale_spec", stale_spec_},
                {"ready", ready_},
            };
            if (detail_.empty()) {
                out["detail"] = nullptr;
            } else {
                out["detail"] = detail_;
            }
            return out;
        }

    private:
        TheoryUri uri_;
        std::string dsl_spec_version_;
        bool stale_spec_ = false;
        bool ready_ = false;
        std::string detail_;
    };

    /// Spec compatibility for a loaded/parsed artifact (MAJOR match + not newer).
    [[nodiscard]] static Status check_dsl_spec(const TheoryArtifact& artifact) {
        return check_dsl_spec_string(artifact.dsl_spec_version());
    }

    [[nodiscard]] static Status check_dsl_spec_string(std::string_view dsl_spec_version) {
        StatusOr<DslSpecVersion> parsed = DslSpecVersion::parse(dsl_spec_version);
        if (!parsed.ok()) {
            return Status::error(
                "DSL spec " + std::string(DslSpecVersion::current_string) +
                " required; artifact dsl_spec_version '" + std::string(dsl_spec_version) +
                "' is not valid SemVer — re-run parcae-compile");
        }
        const Status compat = parsed.value().check_compatible_with_current();
        if (!compat.ok()) {
            return Status::error(
                "DSL spec " + std::string(DslSpecVersion::current_string) +
                " required; artifact built for " + std::string(dsl_spec_version) +
                " — re-run parcae-compile");
        }
        return Status::success();
    }

    /// Catalog helper: MAJOR mismatch ⇒ `stale_spec: true`.
    [[nodiscard]] static bool is_stale_spec(std::string_view dsl_spec_version) {
        StatusOr<DslSpecVersion> parsed = DslSpecVersion::parse(dsl_spec_version);
        if (!parsed.ok()) {
            return true;
        }
        return parsed.value().stale_spec_relative_to_current();
    }

    [[nodiscard]] static bool is_stale_spec(const TheoryArtifact& artifact) {
        return is_stale_spec(artifact.dsl_spec_version());
    }

    /// Load manifest and enforce dsl_spec_version compatibility. Stale / newer → error.
    [[nodiscard]] static StatusOr<TheoryArtifact> load(
        const std::filesystem::path& theories_root,
        std::string_view name,
        std::uint32_t version) {
        StatusOr<TheoryArtifact> artifact = TheoryArtifact::load(theories_root, name, version);
        if (!artifact.ok()) {
            return artifact.status();
        }
        Status spec = check_dsl_spec(artifact.value());
        if (!spec.ok()) {
            return spec;
        }
        return artifact;
    }

    [[nodiscard]] static StatusOr<TheoryArtifact> load_uri(
        const std::filesystem::path& theories_root,
        std::string_view uri_text) {
        StatusOr<TheoryUri> uri = TheoryUri::parse(uri_text);
        if (!uri.ok()) {
            return uri.status();
        }
        return load(theories_root, uri.value().name(), uri.value().version());
    }

    [[nodiscard]] static StatusOr<TheoryArtifact> load_uri(
        const std::filesystem::path& theories_root,
        const TheoryUri& uri) {
        return load(theories_root, uri.name(), uri.version());
    }

    /// Scan `theories_root/<name>/<version>/manifest.json`. Stale-MAJOR entries
    /// are included with `stale_spec == true` (not ready-to-run).
    [[nodiscard]] static StatusOr<std::vector<CatalogEntry>> list(
        const std::filesystem::path& theories_root) {
        std::vector<CatalogEntry> out;
        std::error_code ec;
        if (!std::filesystem::exists(theories_root, ec) || ec) {
            return out;
        }
        if (!std::filesystem::is_directory(theories_root, ec) || ec) {
            return Status::error(
                "TheoryRegistry::list: theories_root is not a directory: " +
                theories_root.string());
        }

        for (const auto& name_entry :
             std::filesystem::directory_iterator(theories_root, ec)) {
            if (ec) {
                return Status::error(
                    "TheoryRegistry::list: failed to iterate theories_root: " + ec.message());
            }
            if (!name_entry.is_directory()) {
                continue;
            }
            const std::string name = name_entry.path().filename().string();
            StatusOr<std::string> name_ok = TheoryUri::validate_name(name);
            if (!name_ok.ok()) {
                continue;
            }

            std::error_code ver_ec;
            for (const auto& ver_entry :
                 std::filesystem::directory_iterator(name_entry.path(), ver_ec)) {
                if (ver_ec) {
                    return Status::error(
                        "TheoryRegistry::list: failed to iterate theory versions: " +
                        ver_ec.message());
                }
                if (!ver_entry.is_directory()) {
                    continue;
                }
                const std::string ver_text = ver_entry.path().filename().string();
                StatusOr<std::uint32_t> version = parse_version_dir(ver_text);
                if (!version.ok()) {
                    continue;
                }
                const std::filesystem::path manifest =
                    ver_entry.path() / "manifest.json";
                if (!std::filesystem::is_regular_file(manifest, ver_ec) || ver_ec) {
                    continue;
                }

                StatusOr<TheoryArtifact> artifact =
                    TheoryArtifact::load(theories_root, name, version.value());
                if (!artifact.ok()) {
                    // Unreadable / invalid manifest: skip from catalog (tools may
                    // surface via validate). Do not pretend ready.
                    continue;
                }

                const std::string& spec = artifact.value().dsl_spec_version();
                const bool stale = is_stale_spec(spec);
                const Status compat = check_dsl_spec_string(spec);
                const bool ready = compat.ok();
                std::string detail;
                if (!ready) {
                    detail = compat.message();
                }
                out.emplace_back(
                    artifact.value().uri(),
                    spec,
                    stale,
                    ready,
                    std::move(detail));
            }
        }
        return out;
    }

private:
    TheoryRegistry() = delete;

    [[nodiscard]] static StatusOr<std::uint32_t> parse_version_dir(std::string_view text) {
        if (text.empty() || (text.size() > 1 && text[0] == '0') ||
            text.find_first_not_of("0123456789") != std::string_view::npos) {
            return Status::error("invalid version directory");
        }
        unsigned long value = 0;
        for (char ch : text) {
            value = value * 10u + static_cast<unsigned long>(ch - '0');
            if (value > 1'000'000'000ul) {
                return Status::error("version directory out of range");
            }
        }
        if (value == 0) {
            return Status::error("version directory must be positive");
        }
        return static_cast<std::uint32_t>(value);
    }
};

#endif // THEORY_REGISTRY_HPP
