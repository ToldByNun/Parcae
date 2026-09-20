#ifndef DSL_SPEC_VERSION_HPP
#define DSL_SPEC_VERSION_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cctype>
#include <string>
#include <string_view>

/// Normative language / verify-gate version from docs/spec/dsl.md (`dsl_spec_version`).
/// Artifacts embed this string; MAJOR mismatch → reject / recompile (see dsl.md policy).
class DslSpecVersion {
public:
    /// Compiler / toolkit current language spec (must match docs/spec/dsl.md).
    static constexpr int current_major = 1;
    static constexpr int current_minor = 0;
    static constexpr int current_patch = 0;
    static constexpr std::string_view current_string = "1.0.0";

    [[nodiscard]] static DslSpecVersion current() noexcept {
        return DslSpecVersion{current_major, current_minor, current_patch};
    }

    [[nodiscard]] static StatusOr<DslSpecVersion> parse(std::string_view text) {
        if (text.empty()) {
            return Status::error("dsl_spec_version must be non-empty SemVer MAJOR.MINOR.PATCH");
        }

        int major = 0;
        int minor = 0;
        int patch = 0;
        std::size_t i = 0;
        Status part = parse_nonneg_int(text, i, major);
        if (!part.ok()) {
            return part;
        }
        if (i >= text.size() || text[i] != '.') {
            return Status::error("dsl_spec_version must be MAJOR.MINOR.PATCH");
        }
        ++i;
        part = parse_nonneg_int(text, i, minor);
        if (!part.ok()) {
            return part;
        }
        if (i >= text.size() || text[i] != '.') {
            return Status::error("dsl_spec_version must be MAJOR.MINOR.PATCH");
        }
        ++i;
        part = parse_nonneg_int(text, i, patch);
        if (!part.ok()) {
            return part;
        }
        if (i != text.size()) {
            return Status::error("dsl_spec_version has trailing garbage");
        }
        return DslSpecVersion{major, minor, patch};
    }

    [[nodiscard]] int major() const noexcept {
        return major_;
    }

    [[nodiscard]] int minor() const noexcept {
        return minor_;
    }

    [[nodiscard]] int patch() const noexcept {
        return patch_;
    }

    [[nodiscard]] std::string to_string() const {
        return std::to_string(major_) + "." + std::to_string(minor_) + "." +
               std::to_string(patch_);
    }

    /// True when this version equals current_major/minor/patch.
    [[nodiscard]] bool equals_current() const noexcept {
        return major_ == current_major && minor_ == current_minor && patch_ == current_patch;
    }

    /// MAJOR differs from the running compiler language spec.
    [[nodiscard]] bool major_mismatch_with_current() const noexcept {
        return major_ != current_major;
    }

    /// Artifact is newer than the running compiler (same or any major; SemVer order).
    [[nodiscard]] bool is_newer_than_current() const noexcept {
        return compare(*this, current()) > 0;
    }

    /// Load/validate/sweep may accept this artifact version for the running compiler.
    /// Rejects MAJOR mismatch and forward-incompatible (newer) versions.
    [[nodiscard]] Status check_compatible_with_current() const {
        if (major_mismatch_with_current()) {
            return Status::error(
                "dsl_spec_version major mismatch: artifact " + to_string() + ", toolchain " +
                std::string(current_string) + " — re-run parcae-compile");
        }
        if (is_newer_than_current()) {
            return Status::error(
                "dsl_spec_version is newer than toolchain: artifact " + to_string() +
                ", toolchain " + std::string(current_string) +
                " — upgrade Parcae or rebuild the artifact with this toolkit");
        }
        return Status::success();
    }

    /// Catalog helper: MAJOR mismatch ⇒ stale_spec listing.
    [[nodiscard]] bool stale_spec_relative_to_current() const noexcept {
        return major_mismatch_with_current();
    }

    [[nodiscard]] static int compare(const DslSpecVersion& a, const DslSpecVersion& b) noexcept {
        if (a.major_ != b.major_) {
            return a.major_ < b.major_ ? -1 : 1;
        }
        if (a.minor_ != b.minor_) {
            return a.minor_ < b.minor_ ? -1 : 1;
        }
        if (a.patch_ != b.patch_) {
            return a.patch_ < b.patch_ ? -1 : 1;
        }
        return 0;
    }

    [[nodiscard]] friend bool operator==(const DslSpecVersion& a, const DslSpecVersion& b) noexcept {
        return compare(a, b) == 0;
    }

    [[nodiscard]] friend bool operator!=(const DslSpecVersion& a, const DslSpecVersion& b) noexcept {
        return !(a == b);
    }

private:
    DslSpecVersion(int major, int minor, int patch) noexcept
        : major_(major), minor_(minor), patch_(patch) {}

    [[nodiscard]] static Status parse_nonneg_int(std::string_view text, std::size_t& i, int& out) {
        if (i >= text.size() || !std::isdigit(static_cast<unsigned char>(text[i]))) {
            return Status::error("dsl_spec_version expected a decimal integer component");
        }
        // No leading zeros except a single "0".
        if (text[i] == '0' && i + 1 < text.size() && std::isdigit(static_cast<unsigned char>(text[i + 1]))) {
            return Status::error("dsl_spec_version components must not have leading zeros");
        }
        long value = 0;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            value = value * 10 + (text[i] - '0');
            if (value > 1'000'000) {
                return Status::error("dsl_spec_version component out of range");
            }
            ++i;
        }
        out = static_cast<int>(value);
        return Status::success();
    }

    int major_;
    int minor_;
    int patch_;
};

#endif // DSL_SPEC_VERSION_HPP
