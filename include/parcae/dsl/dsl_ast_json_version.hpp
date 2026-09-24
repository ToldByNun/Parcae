#ifndef DSL_AST_JSON_VERSION_HPP
#define DSL_AST_JSON_VERSION_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"

#include <string>
#include <string_view>

/// Wire-protocol version for docs/spec/dsl-ast-json.md (`dsl_ast_json_version`).
/// Independent of DslSpecVersion (language semantics), but both travel in diagnostics.
class DslAstJsonVersion {
public:
    static constexpr int current_major = 1;
    static constexpr int current_minor = 1;
    static constexpr int current_patch = 0;
    static constexpr std::string_view current_string = "1.1.0";
    static constexpr std::string_view schema_id = "parcae.dsl_ast_json.v0";

    [[nodiscard]] static DslAstJsonVersion current() noexcept {
        return DslAstJsonVersion{current_major, current_minor, current_patch};
    }

    /// Same SemVer grammar as DslSpecVersion.
    [[nodiscard]] static StatusOr<DslAstJsonVersion> parse(std::string_view text) {
        StatusOr<DslSpecVersion> parsed = DslSpecVersion::parse(text);
        if (!parsed.ok()) {
            return Status::error(std::string("dsl_ast_json_version: ") + parsed.status().message());
        }
        return DslAstJsonVersion{parsed.value().major(), parsed.value().minor(),
                                 parsed.value().patch()};
    }

    [[nodiscard]] int major() const noexcept { return major_; }

    [[nodiscard]] int minor() const noexcept { return minor_; }

    [[nodiscard]] int patch() const noexcept { return patch_; }

    [[nodiscard]] std::string to_string() const {
        return std::to_string(major_) + "." + std::to_string(minor_) + "." + std::to_string(patch_);
    }

    [[nodiscard]] bool major_mismatch_with_current() const noexcept {
        return major_ != current_major;
    }

    [[nodiscard]] bool is_newer_than_current() const noexcept {
        return compare(*this, current()) > 0;
    }

    [[nodiscard]] Status check_compatible_with_current() const {
        if (major_mismatch_with_current()) {
            return Status::error("dsl_ast_json_version major mismatch: document " + to_string() +
                                 ", toolchain " + std::string(current_string));
        }
        if (is_newer_than_current()) {
            return Status::error("dsl_ast_json_version is newer than toolchain: document " +
                                 to_string() + ", toolchain " + std::string(current_string));
        }
        return Status::success();
    }

    [[nodiscard]] static int compare(const DslAstJsonVersion& a,
                                     const DslAstJsonVersion& b) noexcept {
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

    [[nodiscard]] friend bool operator==(const DslAstJsonVersion& a,
                                         const DslAstJsonVersion& b) noexcept {
        return compare(a, b) == 0;
    }

    [[nodiscard]] friend bool operator!=(const DslAstJsonVersion& a,
                                         const DslAstJsonVersion& b) noexcept {
        return !(a == b);
    }

private:
    DslAstJsonVersion(int major, int minor, int patch) noexcept
        : major_(major), minor_(minor), patch_(patch) {}

    int major_;
    int minor_;
    int patch_;
};

#endif // DSL_AST_JSON_VERSION_HPP
