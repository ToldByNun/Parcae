#ifndef DSL_DIAG_HPP
#define DSL_DIAG_HPP

#include "parcae/core/status.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

/// User-facing DSL compile diagnostic with optional source location.
/// Format (when location known): `path:line:col: RULE message`
/// Format (no location): `RULE message` or `path: RULE message`
/// Normative: docs/spec/dsl.md § Diagnostics.
class DslDiag {
public:
    [[nodiscard]] static DslDiag make(
        std::string_view rule_id,
        std::string message,
        std::string path = {},
        std::optional<int> lineno = std::nullopt,
        std::optional<int> col = std::nullopt,
        std::string hint = {}) {
        return DslDiag{
            std::string(rule_id),
            std::move(message),
            std::move(path),
            lineno,
            col,
            std::move(hint),
        };
    }

    [[nodiscard]] const std::string& rule_id() const noexcept {
        return rule_id_;
    }

    [[nodiscard]] const std::string& message() const noexcept {
        return message_;
    }

    [[nodiscard]] const std::string& path() const noexcept {
        return path_;
    }

    [[nodiscard]] std::optional<int> lineno() const noexcept {
        return lineno_;
    }

    [[nodiscard]] std::optional<int> col() const noexcept {
        return col_;
    }

    [[nodiscard]] const std::string& hint() const noexcept {
        return hint_;
    }

    [[nodiscard]] bool has_location() const noexcept {
        return lineno_.has_value();
    }

    /// Primary one-line diagnostic (never a C++/Python stack trace).
    [[nodiscard]] std::string format() const {
        std::string out;
        if (!path_.empty()) {
            out += path_;
            if (lineno_.has_value()) {
                out.push_back(':');
                out += std::to_string(*lineno_);
                // col is printed when lineno is present (0-based or 0 if unknown).
                out.push_back(':');
                out += std::to_string(col_.value_or(0));
            }
            out += ": ";
        } else if (lineno_.has_value()) {
            out += std::to_string(*lineno_);
            out.push_back(':');
            out += std::to_string(col_.value_or(0));
            out += ": ";
        }
        out += rule_id_;
        out.push_back(' ');
        out += message_;
        return out;
    }

    /// format() plus optional hint on a second line (`note: …`).
    [[nodiscard]] std::string format_with_hint() const {
        std::string out = format();
        if (!hint_.empty()) {
            out += "\nnote: ";
            out += hint_;
        }
        return out;
    }

    [[nodiscard]] Status to_status() const {
        return Status::error(format_with_hint());
    }

private:
    DslDiag(
        std::string rule_id,
        std::string message,
        std::string path,
        std::optional<int> lineno,
        std::optional<int> col,
        std::string hint)
        : rule_id_(std::move(rule_id)),
          message_(std::move(message)),
          path_(std::move(path)),
          lineno_(lineno),
          col_(col),
          hint_(std::move(hint)) {}

    std::string rule_id_;
    std::string message_;
    std::string path_;
    std::optional<int> lineno_;
    std::optional<int> col_;
    std::string hint_;
};

#endif // DSL_DIAG_HPP
