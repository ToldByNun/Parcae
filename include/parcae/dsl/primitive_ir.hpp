#ifndef PRIMITIVE_IR_HPP
#define PRIMITIVE_IR_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cctype>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Compiled `@define_primitive` record: name, signature arity, Z29Expr body.
/// AST→IR lowering is `DslBuildIr` (later); this type holds the IR + checks.
class PrimitiveIr {
public:
    [[nodiscard]] static StatusOr<PrimitiveIr> make(std::string name, std::string signature,
                                                    Z29Expr::Ptr body, std::string source_path = {},
                                                    std::optional<int> lineno = std::nullopt,
                                                    std::optional<int> col = std::nullopt) {
        if (name.empty()) {
            return fail(DslRuleId::E032_primitive_body, "primitive name must be non-empty",
                        std::move(source_path), lineno, col);
        }
        if (!body) {
            return fail(DslRuleId::E032_primitive_body, "primitive '" + name + "' body is null",
                        std::move(source_path), lineno, col);
        }
        StatusOr<std::vector<std::string>> params = parse_signature_params(signature);
        if (!params.ok()) {
            return DslDiag::make(DslRuleId::E032_primitive_body, params.status().message(),
                                 source_path, lineno, col)
                .to_status();
        }
        return PrimitiveIr{
            std::move(name),
            std::move(signature),
            std::move(params.value()),
            std::move(body),
            std::move(source_path),
            lineno,
            col,
        };
    }

    /// Parse `(i: Z29, c2: Z29) -> Z29` → `{"i","c2"}`.
    [[nodiscard]] static StatusOr<std::vector<std::string>>
    parse_signature_params(std::string_view signature) {
        std::string_view s = trim(signature);
        if (s.empty() || s.front() != '(') {
            return Status::error("signature must start with '('");
        }
        const std::size_t close = s.find(')');
        if (close == std::string_view::npos) {
            return Status::error("signature missing ')'");
        }
        std::string_view after = trim(s.substr(close + 1));
        if (!starts_with(after, "->")) {
            return Status::error("signature must end with '-> Z29'");
        }
        after = trim(after.substr(2));
        if (after != "Z29") {
            return Status::error("signature return type must be Z29");
        }

        std::string_view inner = trim(s.substr(1, close - 1));
        std::vector<std::string> names;
        if (inner.empty()) {
            return names;
        }
        std::size_t start = 0;
        while (start <= inner.size()) {
            std::size_t comma = inner.find(',', start);
            const std::string_view part =
                trim(comma == std::string_view::npos ? inner.substr(start)
                                                     : inner.substr(start, comma - start));
            if (part.empty()) {
                return Status::error("empty parameter slot in signature");
            }
            const std::size_t colon = part.find(':');
            if (colon == std::string_view::npos) {
                return Status::error("signature parameter missing ': Z29'");
            }
            std::string_view pname = trim(part.substr(0, colon));
            std::string_view ptype = trim(part.substr(colon + 1));
            if (pname.empty() || !is_ident(pname)) {
                return Status::error("invalid parameter name in signature");
            }
            if (ptype != "Z29") {
                return Status::error("signature parameter '" + std::string(pname) +
                                     "' type must be Z29");
            }
            names.emplace_back(pname);
            if (comma == std::string_view::npos) {
                break;
            }
            start = comma + 1;
        }
        return names;
    }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    [[nodiscard]] const std::string& signature() const noexcept { return signature_; }

    [[nodiscard]] const std::vector<std::string>& param_names() const noexcept {
        return param_names_;
    }

    [[nodiscard]] std::size_t arity() const noexcept { return param_names_.size(); }

    [[nodiscard]] const Z29Expr::Ptr& body() const noexcept { return body_; }

    [[nodiscard]] Status validate() const {
        if (name_.empty()) {
            return fail_here(DslRuleId::E032_primitive_body, "primitive name must be non-empty");
        }
        if (!body_) {
            return fail_here(DslRuleId::E032_primitive_body, "primitive body is null");
        }
        StatusOr<std::vector<std::string>> parsed = parse_signature_params(signature_);
        if (!parsed.ok()) {
            return fail_here(DslRuleId::E032_primitive_body, parsed.status().message());
        }
        if (parsed.value() != param_names_) {
            return fail_here(DslRuleId::E032_primitive_body,
                             "primitive '" + name_ + "' param_names out of sync with signature");
        }
        return Status::success();
    }

    /// Evaluate body with positional args bound to `param_names` order.
    [[nodiscard]] StatusOr<Index29> eval(std::span<const Index29> args) const {
        Status st = validate();
        if (!st.ok()) {
            return st;
        }
        if (args.size() != param_names_.size()) {
            return fail_here(DslRuleId::E032_primitive_body,
                             "primitive '" + name_ + "' expects " +
                                 std::to_string(param_names_.size()) + " args, got " +
                                 std::to_string(args.size()));
        }
        Z29Expr::Env env;
        for (std::size_t i = 0; i < param_names_.size(); ++i) {
            env.emplace(param_names_[i], args[i]);
        }
        return body_->eval(env);
    }

private:
    PrimitiveIr(std::string name, std::string signature, std::vector<std::string> param_names,
                Z29Expr::Ptr body, std::string source_path, std::optional<int> lineno,
                std::optional<int> col)
        : name_(std::move(name)), signature_(std::move(signature)),
          param_names_(std::move(param_names)), body_(std::move(body)),
          source_path_(std::move(source_path)), lineno_(lineno), col_(col) {}

    [[nodiscard]] Status fail_here(std::string_view rule_id, std::string message) const {
        return DslDiag::make(rule_id, std::move(message), source_path_, lineno_, col_).to_status();
    }

    [[nodiscard]] static Status fail(std::string_view rule_id, std::string message,
                                     std::string path, std::optional<int> lineno,
                                     std::optional<int> col) {
        return DslDiag::make(rule_id, std::move(message), std::move(path), lineno, col).to_status();
    }

    [[nodiscard]] static std::string_view trim(std::string_view s) {
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
            s.remove_prefix(1);
        }
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
            s.remove_suffix(1);
        }
        return s;
    }

    [[nodiscard]] static bool starts_with(std::string_view s, std::string_view prefix) {
        return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
    }

    [[nodiscard]] static bool is_ident(std::string_view s) {
        if (s.empty()) {
            return false;
        }
        const unsigned char c0 = static_cast<unsigned char>(s[0]);
        if (!(std::isalpha(c0) || s[0] == '_')) {
            return false;
        }
        for (char ch : s) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if (!(std::isalnum(c) || ch == '_')) {
                return false;
            }
        }
        return true;
    }

    std::string name_;
    std::string signature_;
    std::vector<std::string> param_names_;
    Z29Expr::Ptr body_;
    std::string source_path_;
    std::optional<int> lineno_;
    std::optional<int> col_;
};

#endif // PRIMITIVE_IR_HPP
