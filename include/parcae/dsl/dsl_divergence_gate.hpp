#ifndef DSL_DIVERGENCE_GATE_HPP
#define DSL_DIVERGENCE_GATE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_directive_table.hpp"
#include "parcae/dsl/dsl_exec_scope.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/dsl_scope_analyzer.hpp"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Predicate class for HotLoop `If.test` (docs/spec/dsl.md § Execution scopes).
class DslPredicateClass {
public:
    enum class Kind : std::uint8_t {
        CompileTimeConstant = 0,
        LoopInvariant,
        HostFlag,
        ThreadVarying,
    };

    explicit DslPredicateClass(Kind kind, std::string evidence = {})
        : kind_(kind), evidence_(std::move(evidence)) {}

    [[nodiscard]] Kind kind() const noexcept {
        return kind_;
    }

    [[nodiscard]] const std::string& evidence() const noexcept {
        return evidence_;
    }

    [[nodiscard]] bool is_thread_varying() const noexcept {
        return kind_ == Kind::ThreadVarying;
    }

    [[nodiscard]] bool is_relaxed_ok() const noexcept {
        return kind_ == Kind::CompileTimeConstant || kind_ == Kind::LoopInvariant ||
               kind_ == Kind::HostFlag;
    }

    [[nodiscard]] std::string_view kind_string() const noexcept {
        switch (kind_) {
        case Kind::CompileTimeConstant:
            return "CompileTimeConstant";
        case Kind::LoopInvariant:
            return "LoopInvariant";
        case Kind::HostFlag:
            return "HostFlag";
        case Kind::ThreadVarying:
            return "ThreadVarying";
        }
        return "CompileTimeConstant";
    }

private:
    Kind kind_ = Kind::CompileTimeConstant;
    std::string evidence_;
};

/// Post-semantic HotLoop branch gate: ThreadVarying `If` → **E033**;
/// accepted const/Param/`HostFlag` predicates → **W011** (warnings only).
/// OuterControl `If` is ignored here (host glue / later HostGlue).
/// `divergent_branch` via `DslDirectiveTable` suppresses **E033** (+ **W010**).
class DslDivergenceGate {
public:
    class Report {
    public:
        explicit Report(std::vector<DslDiag> warnings = {})
            : warnings_(std::move(warnings)) {}

        [[nodiscard]] const std::vector<DslDiag>& warnings() const noexcept {
            return warnings_;
        }

        [[nodiscard]] bool empty() const noexcept {
            return warnings_.empty();
        }

    private:
        std::vector<DslDiag> warnings_;
    };

    /// Fail-loud on first E033; on success returns W011 warnings (may be empty).
    /// `default_cipher_var` is used when a HotLoop function has no positional args.
    [[nodiscard]] static StatusOr<Report> check(
        const DslAstDocument& doc,
        std::string_view default_cipher_var = "x") {
        return check(doc, default_cipher_var, nullptr);
    }

    [[nodiscard]] static StatusOr<Report> check(
        const DslAstDocument& doc,
        std::string_view default_cipher_var,
        DslDirectiveTable* directives) {
        if (!doc.module()) {
            return DslDiag::make(
                       DslRuleId::E031_forbidden_construct,
                       "document has no module AST",
                       doc.source_path())
                .to_status();
        }
        if (default_cipher_var.empty()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "default_cipher_var must be non-empty",
                       doc.source_path())
                .to_status();
        }

        StatusOr<DslScopeMap> scopes = DslScopeAnalyzer::analyze(doc);
        if (!scopes.ok()) {
            return scopes.status();
        }

        GateState state;
        state.source_path = doc.source_path();
        state.scopes = &scopes.value();
        state.cipher_var = std::string(default_cipher_var);
        state.directives = directives;
        Status st = walk_node(*doc.module(), state);
        if (!st.ok()) {
            return st;
        }
        return Report{std::move(state.warnings)};
    }

    /// Same as `check` but discards W011 (compile pipelines that only need E033).
    [[nodiscard]] static Status check_errors_only(
        const DslAstDocument& doc,
        std::string_view default_cipher_var = "x") {
        return check_errors_only(doc, default_cipher_var, nullptr);
    }

    [[nodiscard]] static Status check_errors_only(
        const DslAstDocument& doc,
        std::string_view default_cipher_var,
        DslDirectiveTable* directives) {
        StatusOr<Report> r = check(doc, default_cipher_var, directives);
        if (!r.ok()) {
            return r.status();
        }
        return Status::success();
    }

    /// Classify an expression AST for divergence policy (testing / BuildIr).
    [[nodiscard]] static DslPredicateClass classify_expr(
        const DslAstNode& expr,
        std::string_view cipher_var) {
        return classify_value_node(expr, cipher_var);
    }

private:
    struct GateState {
        std::string source_path;
        const DslScopeMap* scopes = nullptr;
        std::string cipher_var;
        std::vector<DslDiag> warnings;
        DslDirectiveTable* directives = nullptr;
    };

    DslDivergenceGate() = delete;

    [[nodiscard]] static DslExecScope scope_of(const DslAstNode& node, const GateState& state) {
        if (state.scopes != nullptr) {
            const std::optional<DslExecScope> found = state.scopes->get(&node);
            if (found.has_value()) {
                return found.value();
            }
        }
        return DslExecScope{DslExecScope::Kind::OuterControl};
    }

    [[nodiscard]] static std::optional<int> lineno_of(const DslAstNode& node) {
        return node.lineno();
    }

    [[nodiscard]] static std::optional<int> col_of(const DslAstNode& node) {
        return node.col_offset();
    }

    [[nodiscard]] static bool is_stream_index_name(std::string_view id) noexcept {
        return id == "i";
    }

    [[nodiscard]] static bool is_host_flag_name(std::string_view id) noexcept {
        return id == "flag" || id == "FLAG" || id == "enabled" || id == "use_alt" ||
               id == "host_flag";
    }

    [[nodiscard]] static DslPredicateClass join(DslPredicateClass a, DslPredicateClass b) {
        if (a.is_thread_varying()) {
            return a;
        }
        if (b.is_thread_varying()) {
            return b;
        }
        if (a.kind() == DslPredicateClass::Kind::HostFlag ||
            b.kind() == DslPredicateClass::Kind::HostFlag) {
            return DslPredicateClass{
                DslPredicateClass::Kind::HostFlag,
                !a.evidence().empty() ? a.evidence() : b.evidence()};
        }
        if (a.kind() == DslPredicateClass::Kind::LoopInvariant ||
            b.kind() == DslPredicateClass::Kind::LoopInvariant) {
            return DslPredicateClass{
                DslPredicateClass::Kind::LoopInvariant,
                !a.evidence().empty() ? a.evidence() : b.evidence()};
        }
        return a;
    }

    [[nodiscard]] static DslPredicateClass classify_name(
        std::string_view id,
        std::string_view cipher_var) {
        if (id == cipher_var || is_stream_index_name(id)) {
            return DslPredicateClass{
                DslPredicateClass::Kind::ThreadVarying, std::string(id)};
        }
        if (is_host_flag_name(id)) {
            return DslPredicateClass{DslPredicateClass::Kind::HostFlag, std::string(id)};
        }
        return DslPredicateClass{DslPredicateClass::Kind::LoopInvariant, std::string(id)};
    }

    [[nodiscard]] static DslPredicateClass classify_value(
        const DslAstValue& value,
        std::string_view cipher_var) {
        switch (value.type()) {
        case DslAstValue::Type::Null:
        case DslAstValue::Type::Bool:
        case DslAstValue::Type::Int:
        case DslAstValue::Type::Float:
        case DslAstValue::Type::String:
            return DslPredicateClass{DslPredicateClass::Kind::CompileTimeConstant};
        case DslAstValue::Type::Node:
            if (!value.as_node()) {
                return DslPredicateClass{DslPredicateClass::Kind::CompileTimeConstant};
            }
            return classify_value_node(*value.as_node(), cipher_var);
        case DslAstValue::Type::Array: {
            DslPredicateClass acc{DslPredicateClass::Kind::CompileTimeConstant};
            for (const DslAstValue& item : value.as_array()) {
                acc = join(acc, classify_value(item, cipher_var));
                if (acc.is_thread_varying()) {
                    return acc;
                }
            }
            return acc;
        }
        }
        return DslPredicateClass{DslPredicateClass::Kind::CompileTimeConstant};
    }

    [[nodiscard]] static DslPredicateClass classify_value_node(
        const DslAstNode& node,
        std::string_view cipher_var) {
        const std::string& kind = node.kind();

        if (kind == "Constant") {
            return DslPredicateClass{DslPredicateClass::Kind::CompileTimeConstant};
        }
        if (kind == "Name") {
            const DslAstValue* idv = node.find_field("id");
            if (idv && idv->type() == DslAstValue::Type::String) {
                return classify_name(idv->as_string(), cipher_var);
            }
            return DslPredicateClass{DslPredicateClass::Kind::LoopInvariant};
        }
        if (kind == "Attribute") {
            const DslAstValue* val = node.find_field("value");
            if (val) {
                return classify_value(*val, cipher_var);
            }
            return DslPredicateClass{DslPredicateClass::Kind::LoopInvariant};
        }
        if (kind == "Subscript") {
            DslPredicateClass acc{DslPredicateClass::Kind::CompileTimeConstant};
            for (const char* field : {"value", "slice"}) {
                const DslAstValue* v = node.find_field(field);
                if (v) {
                    acc = join(acc, classify_value(*v, cipher_var));
                }
            }
            return acc;
        }
        if (kind == "UnaryOp" || kind == "UAdd" || kind == "USub" || kind == "Not" ||
            kind == "Invert") {
            const DslAstValue* operand = node.find_field("operand");
            if (operand) {
                return classify_value(*operand, cipher_var);
            }
            // Nested operator kind nodes — walk fields.
        }
        if (kind == "BinOp" || kind == "BoolOp" || kind == "Compare") {
            DslPredicateClass acc{DslPredicateClass::Kind::CompileTimeConstant};
            for (const auto& entry : node.fields()) {
                if (entry.first == "op" || entry.first == "ops") {
                    continue;
                }
                acc = join(acc, classify_value(entry.second, cipher_var));
                if (acc.is_thread_varying()) {
                    return acc;
                }
            }
            return acc;
        }
        if (kind == "Call") {
            DslPredicateClass acc{DslPredicateClass::Kind::CompileTimeConstant};
            for (const auto& entry : node.fields()) {
                acc = join(acc, classify_value(entry.second, cipher_var));
                if (acc.is_thread_varying()) {
                    return acc;
                }
            }
            return acc;
        }
        if (kind == "Tuple" || kind == "List" || kind == "Dict") {
            DslPredicateClass acc{DslPredicateClass::Kind::CompileTimeConstant};
            for (const auto& entry : node.fields()) {
                acc = join(acc, classify_value(entry.second, cipher_var));
                if (acc.is_thread_varying()) {
                    return acc;
                }
            }
            return acc;
        }

        // Unknown expression shape: treat conservatively as loop-invariant host data
        // (not thread-varying) so we do not false-positive E033 on ops/ctx nodes.
        if (kind == "Load" || kind == "Store" || kind == "Eq" || kind == "NotEq" || kind == "Lt" ||
            kind == "LtE" || kind == "Gt" || kind == "GtE" || kind == "And" || kind == "Or" ||
            kind == "Add" || kind == "Sub" || kind == "Mult" || kind == "Div" || kind == "Mod" ||
            kind == "Pow" || kind == "BitAnd" || kind == "BitOr" || kind == "BitXor" ||
            kind == "LShift" || kind == "RShift" || kind == "FloorDiv") {
            return DslPredicateClass{DslPredicateClass::Kind::CompileTimeConstant};
        }

        DslPredicateClass acc{DslPredicateClass::Kind::CompileTimeConstant};
        for (const auto& entry : node.fields()) {
            acc = join(acc, classify_value(entry.second, cipher_var));
            if (acc.is_thread_varying()) {
                return acc;
            }
        }
        return acc;
    }

    /// First hot-loop cipher-like positional: skip leading `self` on methods.
    [[nodiscard]] static std::optional<std::string> first_positional_arg_name(
        const DslAstNode& fn) {
        const DslAstValue* args = fn.find_field("args");
        if (!args || args->type() != DslAstValue::Type::Node || !args->as_node()) {
            return std::nullopt;
        }
        const DslAstNode& arguments = *args->as_node();
        const DslAstValue* pos = arguments.find_field("args");
        if (!pos || pos->type() != DslAstValue::Type::Array || pos->as_array().empty()) {
            return std::nullopt;
        }
        for (const DslAstValue& item : pos->as_array()) {
            if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                continue;
            }
            const DslAstValue* name = item.as_node()->find_field("arg");
            if (!name || name->type() != DslAstValue::Type::String) {
                continue;
            }
            const std::string& id = name->as_string();
            if (id == "self" || id.empty()) {
                continue;
            }
            return id;
        }
        return std::nullopt;
    }

    [[nodiscard]] static Status check_hotloop_if(
        const DslAstNode& if_node,
        GateState& state) {
        const DslAstValue* test = if_node.find_field("test");
        if (!test || test->type() != DslAstValue::Type::Node || !test->as_node()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "If missing test expression",
                       state.source_path,
                       lineno_of(if_node),
                       col_of(if_node))
                .to_status();
        }

        const DslPredicateClass pred = classify_value(*test, state.cipher_var);
        const DslExecScope scope = scope_of(if_node, state);

        if (pred.is_thread_varying()) {
            if (state.directives != nullptr &&
                state.directives->honor(
                    DslDirectiveTable::flag_divergent_branch,
                    if_node,
                    DslRuleId::E033_divergent_branch)) {
                return Status::success();
            }
            std::string msg =
                "HotLoop if depends on rune-varying data; use const/Param flag or Select";
            if (!pred.evidence().empty()) {
                msg += " (name '";
                msg += pred.evidence();
                msg += "')";
            }
            if (scope.loop_depth() > 0) {
                msg += " (loop_depth=";
                msg += std::to_string(scope.loop_depth());
                msg += ")";
            }
            return DslDiag::make(
                       DslRuleId::E033_divergent_branch,
                       std::move(msg),
                       state.source_path,
                       lineno_of(if_node),
                       col_of(if_node),
                       "Prefer Param/host flags, compile-time constants, or Z29Expr Select")
                .to_status();
        }

        std::string wmsg = "HotLoop if accepted as ";
        wmsg += pred.kind_string();
        if (!pred.evidence().empty()) {
            wmsg += " (";
            wmsg += pred.evidence();
            wmsg += ")";
        }
        state.warnings.push_back(DslDiag::make(
            DslRuleId::W011_relaxed_branch,
            std::move(wmsg),
            state.source_path,
            lineno_of(if_node),
            col_of(if_node)));
        return Status::success();
    }

    [[nodiscard]] static Status check_hotloop_if_exp(
        const DslAstNode& ifexp,
        GateState& state) {
        const DslAstValue* test = ifexp.find_field("test");
        if (!test || test->type() != DslAstValue::Type::Node || !test->as_node()) {
            return DslDiag::make(
                       DslRuleId::E032_primitive_body,
                       "IfExp missing test expression",
                       state.source_path,
                       lineno_of(ifexp),
                       col_of(ifexp))
                .to_status();
        }

        const DslPredicateClass pred = classify_value(*test, state.cipher_var);
        const DslExecScope scope = scope_of(ifexp, state);

        if (pred.is_thread_varying()) {
            const bool honored =
                state.directives != nullptr &&
                (state.directives->honor(
                     DslDirectiveTable::flag_divergent_branch,
                     ifexp,
                     DslRuleId::E033_divergent_branch) ||
                 (lineno_of(ifexp).has_value() &&
                  state.directives->honor_at_line(
                      DslDirectiveTable::flag_divergent_branch,
                      *lineno_of(ifexp),
                      DslRuleId::E033_divergent_branch)));
            if (honored) {
                return Status::success();
            }
            std::string msg =
                "HotLoop if-expression depends on rune-varying data; use const/Param flag or Select";
            if (!pred.evidence().empty()) {
                msg += " (name '";
                msg += pred.evidence();
                msg += "')";
            }
            if (scope.loop_depth() > 0) {
                msg += " (loop_depth=";
                msg += std::to_string(scope.loop_depth());
                msg += ")";
            }
            return DslDiag::make(
                       DslRuleId::E033_divergent_branch,
                       std::move(msg),
                       state.source_path,
                       lineno_of(ifexp),
                       col_of(ifexp),
                       "Prefer Param/host flags, compile-time constants, or Z29Expr Select")
                .to_status();
        }

        std::string wmsg = "HotLoop if-expression accepted as ";
        wmsg += pred.kind_string();
        if (!pred.evidence().empty()) {
            wmsg += " (";
            wmsg += pred.evidence();
            wmsg += ")";
        }
        state.warnings.push_back(DslDiag::make(
            DslRuleId::W011_relaxed_branch,
            std::move(wmsg),
            state.source_path,
            lineno_of(ifexp),
            col_of(ifexp)));
        return Status::success();
    }

    [[nodiscard]] static Status walk_value(const DslAstValue& value, GateState& state) {
        if (value.type() == DslAstValue::Type::Node && value.as_node()) {
            return walk_node(*value.as_node(), state);
        }
        if (value.type() == DslAstValue::Type::Array) {
            for (const DslAstValue& item : value.as_array()) {
                Status st = walk_value(item, state);
                if (!st.ok()) {
                    return st;
                }
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status walk_node(const DslAstNode& node, GateState& state) {
        const std::string& kind = node.kind();

        if (kind == "FunctionDef" || kind == "AsyncFunctionDef") {
            GateState child = state;
            const std::optional<std::string> first = first_positional_arg_name(node);
            const DslAstValue* body = node.find_field("body");
            bool body_hot = false;
            if (body && body->type() == DslAstValue::Type::Array && !body->as_array().empty()) {
                const DslAstValue& first_stmt = body->as_array().front();
                if (first_stmt.type() == DslAstValue::Type::Node && first_stmt.as_node()) {
                    body_hot = scope_of(*first_stmt.as_node(), state).is_hot_loop();
                }
            }
            if (body_hot && first.has_value() && !first->empty()) {
                child.cipher_var = *first;
            }

            for (const auto& entry : node.fields()) {
                Status st = walk_value(entry.second, child);
                if (!st.ok()) {
                    return st;
                }
            }
            state.warnings.insert(
                state.warnings.end(),
                std::make_move_iterator(child.warnings.begin()),
                std::make_move_iterator(child.warnings.end()));
            return Status::success();
        }

        if (kind == "If") {
            const DslExecScope scope = scope_of(node, state);
            if (scope.is_hot_loop()) {
                Status st = check_hotloop_if(node, state);
                if (!st.ok()) {
                    return st;
                }
            }
            // Always walk children (elif chain, nested If).
        }

        if (kind == "IfExp") {
            const DslExecScope scope = scope_of(node, state);
            if (scope.is_hot_loop()) {
                Status st = check_hotloop_if_exp(node, state);
                if (!st.ok()) {
                    return st;
                }
            }
        }

        for (const auto& entry : node.fields()) {
            Status st = walk_value(entry.second, state);
            if (!st.ok()) {
                return st;
            }
        }
        return Status::success();
    }
};

#endif // DSL_DIVERGENCE_GATE_HPP
