#ifndef DSL_SEMANTIC_GATE_HPP
#define DSL_SEMANTIC_GATE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_directive_table.hpp"
#include "parcae/dsl/dsl_exec_scope.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/dsl_scope_analyzer.hpp"
#include "parcae/dsl/dsl_z29_builtins.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

/// Post-ingest DSL whitelist gate (docs/spec/dsl.md, docs/spec/dsl-ast-json.md).
///
/// Rejects forbidden node kinds, non-whitelist imports, nested ClassDef in
/// functions, starred args / **kwargs, illegal op/ctx strings, and unknown
/// `z29_*` Call names (see `DslZ29Builtins`; includes `z29_matmul` /
/// `z29_det` / `z29_autokey_shift`).
/// Scope-aware control flow (`If` / `For` / `While` / `Break` / `Continue`):
/// OuterControl allowed; HotLoop loops/`break`/`continue` → **E034**
/// (suppressible via `DslDirectiveTable` + `hotloop_restriction`).
/// HotLoop divergent `if` is **E033** (handled by `DslDivergenceGate`, not here —
/// HotLoop `If` is accepted at this gate).
/// Does not build IR (that is DslBuildIr) or check verify/tier claims.
class DslSemanticGate {
public:
    [[nodiscard]] static Status check(const DslAstDocument& doc) { return check(doc, nullptr); }

    [[nodiscard]] static Status check(const DslAstDocument& doc, DslDirectiveTable* directives) {
        if (!doc.module()) {
            return fail(DslRuleId::E031_forbidden_construct, "document has no module AST",
                        doc.source_path());
        }

        StatusOr<DslScopeMap> scopes = DslScopeAnalyzer::analyze(doc);
        if (!scopes.ok()) {
            return scopes.status();
        }

        GateState state;
        state.source_path = doc.source_path();
        state.scopes = &scopes.value();
        state.directives = directives;
        return walk_node(*doc.module(), state, /*function_depth=*/0);
    }

private:
    struct Loc {
        std::optional<int> lineno;
        std::optional<int> col;
    };

    struct GateState {
        std::string source_path;
        Loc nearest; // nearest ancestor location for nodes lacking lineno
        const DslScopeMap* scopes = nullptr;
        DslDirectiveTable* directives = nullptr;
    };

    DslSemanticGate() = delete;

    [[nodiscard]] static Status fail(std::string_view rule_id, std::string message,
                                     const std::string& path, Loc loc = {}) {
        return DslDiag::make(rule_id, std::move(message), path, loc.lineno, loc.col).to_status();
    }

    [[nodiscard]] static Loc location_of(const DslAstNode& node, const GateState& state) {
        if (node.lineno().has_value()) {
            return Loc{node.lineno(), node.col_offset()};
        }
        return state.nearest;
    }

    [[nodiscard]] static DslExecScope scope_of(const DslAstNode& node, const GateState& state) {
        if (state.scopes != nullptr) {
            const std::optional<DslExecScope> found = state.scopes->get(&node);
            if (found.has_value()) {
                return found.value();
            }
        }
        return DslExecScope{DslExecScope::Kind::OuterControl};
    }

    [[nodiscard]] static bool is_allowed_import_module(std::string_view module) {
        return module == "parcae.dsl.math" || module == "parcae.dsl.theory" ||
               module == "parcae.dsl.primitives" || module == "parcae.dsl.testing";
    }

    [[nodiscard]] static bool is_binop_kind(std::string_view kind) {
        return kind == "Add" || kind == "Sub" || kind == "Mult" || kind == "Div" ||
               kind == "FloorDiv" || kind == "Mod" || kind == "Pow" || kind == "LShift" ||
               kind == "RShift" || kind == "BitOr" || kind == "BitXor" || kind == "BitAnd";
    }

    [[nodiscard]] static bool is_unaryop_kind(std::string_view kind) {
        return kind == "UAdd" || kind == "USub" || kind == "Not" || kind == "Invert";
    }

    [[nodiscard]] static bool is_boolop_kind(std::string_view kind) {
        return kind == "And" || kind == "Or";
    }

    [[nodiscard]] static bool is_cmpop_kind(std::string_view kind) {
        // Pure Index29 comparisons only (no Is/In — those are identity/container).
        return kind == "Eq" || kind == "NotEq" || kind == "Lt" || kind == "LtE" || kind == "Gt" ||
               kind == "GtE";
    }

    [[nodiscard]] static bool is_ctx_kind(std::string_view kind) {
        return kind == "Load" || kind == "Store";
    }

    [[nodiscard]] static bool is_operator_or_ctx_kind(std::string_view kind) {
        return is_binop_kind(kind) || is_unaryop_kind(kind) || is_boolop_kind(kind) ||
               is_cmpop_kind(kind) || is_ctx_kind(kind);
    }

    /// Scope-conditioned control-flow kinds (not globally forbidden).
    [[nodiscard]] static bool is_scope_control_kind(std::string_view kind) noexcept {
        return kind == "If" || kind == "For" || kind == "While" || kind == "Break" ||
               kind == "Continue";
    }

    /// Always-illegal async loop form (never OuterControl-legal).
    [[nodiscard]] static bool is_always_forbidden_async_loop(std::string_view kind) noexcept {
        return kind == "AsyncFor";
    }

    /// DSL-facing node whitelist from dsl-ast-json.md § Allowed kind values,
    /// plus scope-aware control-flow kinds validated separately.
    [[nodiscard]] static bool is_allowed_kind(std::string_view kind) {
        // Structural
        if (kind == "Module" || kind == "ClassDef" || kind == "FunctionDef" ||
            kind == "arguments" || kind == "arg" || kind == "Return" || kind == "Expr" ||
            kind == "Assign" || kind == "AnnAssign" || kind == "Pass" || kind == "Raise") {
            return true;
        }
        // Scope-aware control flow (further checked in check_scope_control)
        if (is_scope_control_kind(kind)) {
            return true;
        }
        // Imports
        if (kind == "ImportFrom" || kind == "alias") {
            return true;
        }
        // Expressions (+ Call keyword nodes)
        if (kind == "Name" || kind == "Attribute" || kind == "Call" || kind == "Constant" ||
            kind == "BinOp" || kind == "UnaryOp" || kind == "Compare" || kind == "BoolOp" ||
            kind == "Subscript" || kind == "Tuple" || kind == "List" || kind == "Dict" ||
            kind == "Starred" || kind == "keyword" || kind == "IfExp") {
            return true;
        }
        // Operators / ctx as nested nodes
        if (is_operator_or_ctx_kind(kind)) {
            return true;
        }
        return false;
    }

    [[nodiscard]] static Status check_scope_control(const DslAstNode& node, const GateState& state,
                                                    Loc loc) {
        const std::string& kind = node.kind();
        const DslExecScope scope = scope_of(node, state);

        if (kind == "If") {
            // HotLoop divergent-if → E033 in DslDivergenceGate (follow-on).
            return Status::success();
        }

        if (kind == "For" || kind == "While") {
            if (scope.is_hot_loop()) {
                if (state.directives != nullptr &&
                    state.directives->honor(DslDirectiveTable::flag_hotloop_restriction, node,
                                            DslRuleId::E034_hotloop_control)) {
                    return Status::success();
                }
                return fail(DslRuleId::E034_hotloop_control,
                            std::string(kind) + " not allowed in HotLoop", state.source_path, loc);
            }
            return Status::success();
        }

        if (kind == "Break" || kind == "Continue") {
            if (scope.is_hot_loop()) {
                if (state.directives != nullptr &&
                    state.directives->honor(DslDirectiveTable::flag_hotloop_restriction, node,
                                            DslRuleId::E034_hotloop_control)) {
                    return Status::success();
                }
                return fail(DslRuleId::E034_hotloop_control,
                            std::string(kind) + " not allowed in HotLoop", state.source_path, loc);
            }
            if (!scope.in_loop()) {
                return fail(DslRuleId::E031_forbidden_construct,
                            std::string(kind) + " outside a loop", state.source_path, loc);
            }
            return Status::success();
        }

        return Status::success();
    }

    [[nodiscard]] static Status check_op_string(std::string_view field, std::string_view op,
                                                const GateState& state, Loc loc,
                                                std::string_view parent_kind) {
        if (field == "ctx") {
            if (!is_ctx_kind(op)) {
                return fail(DslRuleId::E031_forbidden_construct,
                            "ctx '" + std::string(op) + "' is not allowed", state.source_path, loc);
            }
            return Status::success();
        }
        if (field == "op" || field == "ops") {
            bool ok = false;
            if (parent_kind == "BinOp") {
                ok = is_binop_kind(op);
            } else if (parent_kind == "UnaryOp") {
                ok = is_unaryop_kind(op);
            } else if (parent_kind == "BoolOp") {
                ok = is_boolop_kind(op);
            } else if (parent_kind == "Compare") {
                ok = is_cmpop_kind(op);
            } else {
                // Nested op node already validated via kind whitelist; strings on
                // unexpected parents are rejected.
                ok = is_operator_or_ctx_kind(op);
            }
            if (!ok) {
                return fail(DslRuleId::E031_forbidden_construct,
                            "operator '" + std::string(op) + "' is not allowed", state.source_path,
                            loc);
            }
            return Status::success();
        }
        return Status::success();
    }

    [[nodiscard]] static Status check_import_from(const DslAstNode& node, const GateState& state) {
        const Loc loc = location_of(node, state);

        const DslAstValue* level_v = node.find_field("level");
        if (level_v != nullptr && level_v->type() == DslAstValue::Type::Int &&
            level_v->as_int() != 0) {
            return fail(DslRuleId::E021_import_whitelist,
                        "relative imports are not allowed (ImportFrom.level must be 0)",
                        state.source_path, loc);
        }

        const DslAstValue* module_v = node.find_field("module");
        if (module_v == nullptr || module_v->is_null()) {
            return fail(DslRuleId::E021_import_whitelist,
                        "ImportFrom.module is required (relative imports are not allowed)",
                        state.source_path, loc);
        }
        if (module_v->type() != DslAstValue::Type::String) {
            return fail(DslRuleId::E021_import_whitelist, "ImportFrom.module must be a string",
                        state.source_path, loc);
        }
        const std::string& module = module_v->as_string();
        if (!is_allowed_import_module(module)) {
            return fail(DslRuleId::E021_import_whitelist,
                        "import '" + module + "' not in parcae.dsl.* whitelist", state.source_path,
                        loc);
        }
        return Status::success();
    }

    /// `z29_*` Call names must be on `DslZ29Builtins` (arity checked when known).
    /// Non-`z29_` names (custom primitives / methods) pass through.
    [[nodiscard]] static Status check_call(const DslAstNode& node, const GateState& state,
                                           Loc loc) {
        const DslAstValue* func = node.find_field("func");
        if (func == nullptr || func->type() != DslAstValue::Type::Node || func->as_node() == nullptr) {
            return Status::success(); // malformed Call → BuildIr E032
        }
        const DslAstNode& f = *func->as_node();
        if (f.kind() != "Name") {
            return Status::success(); // Attribute / other → later stages
        }
        const DslAstValue* idv = f.find_field("id");
        if (idv == nullptr || idv->type() != DslAstValue::Type::String) {
            return Status::success();
        }
        const std::string& id = idv->as_string();
        if (!DslZ29Builtins::is_allowed_call_name(id)) {
            return fail(DslRuleId::E032_primitive_body,
                        "unknown z29_* intrinsic '" + id + "' (not on DslZ29Builtins allowlist)",
                        state.source_path, loc);
        }
        const std::optional<std::size_t> expected = DslZ29Builtins::arity(id);
        if (!expected.has_value()) {
            return Status::success();
        }
        const DslAstValue* args_v = node.find_field("args");
        if (args_v == nullptr || args_v->type() != DslAstValue::Type::Array) {
            return Status::success();
        }
        const std::size_t got = args_v->as_array().size();
        if (got != expected.value()) {
            return fail(DslRuleId::E032_primitive_body,
                        "call '" + id + "' expects " + std::to_string(expected.value()) +
                            " arguments, got " + std::to_string(got),
                        state.source_path, loc);
        }
        return Status::success();
    }

    [[nodiscard]] static Status check_arguments(const DslAstNode& node, const GateState& state) {
        const Loc loc = location_of(node, state);
        for (const char* field : {"vararg", "kwarg"}) {
            const DslAstValue* v = node.find_field(field);
            if (v != nullptr && !v->is_null()) {
                return fail(DslRuleId::E031_forbidden_construct,
                            std::string("arguments.") + field + " is not allowed in v0 DSL",
                            state.source_path, loc);
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status check_keyword(const DslAstNode& node, const GateState& state) {
        const DslAstValue* arg = node.find_field("arg");
        // CPython uses arg=null for **kwargs.
        if (arg == nullptr || arg->is_null()) {
            return fail(DslRuleId::E031_forbidden_construct, "**kwargs is not allowed in v0 DSL",
                        state.source_path, location_of(node, state));
        }
        return Status::success();
    }

    [[nodiscard]] static Status walk_value(const DslAstValue& value, GateState& state,
                                           int function_depth, std::string_view parent_kind,
                                           std::string_view field_name) {
        switch (value.type()) {
        case DslAstValue::Type::Null:
        case DslAstValue::Type::Bool:
        case DslAstValue::Type::Int:
        case DslAstValue::Type::Float:
            return Status::success();
        case DslAstValue::Type::String: {
            if (field_name == "op" || field_name == "ops" || field_name == "ctx") {
                return check_op_string(field_name, value.as_string(), state, state.nearest,
                                       parent_kind);
            }
            return Status::success();
        }
        case DslAstValue::Type::Node:
            if (!value.as_node()) {
                return Status::success();
            }
            return walk_node(*value.as_node(), state, function_depth);
        case DslAstValue::Type::Array: {
            for (const DslAstValue& item : value.as_array()) {
                if (item.type() == DslAstValue::Type::String &&
                    (field_name == "ops" || field_name == "op" || field_name == "ctx")) {
                    Status st = check_op_string(field_name, item.as_string(), state, state.nearest,
                                                parent_kind);
                    if (!st.ok()) {
                        return st;
                    }
                    continue;
                }
                Status st = walk_value(item, state, function_depth, parent_kind, field_name);
                if (!st.ok()) {
                    return st;
                }
            }
            return Status::success();
        }
        }
        return Status::success();
    }

    [[nodiscard]] static Status walk_node(const DslAstNode& node, GateState& state,
                                          int function_depth) {
        GateState child_state = state;
        if (node.lineno().has_value()) {
            child_state.nearest = Loc{node.lineno(), node.col_offset()};
        }
        const Loc loc = location_of(node, child_state);
        const std::string& kind = node.kind();

        // Bare Import is explicitly forbidden (even though not on the allow-list).
        if (kind == "Import") {
            return fail(DslRuleId::E031_forbidden_construct,
                        "forbidden construct 'Import' (use ImportFrom from parcae.dsl.*)",
                        child_state.source_path, loc);
        }

        if (is_always_forbidden_async_loop(kind)) {
            return fail(DslRuleId::E031_forbidden_construct, "forbidden construct '" + kind + "'",
                        child_state.source_path, loc);
        }

        if (!is_allowed_kind(kind)) {
            return fail(DslRuleId::E031_forbidden_construct, "forbidden construct '" + kind + "'",
                        child_state.source_path, loc);
        }

        if (is_scope_control_kind(kind)) {
            Status control = check_scope_control(node, child_state, loc);
            if (!control.ok()) {
                return control;
            }
        }

        if (kind == "Starred") {
            return fail(DslRuleId::E031_forbidden_construct,
                        "starred expressions are not allowed in v0 DSL", child_state.source_path,
                        loc);
        }

        if (kind == "ClassDef" && function_depth > 0) {
            return fail(DslRuleId::E031_forbidden_construct,
                        "ClassDef nested inside functions is not allowed in v0 DSL",
                        child_state.source_path, loc);
        }

        if (kind == "ImportFrom") {
            Status st = check_import_from(node, child_state);
            if (!st.ok()) {
                return st;
            }
        }

        if (kind == "Call") {
            Status st = check_call(node, child_state, loc);
            if (!st.ok()) {
                return st;
            }
        }

        if (kind == "arguments") {
            Status st = check_arguments(node, child_state);
            if (!st.ok()) {
                return st;
            }
        }

        if (kind == "keyword") {
            Status st = check_keyword(node, child_state);
            if (!st.ok()) {
                return st;
            }
        }

        const int next_function_depth =
            (kind == "FunctionDef") ? function_depth + 1 : function_depth;

        for (const auto& entry : node.fields()) {
            Status st =
                walk_value(entry.second, child_state, next_function_depth, kind, entry.first);
            if (!st.ok()) {
                return st;
            }
        }
        return Status::success();
    }
};

#endif // DSL_SEMANTIC_GATE_HPP
