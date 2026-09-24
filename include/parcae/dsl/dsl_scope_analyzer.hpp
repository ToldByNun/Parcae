#ifndef DSL_SCOPE_ANALYZER_HPP
#define DSL_SCOPE_ANALYZER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_exec_scope.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

/// Per-node scope map produced by `DslScopeAnalyzer` for one ingested document.
/// Keys are raw pointers into that document's live AST (valid only while the
/// document / shared_ptr graph remains alive).
class DslScopeMap {
public:
    void set(const DslAstNode* node, DslExecScope scope) {
        if (node == nullptr) {
            return;
        }
        scopes_[node] = scope;
    }

    [[nodiscard]] bool contains(const DslAstNode* node) const noexcept {
        return node != nullptr && scopes_.find(node) != scopes_.end();
    }

    [[nodiscard]] std::optional<DslExecScope> get(const DslAstNode* node) const {
        if (node == nullptr) {
            return std::nullopt;
        }
        const auto it = scopes_.find(node);
        if (it == scopes_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    [[nodiscard]] std::size_t size() const noexcept { return scopes_.size(); }

private:
    std::unordered_map<const DslAstNode*, DslExecScope> scopes_;
};

/// Walk a `DslAstDocument` and classify OuterControl vs HotLoop (+ loop depth).
///
/// Normative: docs/spec/dsl.md § Execution scopes.
/// Does not reject forbidden constructs (that remains `DslSemanticGate`).
class DslScopeAnalyzer {
public:
    [[nodiscard]] static StatusOr<DslScopeMap> analyze(const DslAstDocument& doc) {
        if (!doc.module()) {
            return DslDiag::make(DslRuleId::E031_forbidden_construct, "document has no module AST",
                                 doc.source_path())
                .to_status();
        }
        DslScopeMap map;
        walk_node(*doc.module(), DslExecScope{DslExecScope::Kind::OuterControl}, map);
        return map;
    }

private:
    DslScopeAnalyzer() = delete;

    [[nodiscard]] static bool is_loop_kind(std::string_view kind) noexcept {
        return kind == "For" || kind == "While" || kind == "AsyncFor";
    }

    [[nodiscard]] static bool is_hot_loop_method_name(std::string_view name) noexcept {
        return name == "encrypt_step" || name == "decrypt_step" || name == "keystream_at" ||
               name == "interrupt_policy";
    }

    [[nodiscard]] static std::optional<std::string> function_name(const DslAstNode& fn) {
        if (fn.kind() != "FunctionDef" && fn.kind() != "AsyncFunctionDef") {
            return std::nullopt;
        }
        const DslAstValue* name = fn.find_field("name");
        if (!name || name->type() != DslAstValue::Type::String) {
            return std::nullopt;
        }
        return name->as_string();
    }

    [[nodiscard]] static bool decorator_list_has(const DslAstNode& defn,
                                                 std::string_view deco_name) {
        const DslAstValue* list = defn.find_field("decorator_list");
        if (!list || list->type() != DslAstValue::Type::Array) {
            return false;
        }
        for (const DslAstValue& item : list->as_array()) {
            if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                continue;
            }
            const DslAstNode& deco = *item.as_node();
            if (deco.kind() == "Name") {
                const DslAstValue* idv = deco.find_field("id");
                if (idv && idv->type() == DslAstValue::Type::String &&
                    idv->as_string() == deco_name) {
                    return true;
                }
                continue;
            }
            if (deco.kind() != "Call") {
                continue;
            }
            const DslAstValue* func = deco.find_field("func");
            if (!func || func->type() != DslAstValue::Type::Node || !func->as_node()) {
                continue;
            }
            const DslAstNode& f = *func->as_node();
            if (f.kind() != "Name") {
                continue;
            }
            const DslAstValue* idv = f.find_field("id");
            if (idv && idv->type() == DslAstValue::Type::String && idv->as_string() == deco_name) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static bool is_theory_class(const DslAstNode& cls) {
        return decorator_list_has(cls, "Theory") || decorator_list_has(cls, "ComposedTheory");
    }

    static void walk_value(const DslAstValue& value, const DslExecScope& scope, DslScopeMap& map) {
        if (value.type() == DslAstValue::Type::Node && value.as_node()) {
            walk_node(*value.as_node(), scope, map);
            return;
        }
        if (value.type() == DslAstValue::Type::Array) {
            for (const DslAstValue& child : value.as_array()) {
                walk_value(child, scope, map);
            }
        }
    }

    static void walk_fields(const DslAstNode& node, const DslExecScope& scope, DslScopeMap& map) {
        for (const auto& entry : node.fields()) {
            walk_value(entry.second, scope, map);
        }
    }

    static void walk_stmt_list(const DslAstValue* list, const DslExecScope& scope,
                               DslScopeMap& map) {
        if (!list || list->type() != DslAstValue::Type::Array) {
            return;
        }
        for (const DslAstValue& item : list->as_array()) {
            if (item.type() == DslAstValue::Type::Node && item.as_node()) {
                walk_node(*item.as_node(), scope, map);
            }
        }
    }

    /// Walk a function body under `body_scope` (decorators/args stay OuterControl-ish
    /// of the caller — recorded with `header_scope`).
    static void walk_function_def(const DslAstNode& fn, const DslExecScope& header_scope,
                                  const DslExecScope& body_scope, DslScopeMap& map) {
        map.set(&fn, header_scope);
        for (const auto& entry : fn.fields()) {
            if (entry.first == "body") {
                walk_stmt_list(&entry.second, body_scope, map);
                continue;
            }
            // Decorators / args / returns are structural OuterControl metadata.
            walk_value(entry.second, header_scope.with_kind(DslExecScope::Kind::OuterControl), map);
        }
    }

    static void walk_class_def(const DslAstNode& cls, const DslExecScope& outer, DslScopeMap& map) {
        map.set(&cls, outer);
        const bool theory = is_theory_class(cls);
        for (const auto& entry : cls.fields()) {
            if (entry.first == "body" && entry.second.type() == DslAstValue::Type::Array) {
                for (const DslAstValue& item : entry.second.as_array()) {
                    if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                        continue;
                    }
                    const DslAstNode& member = *item.as_node();
                    if (member.kind() == "FunctionDef" || member.kind() == "AsyncFunctionDef") {
                        const std::optional<std::string> name = function_name(member);
                        const bool hot_method =
                            theory && name.has_value() && is_hot_loop_method_name(*name);
                        const DslExecScope body_scope =
                            hot_method ? DslExecScope{DslExecScope::Kind::HotLoop}
                                       : DslExecScope{DslExecScope::Kind::OuterControl};
                        walk_function_def(member, outer, body_scope, map);
                        continue;
                    }
                    walk_node(member, outer, map);
                }
                continue;
            }
            walk_value(entry.second, outer, map);
        }
    }

    static void walk_node(const DslAstNode& node, const DslExecScope& incoming, DslScopeMap& map) {
        const std::string& kind = node.kind();

        if (kind == "Module") {
            map.set(&node, incoming);
            walk_stmt_list(node.find_field("body"), incoming, map);
            // type_ignores etc.
            for (const auto& entry : node.fields()) {
                if (entry.first == "body") {
                    continue;
                }
                walk_value(entry.second, incoming, map);
            }
            return;
        }

        if (kind == "FunctionDef" || kind == "AsyncFunctionDef") {
            // Module-level or nested function not handled via ClassDef path.
            const bool primitive = decorator_list_has(node, "define_primitive");
            const DslExecScope body_scope = primitive
                                                ? DslExecScope{DslExecScope::Kind::HotLoop}
                                                : DslExecScope{DslExecScope::Kind::OuterControl};
            // Nested defs inherit OuterControl headers even under HotLoop callers.
            walk_function_def(node, incoming.with_kind(DslExecScope::Kind::OuterControl),
                              body_scope, map);
            return;
        }

        if (kind == "ClassDef") {
            walk_class_def(node, incoming.with_kind(DslExecScope::Kind::OuterControl), map);
            return;
        }

        if (is_loop_kind(kind)) {
            const DslExecScope loop_scope = incoming.enter_loop();
            map.set(&node, loop_scope);
            walk_fields(node, loop_scope, map);
            return;
        }

        map.set(&node, incoming);
        walk_fields(node, incoming, map);
    }
};

#endif // DSL_SCOPE_ANALYZER_HPP
