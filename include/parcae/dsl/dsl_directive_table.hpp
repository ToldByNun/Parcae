#ifndef DSL_DIRECTIVE_TABLE_HPP
#define DSL_DIRECTIVE_TABLE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/// Bind `#ignore DSL_FLAG:…` entries from `DslAstDocument::directives()` to AST
/// statements and honor them during semantic / divergence / host-glue gates.
///
/// Normative: docs/spec/dsl-ast-json.md § Directives, docs/spec/dsl.md.
///
/// Binding: a directive on line `L` applies to the **first statement** in the
/// **innermost** statement body whose parent started before `L` and that
/// contains a stmt with `lineno >= L` (same-indent-block approximation without
/// tokenizer DEDENT).
///
/// Without `allow_dsl_ignores`, any non-empty `directives[]` is **E031**.
/// Successful suppress emits **W010** (never alone fails compile).
class DslDirectiveTable {
public:
    static constexpr std::string_view flag_divergent_branch = "divergent_branch";
    static constexpr std::string_view flag_hotloop_restriction = "hotloop_restriction";
    static constexpr std::string_view flag_host_loop_bound = "host_loop_bound";

    /// Author-asserted OuterControl while cap when `host_loop_bound` is honored.
    static constexpr std::int64_t host_loop_bound_asserted_max = 1'000'000;

    class Options {
    public:
        // Explicit ctor so Options{} is usable as a default arg of
        // DslDirectiveTable::build on GCC/Clang (nested DMIs are not).
        Options() noexcept : allow_dsl_ignores_(false) {}

        [[nodiscard]] Options& set_allow_dsl_ignores(bool allow) {
            allow_dsl_ignores_ = allow;
            return *this;
        }

        [[nodiscard]] bool allow_dsl_ignores() const noexcept { return allow_dsl_ignores_; }

    private:
        bool allow_dsl_ignores_;
    };

    [[nodiscard]] static bool is_known_flag(std::string_view flag) noexcept {
        return flag == flag_divergent_branch || flag == flag_hotloop_restriction ||
               flag == flag_host_loop_bound;
    }

    [[nodiscard]] static StatusOr<DslDirectiveTable> build(const DslAstDocument& doc,
                                                           Options options = {}) {
        DslDirectiveTable table;
        table.source_path_ = doc.source_path();
        table.allow_dsl_ignores_ = options.allow_dsl_ignores();

        const std::vector<DslAstDirective>& dirs = doc.directives();
        if (dirs.empty()) {
            return table;
        }

        if (!options.allow_dsl_ignores()) {
            const int line = dirs.front().lineno();
            return DslDiag::make(DslRuleId::E031_forbidden_construct,
                                 "DSL_FLAG ignore not allowed without --allow-dsl-ignores",
                                 doc.source_path(), line, 0,
                                 "Pass --allow-dsl-ignores to honor #ignore DSL_FLAG (emits W010)")
                .to_status();
        }

        if (!doc.module()) {
            return DslDiag::make(DslRuleId::E031_forbidden_construct, "document has no module AST",
                                 doc.source_path())
                .to_status();
        }

        std::vector<BodyCtx> bodies;
        collect_bodies(*doc.module(), /*parent_lineno=*/0, /*depth=*/0, bodies);

        for (const DslAstDirective& dir : dirs) {
            if (!is_known_flag(dir.flag())) {
                return DslDiag::make(
                           DslRuleId::E031_forbidden_construct,
                           "unknown DSL_FLAG '" + dir.flag() + "'", doc.source_path(), dir.lineno(),
                           0, "Recognized: divergent_branch, hotloop_restriction, host_loop_bound")
                    .to_status();
            }

            const DslAstNode* target = bind_directive(dir.lineno(), bodies);
            if (target == nullptr) {
                return DslDiag::make(DslRuleId::E031_forbidden_construct,
                                     "DSL_FLAG '" + dir.flag() +
                                         "' has no following statement to bind",
                                     doc.source_path(), dir.lineno(), 0)
                    .to_status();
            }

            Binding b;
            b.flag = dir.flag();
            b.directive_lineno = dir.lineno();
            b.node = target;
            table.bindings_.push_back(std::move(b));
            table.by_node_[target].insert(dir.flag());
        }

        return table;
    }

    [[nodiscard]] bool allow_dsl_ignores() const noexcept { return allow_dsl_ignores_; }

    [[nodiscard]] bool covers(std::string_view flag, const DslAstNode& node) const noexcept {
        const auto it = by_node_.find(&node);
        if (it == by_node_.end()) {
            return false;
        }
        return it->second.find(std::string(flag)) != it->second.end();
    }

    /// If `node` is covered by `flag`, record **W010** and return true (suppress).
    /// Otherwise return false (caller keeps the hard diagnostic).
    [[nodiscard]] bool honor(std::string_view flag, const DslAstNode& node,
                             std::string_view suppressed_rule) {
        if (!covers(flag, node)) {
            return false;
        }
        const std::string key =
            std::string(flag) + "@" + std::to_string(reinterpret_cast<std::uintptr_t>(&node));
        if (honored_keys_.insert(key).second) {
            const int dir_line =
                directive_lineno_for(flag, node).value_or(node.lineno().value_or(0));
            std::string msg = "DSL_FLAG:";
            msg += flag;
            msg += " suppressed ";
            msg += suppressed_rule;
            warnings_.push_back(DslDiag::make(DslRuleId::W010_dsl_ignore_used, std::move(msg),
                                              source_path_, dir_line, 0));
            applied_.insert(std::string(flag));
        }
        return true;
    }

    /// Honor a flag bound to an enclosing statement whose source span covers `lineno`
    /// (for nested `IfExp` under `Return` / `Assign`).
    [[nodiscard]] bool honor_at_line(std::string_view flag, int lineno,
                                     std::string_view suppressed_rule) {
        for (const Binding& b : bindings_) {
            if (b.flag != flag || b.node == nullptr || !b.node->lineno().has_value()) {
                continue;
            }
            const int start = *b.node->lineno();
            const int end = b.node->end_lineno().value_or(start);
            if (lineno >= start && lineno <= end) {
                return honor(flag, *b.node, suppressed_rule);
            }
        }
        return false;
    }

    [[nodiscard]] const std::vector<DslDiag>& warnings() const noexcept { return warnings_; }

    [[nodiscard]] std::vector<std::string> flags_applied() const {
        std::vector<std::string> out(applied_.begin(), applied_.end());
        std::sort(out.begin(), out.end());
        return out;
    }

    [[nodiscard]] std::size_t binding_count() const noexcept { return bindings_.size(); }

private:
    struct Binding {
        std::string flag;
        int directive_lineno = 0;
        const DslAstNode* node = nullptr;
    };

    struct BodyCtx {
        int parent_lineno = 0;
        int depth = 0;
        std::vector<const DslAstNode*> stmts;
    };

    DslDirectiveTable() = default;

    [[nodiscard]] std::optional<int> directive_lineno_for(std::string_view flag,
                                                          const DslAstNode& node) const {
        for (const Binding& b : bindings_) {
            if (b.node == &node && b.flag == flag) {
                return b.directive_lineno;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] static const DslAstNode* bind_directive(int directive_lineno,
                                                          const std::vector<BodyCtx>& bodies) {
        const DslAstNode* best = nullptr;
        int best_depth = -1;
        int best_lineno = 0;
        for (const BodyCtx& body : bodies) {
            if (body.parent_lineno >= directive_lineno) {
                continue;
            }
            const DslAstNode* first = nullptr;
            int first_line = 0;
            for (const DslAstNode* stmt : body.stmts) {
                if (!stmt || !stmt->lineno().has_value()) {
                    continue;
                }
                const int ln = *stmt->lineno();
                if (ln < directive_lineno) {
                    continue;
                }
                if (first == nullptr || ln < first_line) {
                    first = stmt;
                    first_line = ln;
                }
            }
            if (first == nullptr) {
                continue;
            }
            if (body.depth > best_depth || (body.depth == best_depth && first_line < best_lineno)) {
                best = first;
                best_depth = body.depth;
                best_lineno = first_line;
            }
        }
        return best;
    }

    static void collect_bodies(const DslAstNode& node, int parent_lineno, int depth,
                               std::vector<BodyCtx>& out) {
        auto take_list = [&](const char* field) {
            const DslAstValue* list = node.find_field(field);
            if (!list || list->type() != DslAstValue::Type::Array) {
                return;
            }
            BodyCtx ctx;
            ctx.parent_lineno = parent_lineno;
            ctx.depth = depth;
            for (const DslAstValue& item : list->as_array()) {
                if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                    continue;
                }
                const DslAstNode* stmt = item.as_node().get();
                ctx.stmts.push_back(stmt);
            }
            if (!ctx.stmts.empty()) {
                out.push_back(ctx);
            }
            for (const DslAstNode* stmt : ctx.stmts) {
                const int child_parent = stmt->lineno().value_or(parent_lineno);
                collect_bodies(*stmt, child_parent, depth + 1, out);
            }
        };

        take_list("body");
        take_list("orelse");
        // ClassDef / FunctionDef already covered via body; walk non-list children
        // that may nest further statement lists (e.g. If nested under Expr — rare).
        for (const auto& field : node.fields()) {
            const std::string& key = field.first;
            if (key == "body" || key == "orelse") {
                continue;
            }
            walk_value_for_bodies(field.second, parent_lineno, depth, out);
        }
    }

    static void walk_value_for_bodies(const DslAstValue& value, int parent_lineno, int depth,
                                      std::vector<BodyCtx>& out) {
        if (value.type() == DslAstValue::Type::Node && value.as_node()) {
            collect_bodies(*value.as_node(), parent_lineno, depth, out);
        } else if (value.type() == DslAstValue::Type::Array) {
            for (const DslAstValue& item : value.as_array()) {
                walk_value_for_bodies(item, parent_lineno, depth, out);
            }
        }
    }

    std::string source_path_;
    bool allow_dsl_ignores_ = false;
    std::vector<Binding> bindings_;
    std::unordered_map<const DslAstNode*, std::unordered_set<std::string>> by_node_;
    std::vector<DslDiag> warnings_;
    std::unordered_set<std::string> applied_;
    std::unordered_set<std::string> honored_keys_;
};

#endif // DSL_DIRECTIVE_TABLE_HPP
