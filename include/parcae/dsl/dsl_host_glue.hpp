#ifndef DSL_HOST_GLUE_HPP
#define DSL_HOST_GLUE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_exec_scope.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/dsl_scope_analyzer.hpp"
#include "parcae/dsl/host_glue_ir.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Lower OuterControl `for` / `while` / `if` into `HostGlueIr` (docs/spec/dsl.md).
///
/// Policy:
/// 1. Const-bounded `for`/`range` → `ForRange` with `ConstUnroll`
/// 2. Param-/Name-bounded finite `range` → `ForRange` with `HostKnown`
/// 3. Provably finite `while` (const max) → `WhileBounded`
/// 4. Otherwise OuterControl `while` / non-range `for` → **E035**
///
/// HotLoop loops remain `DslSemanticGate` **E034** (this class skips HotLoop).
class DslHostGlue {
public:
    class Program {
    public:
        explicit Program(HostGlueIr::Ptr root = HostGlueIr::make_seq({}))
            : root_(std::move(root)) {}

        [[nodiscard]] const HostGlueIr::Ptr& root() const noexcept {
            return root_;
        }

        [[nodiscard]] std::size_t for_count() const noexcept {
            return for_count_;
        }

        [[nodiscard]] std::size_t while_count() const noexcept {
            return while_count_;
        }

        void set_counts(std::size_t fors, std::size_t whiles) {
            for_count_ = fors;
            while_count_ = whiles;
        }

    private:
        HostGlueIr::Ptr root_;
        std::size_t for_count_ = 0;
        std::size_t while_count_ = 0;
    };

    /// Walk OuterControl regions; fail-loud on E035; return host glue program.
    [[nodiscard]] static StatusOr<Program> build(const DslAstDocument& doc) {
        if (!doc.module()) {
            return DslDiag::make(
                       DslRuleId::E031_forbidden_construct,
                       "document has no module AST",
                       doc.source_path())
                .to_status();
        }

        StatusOr<DslScopeMap> scopes = DslScopeAnalyzer::analyze(doc);
        if (!scopes.ok()) {
            return scopes.status();
        }

        State state;
        state.source_path = doc.source_path();
        state.scopes = &scopes.value();
        StatusOr<HostGlueIr::Ptr> root = lower_stmt_list(
            doc.module()->find_field("body"), state, /*require_outer=*/true);
        if (!root.ok()) {
            return root.status();
        }
        Program prog{root.value()};
        prog.set_counts(state.for_count, state.while_count);
        return prog;
    }

    /// Errors only (compile pipeline).
    [[nodiscard]] static Status check_errors_only(const DslAstDocument& doc) {
        StatusOr<Program> p = build(doc);
        if (!p.ok()) {
            return p.status();
        }
        return Status::success();
    }

private:
    struct State {
        std::string source_path;
        const DslScopeMap* scopes = nullptr;
        std::size_t for_count = 0;
        std::size_t while_count = 0;
    };

    DslHostGlue() = delete;

    [[nodiscard]] static DslExecScope scope_of(const DslAstNode& node, const State& state) {
        if (state.scopes != nullptr) {
            const std::optional<DslExecScope> found = state.scopes->get(&node);
            if (found.has_value()) {
                return found.value();
            }
        }
        return DslExecScope{DslExecScope::Kind::OuterControl};
    }

    [[nodiscard]] static Status fail(
        std::string_view rule,
        std::string message,
        const State& state,
        const DslAstNode& node,
        std::string hint = {}) {
        return DslDiag::make(
                   rule,
                   std::move(message),
                   state.source_path,
                   node.lineno(),
                   node.col_offset(),
                   std::move(hint))
            .to_status();
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_stmt_list(
        const DslAstValue* list,
        State& state,
        bool require_outer) {
        std::vector<HostGlueIr::Ptr> stmts;
        if (!list || list->type() != DslAstValue::Type::Array) {
            return HostGlueIr::make_seq(std::move(stmts));
        }
        for (const DslAstValue& item : list->as_array()) {
            if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                continue;
            }
            const DslAstNode& stmt = *item.as_node();
            // Skip HotLoop regions entirely (E034 already covers illegal loops).
            if (require_outer && scope_of(stmt, state).is_hot_loop()) {
                continue;
            }
            StatusOr<HostGlueIr::Ptr> one = lower_stmt(stmt, state);
            if (!one.ok()) {
                return one.status();
            }
            if (one.value()->kind() != HostGlueIr::Kind::Pass) {
                stmts.push_back(std::move(one.value()));
            }
        }
        if (stmts.size() == 1) {
            return stmts.front();
        }
        return HostGlueIr::make_seq(std::move(stmts));
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_stmt(
        const DslAstNode& stmt,
        State& state) {
        const std::string& kind = stmt.kind();
        if (kind == "Pass") {
            return HostGlueIr::make_pass();
        }
        if (kind == "Expr") {
            // Expression statements: keep Calls we care about; else Pass.
            const DslAstValue* value = stmt.find_field("value");
            if (value && value->type() == DslAstValue::Type::Node && value->as_node() &&
                value->as_node()->kind() == "Call") {
                return lower_call(*value->as_node(), state);
            }
            return HostGlueIr::make_pass();
        }
        if (kind == "Assign" || kind == "AnnAssign") {
            return lower_assign(stmt, state);
        }
        if (kind == "If") {
            return lower_if(stmt, state);
        }
        if (kind == "For") {
            return lower_for(stmt, state);
        }
        if (kind == "While") {
            return lower_while(stmt, state);
        }
        if (kind == "FunctionDef" || kind == "AsyncFunctionDef" || kind == "ClassDef") {
            // Nested defs: walk OuterControl methods' bodies for host loops.
            return lower_nested_def(stmt, state);
        }
        if (kind == "Return" || kind == "Break" || kind == "Continue" || kind == "Raise" ||
            kind == "ImportFrom") {
            return HostGlueIr::make_pass();
        }
        return HostGlueIr::make_pass();
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_nested_def(
        const DslAstNode& defn,
        State& state) {
        // ClassDef: walk member FunctionDefs that are OuterControl
        // (step_params, helpers). FunctionDef: walk body if OuterControl.
        if (defn.kind() == "ClassDef") {
            const DslAstValue* body = defn.find_field("body");
            if (!body || body->type() != DslAstValue::Type::Array) {
                return HostGlueIr::make_pass();
            }
            std::vector<HostGlueIr::Ptr> parts;
            for (const DslAstValue& item : body->as_array()) {
                if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                    continue;
                }
                const DslAstNode& member = *item.as_node();
                if (member.kind() != "FunctionDef" && member.kind() != "AsyncFunctionDef") {
                    continue;
                }
                // HotLoop methods (encrypt_step etc.) skipped via scope on body stmts.
                StatusOr<HostGlueIr::Ptr> sub =
                    lower_stmt_list(member.find_field("body"), state, /*require_outer=*/true);
                if (!sub.ok()) {
                    return sub.status();
                }
                if (sub.value()->kind() != HostGlueIr::Kind::Pass &&
                    !(sub.value()->kind() == HostGlueIr::Kind::Seq &&
                      sub.value()->children().empty())) {
                    parts.push_back(std::move(sub.value()));
                }
            }
            if (parts.empty()) {
                return HostGlueIr::make_pass();
            }
            if (parts.size() == 1) {
                return parts.front();
            }
            return HostGlueIr::make_seq(std::move(parts));
        }

        // Module-level FunctionDef that is not HotLoop (no define_primitive).
        if (scope_of(defn, state).is_hot_loop()) {
            return HostGlueIr::make_pass();
        }
        // Body stmts carry scope — require_outer filters HotLoop.
        // For define_primitive, body is HotLoop → empty seq.
        return lower_stmt_list(defn.find_field("body"), state, /*require_outer=*/true);
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_assign(
        const DslAstNode& stmt,
        State& state) {
        std::string target;
        if (stmt.kind() == "AnnAssign") {
            const DslAstValue* t = stmt.find_field("target");
            if (t && t->type() == DslAstValue::Type::Node && t->as_node() &&
                t->as_node()->kind() == "Name") {
                const DslAstValue* idv = t->as_node()->find_field("id");
                if (idv && idv->type() == DslAstValue::Type::String) {
                    target = idv->as_string();
                }
            }
        } else {
            const DslAstValue* targets = stmt.find_field("targets");
            if (targets && targets->type() == DslAstValue::Type::Array &&
                !targets->as_array().empty() &&
                targets->as_array().front().type() == DslAstValue::Type::Node &&
                targets->as_array().front().as_node() &&
                targets->as_array().front().as_node()->kind() == "Name") {
                const DslAstValue* idv =
                    targets->as_array().front().as_node()->find_field("id");
                if (idv && idv->type() == DslAstValue::Type::String) {
                    target = idv->as_string();
                }
            }
        }
        if (target.empty()) {
            return HostGlueIr::make_pass();
        }
        const DslAstValue* value = stmt.find_field("value");
        StatusOr<HostGlueIr::Ptr> v = lower_expr_value(value, state, stmt);
        if (!v.ok()) {
            return v.status();
        }
        auto node = HostGlueIr::make_assign(std::move(target), std::move(v.value()));
        node->set_location(stmt.lineno(), stmt.col_offset());
        return node;
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_if(
        const DslAstNode& stmt,
        State& state) {
        const DslAstValue* test = stmt.find_field("test");
        StatusOr<HostGlueIr::Ptr> cond = lower_expr_value(test, state, stmt);
        if (!cond.ok()) {
            return cond.status();
        }
        StatusOr<HostGlueIr::Ptr> then_b =
            lower_stmt_list(stmt.find_field("body"), state, /*require_outer=*/true);
        if (!then_b.ok()) {
            return then_b.status();
        }
        StatusOr<HostGlueIr::Ptr> else_b =
            lower_stmt_list(stmt.find_field("orelse"), state, /*require_outer=*/true);
        if (!else_b.ok()) {
            return else_b.status();
        }
        auto node = HostGlueIr::make_if(
            std::move(cond.value()), std::move(then_b.value()), std::move(else_b.value()));
        node->set_location(stmt.lineno(), stmt.col_offset());
        return node;
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_for(
        const DslAstNode& stmt,
        State& state) {
        if (scope_of(stmt, state).is_hot_loop()) {
            return HostGlueIr::make_pass();  // E034 elsewhere
        }

        const DslAstValue* target_v = stmt.find_field("target");
        std::string target = "i";
        if (target_v && target_v->type() == DslAstValue::Type::Node && target_v->as_node() &&
            target_v->as_node()->kind() == "Name") {
            const DslAstValue* idv = target_v->as_node()->find_field("id");
            if (idv && idv->type() == DslAstValue::Type::String) {
                target = idv->as_string();
            }
        }

        const DslAstValue* iter = stmt.find_field("iter");
        StatusOr<RangeBounds> bounds = parse_range_call(iter, state, stmt);
        if (!bounds.ok()) {
            return bounds.status();
        }

        StatusOr<HostGlueIr::Ptr> body =
            lower_stmt_list(stmt.find_field("body"), state, /*require_outer=*/true);
        if (!body.ok()) {
            return body.status();
        }

        ++state.for_count;
        auto node = HostGlueIr::make_for_range(
            std::move(target),
            std::move(bounds.value().start),
            std::move(bounds.value().stop),
            std::move(bounds.value().step),
            std::move(body.value()),
            bounds.value().bound);
        node->set_location(stmt.lineno(), stmt.col_offset());
        return node;
    }

    struct RangeBounds {
        HostGlueIr::Ptr start;
        HostGlueIr::Ptr stop;
        HostGlueIr::Ptr step;
        HostGlueIr::BoundKind bound = HostGlueIr::BoundKind::ConstUnroll;
    };

    [[nodiscard]] static StatusOr<RangeBounds> parse_range_call(
        const DslAstValue* iter,
        State& state,
        const DslAstNode& for_stmt) {
        if (!iter || iter->type() != DslAstValue::Type::Node || !iter->as_node() ||
            iter->as_node()->kind() != "Call") {
            return fail(
                DslRuleId::E035_host_loop_unbounded,
                "OuterControl for requires range(...) with a finite bound",
                state,
                for_stmt,
                "Use for i in range(N) with const or Param N");
        }
        const DslAstNode& call = *iter->as_node();
        const DslAstValue* func = call.find_field("func");
        if (!func || func->type() != DslAstValue::Type::Node || !func->as_node() ||
            func->as_node()->kind() != "Name") {
            return fail(
                DslRuleId::E035_host_loop_unbounded,
                "OuterControl for requires range(...)",
                state,
                for_stmt);
        }
        const DslAstValue* fname = func->as_node()->find_field("id");
        if (!fname || fname->type() != DslAstValue::Type::String || fname->as_string() != "range") {
            return fail(
                DslRuleId::E035_host_loop_unbounded,
                "OuterControl for iter must be range(...), got '" +
                    (fname && fname->type() == DslAstValue::Type::String ? fname->as_string()
                                                                        : std::string("?")) +
                    "'",
                state,
                for_stmt);
        }

        const DslAstValue* args = call.find_field("args");
        if (!args || args->type() != DslAstValue::Type::Array || args->as_array().empty() ||
            args->as_array().size() > 3) {
            return fail(
                DslRuleId::E035_host_loop_unbounded,
                "range() must have 1..3 arguments",
                state,
                for_stmt);
        }

        const auto& argv = args->as_array();
        HostGlueIr::Ptr start = HostGlueIr::make_const_int(0);
        HostGlueIr::Ptr stop;
        HostGlueIr::Ptr step = HostGlueIr::make_const_int(1);
        HostGlueIr::BoundKind bound = HostGlueIr::BoundKind::ConstUnroll;

        auto classify_arg = [&](const DslAstValue& a,
                                HostGlueIr::Ptr& out) -> Status {
            StatusOr<HostGlueIr::Ptr> e = lower_expr_value(&a, state, for_stmt);
            if (!e.ok()) {
                return e.status();
            }
            out = std::move(e.value());
            if (out->kind() == HostGlueIr::Kind::ConstInt) {
                return Status::success();
            }
            if (out->kind() == HostGlueIr::Kind::Name) {
                bound = HostGlueIr::BoundKind::HostKnown;
                return Status::success();
            }
            return fail(
                DslRuleId::E035_host_loop_unbounded,
                "range() bound is not a compile-time constant or host Param name",
                state,
                for_stmt);
        };

        if (argv.size() == 1) {
            Status st = classify_arg(argv[0], stop);
            if (!st.ok()) {
                return st;
            }
        } else if (argv.size() == 2) {
            Status st = classify_arg(argv[0], start);
            if (!st.ok()) {
                return st;
            }
            st = classify_arg(argv[1], stop);
            if (!st.ok()) {
                return st;
            }
        } else {
            Status st = classify_arg(argv[0], start);
            if (!st.ok()) {
                return st;
            }
            st = classify_arg(argv[1], stop);
            if (!st.ok()) {
                return st;
            }
            st = classify_arg(argv[2], step);
            if (!st.ok()) {
                return st;
            }
        }

        // If any bound is HostKnown, mark the loop HostKnown.
        if (start->kind() == HostGlueIr::Kind::Name || stop->kind() == HostGlueIr::Kind::Name ||
            step->kind() == HostGlueIr::Kind::Name) {
            bound = HostGlueIr::BoundKind::HostKnown;
        }

        return RangeBounds{std::move(start), std::move(stop), std::move(step), bound};
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_while(
        const DslAstNode& stmt,
        State& state) {
        if (scope_of(stmt, state).is_hot_loop()) {
            return HostGlueIr::make_pass();
        }

        const DslAstValue* test = stmt.find_field("test");
        // Const False → dead loop (Pass).
        if (test && test->type() == DslAstValue::Type::Node && test->as_node() &&
            test->as_node()->kind() == "Constant") {
            const DslAstValue* val = test->as_node()->find_field("value");
            if (val && val->type() == DslAstValue::Type::Bool && !val->as_bool()) {
                return HostGlueIr::make_pass();
            }
            if (val && val->type() == DslAstValue::Type::Int && val->as_int() == 0) {
                return HostGlueIr::make_pass();
            }
        }

        // Provably finite: while i < N with N const → max_iters = N (conservative).
        std::optional<std::int64_t> max_iters = prove_while_max_iters(test);
        if (!max_iters.has_value()) {
            return fail(
                DslRuleId::E035_host_loop_unbounded,
                "OuterControl while without provable finite bound",
                state,
                stmt,
                "Use a const upper bound (while i < N) or for-range; "
                "#ignore DSL_FLAG:host_loop_bound is a follow-on");
        }

        StatusOr<HostGlueIr::Ptr> cond = lower_expr_value(test, state, stmt);
        if (!cond.ok()) {
            return cond.status();
        }
        StatusOr<HostGlueIr::Ptr> body =
            lower_stmt_list(stmt.find_field("body"), state, /*require_outer=*/true);
        if (!body.ok()) {
            return body.status();
        }

        ++state.while_count;
        auto node = HostGlueIr::make_while_bounded(
            std::move(cond.value()),
            std::move(body.value()),
            *max_iters,
            HostGlueIr::BoundKind::ConstUnroll);
        node->set_location(stmt.lineno(), stmt.col_offset());
        return node;
    }

    /// Accept `while <Name> < Lt/LtE > <Constant int>` as finite with max=const
    /// (or const+1 for LtE). Everything else → nullopt → E035.
    [[nodiscard]] static std::optional<std::int64_t> prove_while_max_iters(
        const DslAstValue* test) {
        if (!test || test->type() != DslAstValue::Type::Node || !test->as_node()) {
            return std::nullopt;
        }
        const DslAstNode& node = *test->as_node();
        if (node.kind() != "Compare") {
            return std::nullopt;
        }
        const DslAstValue* left = node.find_field("left");
        const DslAstValue* ops = node.find_field("ops");
        const DslAstValue* comps = node.find_field("comparators");
        if (!left || left->type() != DslAstValue::Type::Node || !left->as_node() ||
            left->as_node()->kind() != "Name" || !ops || ops->type() != DslAstValue::Type::Array ||
            ops->as_array().size() != 1 || !comps || comps->type() != DslAstValue::Type::Array ||
            comps->as_array().size() != 1) {
            return std::nullopt;
        }
        std::string op;
        if (ops->as_array()[0].type() == DslAstValue::Type::String) {
            op = ops->as_array()[0].as_string();
        } else if (
            ops->as_array()[0].type() == DslAstValue::Type::Node && ops->as_array()[0].as_node()) {
            op = ops->as_array()[0].as_node()->kind();
        }
        if (op != "Lt" && op != "LtE") {
            return std::nullopt;
        }
        const DslAstValue& rhs = comps->as_array()[0];
        if (rhs.type() != DslAstValue::Type::Node || !rhs.as_node() ||
            rhs.as_node()->kind() != "Constant") {
            return std::nullopt;
        }
        const DslAstValue* val = rhs.as_node()->find_field("value");
        if (!val || val->type() != DslAstValue::Type::Int) {
            return std::nullopt;
        }
        const std::int64_t n = val->as_int();
        if (n < 0) {
            return std::nullopt;
        }
        if (op == "LtE") {
            return n + 1;
        }
        return n;
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_call(
        const DslAstNode& call,
        State& state) {
        std::string callee = "call";
        const DslAstValue* func = call.find_field("func");
        if (func && func->type() == DslAstValue::Type::Node && func->as_node()) {
            if (func->as_node()->kind() == "Name") {
                const DslAstValue* idv = func->as_node()->find_field("id");
                if (idv && idv->type() == DslAstValue::Type::String) {
                    callee = idv->as_string();
                }
            } else if (func->as_node()->kind() == "Attribute") {
                const DslAstValue* attr = func->as_node()->find_field("attr");
                if (attr && attr->type() == DslAstValue::Type::String) {
                    callee = attr->as_string();
                }
            }
        }
        std::vector<HostGlueIr::Ptr> args;
        const DslAstValue* argv = call.find_field("args");
        if (argv && argv->type() == DslAstValue::Type::Array) {
            for (const DslAstValue& a : argv->as_array()) {
                StatusOr<HostGlueIr::Ptr> e = lower_expr_value(&a, state, call);
                if (!e.ok()) {
                    return e.status();
                }
                args.push_back(std::move(e.value()));
            }
        }
        auto node = HostGlueIr::make_call(std::move(callee), std::move(args));
        node->set_location(call.lineno(), call.col_offset());
        return node;
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_expr_value(
        const DslAstValue* value,
        State& state,
        const DslAstNode& loc) {
        if (!value) {
            return HostGlueIr::make_const_int(0);
        }
        if (value->type() == DslAstValue::Type::Node && value->as_node()) {
            return lower_expr(*value->as_node(), state);
        }
        if (value->type() == DslAstValue::Type::Int) {
            return HostGlueIr::make_const_int(value->as_int());
        }
        if (value->type() == DslAstValue::Type::Bool) {
            return HostGlueIr::make_const_int(value->as_bool() ? 1 : 0);
        }
        (void)loc;
        (void)state;
        return HostGlueIr::make_pass();
    }

    [[nodiscard]] static StatusOr<HostGlueIr::Ptr> lower_expr(
        const DslAstNode& node,
        State& state) {
        if (node.kind() == "Constant") {
            const DslAstValue* val = node.find_field("value");
            if (val && val->type() == DslAstValue::Type::Int) {
                return HostGlueIr::make_const_int(val->as_int());
            }
            if (val && val->type() == DslAstValue::Type::Bool) {
                return HostGlueIr::make_const_int(val->as_bool() ? 1 : 0);
            }
            return HostGlueIr::make_const_int(0);
        }
        if (node.kind() == "Name") {
            const DslAstValue* idv = node.find_field("id");
            if (idv && idv->type() == DslAstValue::Type::String) {
                return HostGlueIr::make_name(idv->as_string());
            }
            return fail(DslRuleId::E032_primitive_body, "Name.id missing", state, node);
        }
        if (node.kind() == "Attribute") {
            // self.param → Name(param) as host-known binding.
            const DslAstValue* attr = node.find_field("attr");
            if (attr && attr->type() == DslAstValue::Type::String) {
                return HostGlueIr::make_name(attr->as_string());
            }
            return HostGlueIr::make_pass();
        }
        if (node.kind() == "Call") {
            return lower_call(node, state);
        }
        if (node.kind() == "BinOp" || node.kind() == "UnaryOp" || node.kind() == "Compare" ||
            node.kind() == "BoolOp") {
            // Structural host predicate / arithmetic: keep as Call-shaped opaque
            // for prelude emit later; for bounds we only need Const/Name.
            return HostGlueIr::make_call(node.kind(), {});
        }
        (void)state;
        return HostGlueIr::make_pass();
    }
};

#endif // DSL_HOST_GLUE_HPP
