#ifndef DSL_BUILD_IR_HPP
#define DSL_BUILD_IR_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/primitive_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

/// Lower a gated `DslAstDocument` into PrimitiveIr / TheoryIr (docs/spec/dsl.md).
/// Supports `@define_primitive` bodies and `@Theory` classes with
/// `encrypt_step` / `decrypt_step` (and optional `keystream_at` inlining).
class DslBuildIr {
public:
    class Unit {
    public:
        Unit() = default;

        [[nodiscard]] const std::vector<PrimitiveIr>& primitives() const noexcept {
            return primitives_;
        }

        [[nodiscard]] const std::vector<TheoryIr>& theories() const noexcept {
            return theories_;
        }

        [[nodiscard]] const std::string& source_path() const noexcept {
            return source_path_;
        }

        [[nodiscard]] const std::string& source_sha256() const noexcept {
            return source_sha256_;
        }

    private:
        friend class DslBuildIr;
        std::vector<PrimitiveIr> primitives_;
        std::vector<TheoryIr> theories_;
        std::string source_path_;
        std::string source_sha256_;
    };

    [[nodiscard]] static StatusOr<Unit> build(const DslAstDocument& doc) {
        if (!doc.module()) {
            return fail(DslRuleId::E031_forbidden_construct, "document has no module", doc.source_path());
        }
        Builder b;
        b.source_path = doc.source_path();
        b.source_sha256 = doc.source_sha256();
        Status st = b.walk_module(*doc.module());
        if (!st.ok()) {
            return st;
        }
        if (b.unit.theories_.empty()) {
            return fail(
                DslRuleId::E032_primitive_body,
                "module must define at least one @Theory",
                doc.source_path());
        }
        b.unit.source_path_ = doc.source_path();
        b.unit.source_sha256_ = doc.source_sha256();
        return std::move(b.unit);
    }

private:
    struct MethodBody {
        std::vector<std::string> arg_names;  // excluding self
        const DslAstNode* return_expr = nullptr;
        std::optional<int> lineno;
        std::optional<int> col;
    };

    struct TheoryDraft {
        std::string name;
        TheoryIr::Family family = TheoryIr::Family::Elementwise;
        TheoryIr::Tier tier = TheoryIr::Tier::A;
        TheoryIr::InterruptMode interrupts = TheoryIr::InterruptMode::ElementwiseDefault;
        std::vector<ParamIr> params;
        std::unordered_map<std::string, MethodBody> methods;
        std::optional<std::string> structural_claim;
        std::optional<int> lineno;
        std::optional<int> col;
    };

    struct Builder {
        Unit unit;
        std::string source_path;
        std::string source_sha256;
        std::unordered_map<std::string, PrimitiveIr> primitives_by_name;

        [[nodiscard]] Status walk_module(const DslAstNode& module) {
            const DslAstValue* body = module.find_field("body");
            if (!body || body->type() != DslAstValue::Type::Array) {
                return fail(DslRuleId::E031_forbidden_construct, "Module.body missing", source_path);
            }
            for (const DslAstValue& item : body->as_array()) {
                if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                    continue;
                }
                const DslAstNode& stmt = *item.as_node();
                if (stmt.kind() == "FunctionDef") {
                    Status st = maybe_primitive(stmt);
                    if (!st.ok()) {
                        return st;
                    }
                } else if (stmt.kind() == "ClassDef") {
                    Status st = maybe_theory(stmt);
                    if (!st.ok()) {
                        return st;
                    }
                }
            }
            return Status::success();
        }

        [[nodiscard]] Status maybe_primitive(const DslAstNode& fn) {
            StatusOr<const DslAstNode*> deco = find_decorator_call(fn, "define_primitive");
            if (!deco.ok()) {
                return deco.status();
            }
            if (!deco.value()) {
                return Status::success();  // plain function — ignore
            }
            StatusOr<std::string> name = keyword_string(*deco.value(), "name");
            if (!name.ok()) {
                return name.status();
            }
            StatusOr<std::string> signature = keyword_string(*deco.value(), "signature");
            if (!signature.ok()) {
                return signature.status();
            }
            StatusOr<std::vector<std::string>> args = function_arg_names(fn, /*skip_self=*/false);
            if (!args.ok()) {
                return args.status();
            }
            StatusOr<const DslAstNode*> ret = single_return_expr(fn);
            if (!ret.ok()) {
                return ret.status();
            }
            std::unordered_map<std::string, Z29Expr::Ptr> locals;
            for (const std::string& a : args.value()) {
                locals.emplace(a, Z29Expr::var(a));
            }
            StatusOr<Z29Expr::Ptr> body = lower_expr(*ret.value(), locals, /*theory=*/nullptr);
            if (!body.ok()) {
                return body.status();
            }
            StatusOr<PrimitiveIr> prim = PrimitiveIr::make(
                name.value(),
                signature.value(),
                std::move(body.value()),
                source_path,
                fn.lineno(),
                fn.col_offset());
            if (!prim.ok()) {
                return prim.status();
            }
            if (primitives_by_name.count(prim.value().name()) != 0) {
                return fail(
                    DslRuleId::E032_primitive_body,
                    "duplicate primitive '" + prim.value().name() + "'",
                    source_path,
                    fn.lineno(),
                    fn.col_offset());
            }
            primitives_by_name.emplace(prim.value().name(), prim.value());
            unit.primitives_.push_back(std::move(prim.value()));
            return Status::success();
        }

        [[nodiscard]] Status maybe_theory(const DslAstNode& cls) {
            StatusOr<const DslAstNode*> composed = find_decorator_call(cls, "ComposedTheory");
            if (!composed.ok()) {
                return composed.status();
            }
            if (composed.value()) {
                return fail(
                    DslRuleId::E032_primitive_body,
                    "@ComposedTheory lowering not supported in this compile slice",
                    source_path,
                    cls.lineno(),
                    cls.col_offset());
            }
            StatusOr<const DslAstNode*> deco = find_decorator_call(cls, "Theory");
            if (!deco.ok()) {
                return deco.status();
            }
            if (!deco.value()) {
                return Status::success();
            }

            TheoryDraft draft;
            draft.lineno = cls.lineno();
            draft.col = cls.col_offset();
            StatusOr<std::string> name = keyword_string(*deco.value(), "name");
            if (!name.ok()) {
                return name.status();
            }
            draft.name = std::move(name.value());
            StatusOr<std::string> family_s = keyword_string(*deco.value(), "family");
            if (!family_s.ok()) {
                return family_s.status();
            }
            StatusOr<TheoryIr::Family> family = TheoryIr::parse_family(family_s.value());
            if (!family.ok()) {
                return family.status();
            }
            draft.family = family.value();
            StatusOr<std::string> tier_s = keyword_string(*deco.value(), "tier");
            if (!tier_s.ok()) {
                return tier_s.status();
            }
            StatusOr<TheoryIr::Tier> tier = TheoryIr::parse_tier(tier_s.value());
            if (!tier.ok()) {
                return tier.status();
            }
            draft.tier = tier.value();

            StatusOr<std::optional<std::string>> interrupts_s =
                optional_keyword_string(*deco.value(), "interrupts");
            if (!interrupts_s.ok()) {
                return interrupts_s.status();
            }
            if (interrupts_s.value().has_value()) {
                if (*interrupts_s.value() == "none_by_design") {
                    draft.interrupts = TheoryIr::InterruptMode::NoneByDesign;
                } else if (*interrupts_s.value() == "policy_method") {
                    draft.interrupts = TheoryIr::InterruptMode::PolicyMethod;
                } else if (*interrupts_s.value() == "elementwise_default") {
                    draft.interrupts = TheoryIr::InterruptMode::ElementwiseDefault;
                } else {
                    return fail(
                        DslRuleId::E030_interrupt_policy,
                        "unknown interrupts='" + *interrupts_s.value() + "'",
                        source_path,
                        cls.lineno(),
                        cls.col_offset());
                }
            }

            const DslAstValue* body = cls.find_field("body");
            if (!body || body->type() != DslAstValue::Type::Array) {
                return fail(DslRuleId::E032_primitive_body, "ClassDef.body missing", source_path);
            }
            for (const DslAstValue& item : body->as_array()) {
                if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                    continue;
                }
                const DslAstNode& member = *item.as_node();
                if (member.kind() == "AnnAssign") {
                    Status st = parse_param(member, draft);
                    if (!st.ok()) {
                        return st;
                    }
                } else if (member.kind() == "FunctionDef") {
                    Status st = parse_method(member, draft);
                    if (!st.ok()) {
                        return st;
                    }
                }
            }

            auto enc_it = draft.methods.find("encrypt_step");
            auto dec_it = draft.methods.find("decrypt_step");
            if (enc_it == draft.methods.end() || dec_it == draft.methods.end()) {
                return fail(
                    DslRuleId::E032_primitive_body,
                    "theory '" + draft.name + "' requires encrypt_step and decrypt_step",
                    source_path,
                    draft.lineno,
                    draft.col);
            }

            StatusOr<Z29Expr::Ptr> enc = lower_method(enc_it->second, draft);
            if (!enc.ok()) {
                return enc.status();
            }
            StatusOr<Z29Expr::Ptr> dec = lower_method(dec_it->second, draft);
            if (!dec.ok()) {
                return dec.status();
            }

            StatusOr<TheoryIr> theory = TheoryIr::make(
                draft.name,
                draft.family,
                draft.tier,
                draft.interrupts,
                draft.params,
                std::move(enc.value()),
                std::move(dec.value()),
                draft.structural_claim,
                source_path,
                draft.lineno,
                draft.col);
            if (!theory.ok()) {
                return theory.status();
            }
            unit.theories_.push_back(std::move(theory.value()));
            return Status::success();
        }

        [[nodiscard]] Status parse_param(const DslAstNode& ann, TheoryDraft& draft) {
            const DslAstValue* target = ann.find_field("target");
            if (!target || target->type() != DslAstValue::Type::Node || !target->as_node() ||
                target->as_node()->kind() != "Name") {
                return fail(
                    DslRuleId::E040_param_domain,
                    "AnnAssign target must be a Name",
                    source_path,
                    ann.lineno(),
                    ann.col_offset());
            }
            const DslAstValue* idv = target->as_node()->find_field("id");
            if (!idv || idv->type() != DslAstValue::Type::String) {
                return fail(DslRuleId::E040_param_domain, "param Name.id missing", source_path);
            }
            const std::string pname = idv->as_string();

            const DslAstValue* value = ann.find_field("value");
            if (!value || value->type() != DslAstValue::Type::Node || !value->as_node() ||
                value->as_node()->kind() != "Call") {
                return fail(
                    DslRuleId::E040_param_domain,
                    "param '" + pname + "' must be assigned Param(min=…, max=…)",
                    source_path,
                    ann.lineno(),
                    ann.col_offset());
            }
            const DslAstNode& call = *value->as_node();
            StatusOr<std::string> callee = call_name(call);
            if (!callee.ok() || callee.value() != "Param") {
                return fail(
                    DslRuleId::E040_param_domain,
                    "param '" + pname + "' must call Param(...)",
                    source_path,
                    ann.lineno(),
                    ann.col_offset());
            }
            StatusOr<std::int64_t> min_v = keyword_int(call, "min");
            if (!min_v.ok()) {
                return min_v.status();
            }
            StatusOr<std::int64_t> max_v = keyword_int(call, "max");
            if (!max_v.ok()) {
                return max_v.status();
            }
            StatusOr<ParamIr> param =
                ParamIr::make(pname, min_v.value(), max_v.value(), source_path, ann.lineno(), ann.col_offset());
            if (!param.ok()) {
                return param.status();
            }
            draft.params.push_back(std::move(param.value()));
            return Status::success();
        }

        [[nodiscard]] Status parse_method(const DslAstNode& fn, TheoryDraft& draft) {
            const DslAstValue* name_v = fn.find_field("name");
            if (!name_v || name_v->type() != DslAstValue::Type::String) {
                return fail(DslRuleId::E032_primitive_body, "FunctionDef.name missing", source_path);
            }
            const std::string mname = name_v->as_string();
            if (mname == "structural_claim") {
                StatusOr<const DslAstNode*> ret = single_return_expr(fn);
                if (!ret.ok()) {
                    return ret.status();
                }
                if (ret.value()->kind() != "Constant") {
                    return fail(
                        DslRuleId::E013_tier_structural_claim,
                        "structural_claim must return a string constant",
                        source_path,
                        fn.lineno(),
                        fn.col_offset());
                }
                const DslAstValue* val = ret.value()->find_field("value");
                if (!val || val->type() != DslAstValue::Type::String) {
                    return fail(
                        DslRuleId::E013_tier_structural_claim,
                        "structural_claim must return a string",
                        source_path,
                        fn.lineno(),
                        fn.col_offset());
                }
                draft.structural_claim = val->as_string();
                return Status::success();
            }
            if (mname == "interrupt_policy") {
                draft.interrupts = TheoryIr::InterruptMode::PolicyMethod;
                return Status::success();  // predicate IR later
            }

            StatusOr<std::vector<std::string>> args = function_arg_names(fn, /*skip_self=*/true);
            if (!args.ok()) {
                return args.status();
            }
            StatusOr<const DslAstNode*> ret = single_return_expr(fn);
            if (!ret.ok()) {
                return ret.status();
            }
            MethodBody body;
            body.arg_names = std::move(args.value());
            body.return_expr = ret.value();
            body.lineno = fn.lineno();
            body.col = fn.col_offset();
            draft.methods[mname] = std::move(body);
            return Status::success();
        }

        [[nodiscard]] StatusOr<Z29Expr::Ptr> lower_method(
            const MethodBody& method,
            const TheoryDraft& draft) {
            std::unordered_map<std::string, Z29Expr::Ptr> locals;
            for (const std::string& a : method.arg_names) {
                locals.emplace(a, Z29Expr::var(a));
            }
            for (const ParamIr& p : draft.params) {
                locals.emplace(p.name(), Z29Expr::var(p.name()));
            }
            return lower_expr(*method.return_expr, locals, &draft);
        }

        [[nodiscard]] StatusOr<Z29Expr::Ptr> lower_expr(
            const DslAstNode& node,
            std::unordered_map<std::string, Z29Expr::Ptr>& locals,
            const TheoryDraft* theory) {
            if (node.kind() == "Constant") {
                const DslAstValue* val = node.find_field("value");
                if (!val || val->type() != DslAstValue::Type::Int) {
                    return fail(
                        DslRuleId::E032_primitive_body,
                        "only integer constants are allowed in Z29Expr",
                        source_path,
                        node.lineno(),
                        node.col_offset());
                }
                return Z29Expr::constant(val->as_int());
            }
            if (node.kind() == "Name") {
                const DslAstValue* idv = node.find_field("id");
                if (!idv || idv->type() != DslAstValue::Type::String) {
                    return fail(DslRuleId::E032_primitive_body, "Name.id missing", source_path);
                }
                const std::string& id = idv->as_string();
                auto it = locals.find(id);
                if (it != locals.end()) {
                    return it->second;
                }
                return fail(
                    DslRuleId::E032_primitive_body,
                    "unknown name '" + id + "' in expression",
                    source_path,
                    node.lineno(),
                    node.col_offset());
            }
            if (node.kind() == "Attribute") {
                return lower_attribute(node, locals, theory);
            }
            if (node.kind() == "BinOp") {
                const DslAstValue* opv = node.find_field("op");
                const DslAstValue* left = node.find_field("left");
                const DslAstValue* right = node.find_field("right");
                if (!opv || opv->type() != DslAstValue::Type::String || !left ||
                    left->type() != DslAstValue::Type::Node || !left->as_node() || !right ||
                    right->type() != DslAstValue::Type::Node || !right->as_node()) {
                    return fail(DslRuleId::E032_primitive_body, "malformed BinOp", source_path);
                }
                StatusOr<Z29Expr::Ptr> l = lower_expr(*left->as_node(), locals, theory);
                if (!l.ok()) {
                    return l.status();
                }
                StatusOr<Z29Expr::Ptr> r = lower_expr(*right->as_node(), locals, theory);
                if (!r.ok()) {
                    return r.status();
                }
                const std::string& op = opv->as_string();
                if (op == "Add") {
                    return Z29Expr::add(std::move(l.value()), std::move(r.value()));
                }
                if (op == "Sub") {
                    return Z29Expr::sub(std::move(l.value()), std::move(r.value()));
                }
                if (op == "Mult") {
                    return Z29Expr::mul(std::move(l.value()), std::move(r.value()));
                }
                return fail(
                    DslRuleId::E032_primitive_body,
                    "unsupported BinOp '" + op + "'",
                    source_path,
                    node.lineno(),
                    node.col_offset());
            }
            if (node.kind() == "UnaryOp") {
                const DslAstValue* opv = node.find_field("op");
                const DslAstValue* operand = node.find_field("operand");
                if (!opv || opv->type() != DslAstValue::Type::String || !operand ||
                    operand->type() != DslAstValue::Type::Node || !operand->as_node()) {
                    return fail(DslRuleId::E032_primitive_body, "malformed UnaryOp", source_path);
                }
                StatusOr<Z29Expr::Ptr> arg = lower_expr(*operand->as_node(), locals, theory);
                if (!arg.ok()) {
                    return arg.status();
                }
                if (opv->as_string() == "USub") {
                    return Z29Expr::neg(std::move(arg.value()));
                }
                if (opv->as_string() == "UAdd") {
                    return arg;
                }
                return fail(
                    DslRuleId::E032_primitive_body,
                    "unsupported UnaryOp '" + opv->as_string() + "'",
                    source_path,
                    node.lineno(),
                    node.col_offset());
            }
            if (node.kind() == "Call") {
                return lower_call(node, locals, theory);
            }
            return fail(
                DslRuleId::E032_primitive_body,
                "unsupported expression kind '" + node.kind() + "'",
                source_path,
                node.lineno(),
                node.col_offset());
        }

        [[nodiscard]] StatusOr<Z29Expr::Ptr> lower_attribute(
            const DslAstNode& node,
            std::unordered_map<std::string, Z29Expr::Ptr>& locals,
            const TheoryDraft* theory) {
            const DslAstValue* value = node.find_field("value");
            const DslAstValue* attr = node.find_field("attr");
            if (!value || value->type() != DslAstValue::Type::Node || !value->as_node() || !attr ||
                attr->type() != DslAstValue::Type::String) {
                return fail(DslRuleId::E032_primitive_body, "malformed Attribute", source_path);
            }
            if (value->as_node()->kind() == "Name") {
                const DslAstValue* idv = value->as_node()->find_field("id");
                if (idv && idv->type() == DslAstValue::Type::String && idv->as_string() == "self") {
                    const std::string& field = attr->as_string();
                    auto it = locals.find(field);
                    if (it != locals.end()) {
                        return it->second;
                    }
                    if (theory) {
                        for (const ParamIr& p : theory->params) {
                            if (p.name() == field) {
                                return Z29Expr::var(field);
                            }
                        }
                    }
                    return fail(
                        DslRuleId::E032_primitive_body,
                        "unknown self." + field,
                        source_path,
                        node.lineno(),
                        node.col_offset());
                }
            }
            return fail(
                DslRuleId::E032_primitive_body,
                "only self.<param> attributes are supported",
                source_path,
                node.lineno(),
                node.col_offset());
        }

        [[nodiscard]] StatusOr<Z29Expr::Ptr> lower_call(
            const DslAstNode& node,
            std::unordered_map<std::string, Z29Expr::Ptr>& locals,
            const TheoryDraft* theory) {
            const DslAstValue* func = node.find_field("func");
            const DslAstValue* args_v = node.find_field("args");
            if (!func || func->type() != DslAstValue::Type::Node || !func->as_node() || !args_v ||
                args_v->type() != DslAstValue::Type::Array) {
                return fail(DslRuleId::E032_primitive_body, "malformed Call", source_path);
            }
            std::vector<Z29Expr::Ptr> args;
            for (const DslAstValue& a : args_v->as_array()) {
                if (a.type() != DslAstValue::Type::Node || !a.as_node()) {
                    return fail(DslRuleId::E032_primitive_body, "Call arg must be expression", source_path);
                }
                StatusOr<Z29Expr::Ptr> e = lower_expr(*a.as_node(), locals, theory);
                if (!e.ok()) {
                    return e.status();
                }
                args.push_back(std::move(e.value()));
            }

            const DslAstNode& f = *func->as_node();
            if (f.kind() == "Name") {
                const DslAstValue* idv = f.find_field("id");
                if (!idv || idv->type() != DslAstValue::Type::String) {
                    return fail(DslRuleId::E032_primitive_body, "Call Name.id missing", source_path);
                }
                const std::string& id = idv->as_string();
                if (id == "z29_add" && args.size() == 2) {
                    return Z29Expr::add(std::move(args[0]), std::move(args[1]));
                }
                if (id == "z29_sub" && args.size() == 2) {
                    return Z29Expr::sub(std::move(args[0]), std::move(args[1]));
                }
                if (id == "z29_mul" && args.size() == 2) {
                    return Z29Expr::mul(std::move(args[0]), std::move(args[1]));
                }
                if (id == "z29_inv" && args.size() == 1) {
                    return Z29Expr::inv(std::move(args[0]));
                }
                if (id == "z29_mod" && args.size() == 2) {
                    return Z29Expr::mod(std::move(args[0]), std::move(args[1]));
                }
                auto pit = primitives_by_name.find(id);
                if (pit != primitives_by_name.end()) {
                    // Inline primitive body with param binding.
                    const PrimitiveIr& prim = pit->second;
                    if (args.size() != prim.arity()) {
                        return fail(
                            DslRuleId::E032_primitive_body,
                            "primitive '" + id + "' arity mismatch",
                            source_path,
                            node.lineno(),
                            node.col_offset());
                    }
                    return substitute_vars(prim.body(), prim.param_names(), args);
                }
                return fail(
                    DslRuleId::E032_primitive_body,
                    "unknown call '" + id + "'",
                    source_path,
                    node.lineno(),
                    node.col_offset());
            }
            if (f.kind() == "Attribute" && theory) {
                const DslAstValue* value = f.find_field("value");
                const DslAstValue* attr = f.find_field("attr");
                if (value && value->type() == DslAstValue::Type::Node && value->as_node() &&
                    value->as_node()->kind() == "Name") {
                    const DslAstValue* idv = value->as_node()->find_field("id");
                    if (idv && idv->type() == DslAstValue::Type::String && idv->as_string() == "self" &&
                        attr && attr->type() == DslAstValue::Type::String) {
                        const std::string& mname = attr->as_string();
                        auto mit = theory->methods.find(mname);
                        if (mit == theory->methods.end()) {
                            return fail(
                                DslRuleId::E032_primitive_body,
                                "unknown method self." + mname + "(...)",
                                source_path,
                                node.lineno(),
                                node.col_offset());
                        }
                        const MethodBody& mb = mit->second;
                        if (args.size() != mb.arg_names.size()) {
                            return fail(
                                DslRuleId::E032_primitive_body,
                                "method '" + mname + "' arity mismatch",
                                source_path,
                                node.lineno(),
                                node.col_offset());
                        }
                        std::unordered_map<std::string, Z29Expr::Ptr> bound = locals;
                        for (std::size_t i = 0; i < mb.arg_names.size(); ++i) {
                            bound[mb.arg_names[i]] = args[i];
                        }
                        for (const ParamIr& p : theory->params) {
                            bound.emplace(p.name(), Z29Expr::var(p.name()));
                        }
                        return lower_expr(*mb.return_expr, bound, theory);
                    }
                }
            }
            return fail(
                DslRuleId::E032_primitive_body,
                "unsupported Call target",
                source_path,
                node.lineno(),
                node.col_offset());
        }

        [[nodiscard]] static StatusOr<Z29Expr::Ptr> substitute_vars(
            const Z29Expr::Ptr& body,
            const std::vector<std::string>& names,
            const std::vector<Z29Expr::Ptr>& args) {
            if (!body) {
                return Status::error("null primitive body");
            }
            std::unordered_map<std::string, Z29Expr::Ptr> env;
            for (std::size_t i = 0; i < names.size(); ++i) {
                env.emplace(names[i], args[i]);
            }
            return subst_rec(body, env);
        }

        [[nodiscard]] static StatusOr<Z29Expr::Ptr> subst_rec(
            const Z29Expr::Ptr& node,
            const std::unordered_map<std::string, Z29Expr::Ptr>& env) {
            if (!node) {
                return Status::error("null expr in substitute");
            }
            switch (node->kind()) {
            case Z29Expr::Kind::Const:
                return node;
            case Z29Expr::Kind::Var: {
                auto it = env.find(node->name());
                if (it != env.end()) {
                    return it->second;
                }
                return node;
            }
            case Z29Expr::Kind::Add:
            case Z29Expr::Kind::Sub:
            case Z29Expr::Kind::Mul:
            case Z29Expr::Kind::Mod: {
                StatusOr<Z29Expr::Ptr> l = subst_rec(node->left(), env);
                if (!l.ok()) {
                    return l.status();
                }
                StatusOr<Z29Expr::Ptr> r = subst_rec(node->right(), env);
                if (!r.ok()) {
                    return r.status();
                }
                if (node->kind() == Z29Expr::Kind::Add) {
                    return Z29Expr::add(std::move(l.value()), std::move(r.value()));
                }
                if (node->kind() == Z29Expr::Kind::Sub) {
                    return Z29Expr::sub(std::move(l.value()), std::move(r.value()));
                }
                if (node->kind() == Z29Expr::Kind::Mul) {
                    return Z29Expr::mul(std::move(l.value()), std::move(r.value()));
                }
                return Z29Expr::mod(std::move(l.value()), std::move(r.value()));
            }
            case Z29Expr::Kind::Neg:
            case Z29Expr::Kind::Inv:
            case Z29Expr::Kind::Atbash: {
                StatusOr<Z29Expr::Ptr> a = subst_rec(node->arg(), env);
                if (!a.ok()) {
                    return a.status();
                }
                if (node->kind() == Z29Expr::Kind::Neg) {
                    return Z29Expr::neg(std::move(a.value()));
                }
                if (node->kind() == Z29Expr::Kind::Inv) {
                    return Z29Expr::inv(std::move(a.value()));
                }
                return Z29Expr::atbash(std::move(a.value()));
            }
            case Z29Expr::Kind::Call: {
                std::vector<Z29Expr::Ptr> nargs;
                for (const Z29Expr::Ptr& a : node->args()) {
                    StatusOr<Z29Expr::Ptr> s = subst_rec(a, env);
                    if (!s.ok()) {
                        return s.status();
                    }
                    nargs.push_back(std::move(s.value()));
                }
                return Z29Expr::call(node->name(), std::move(nargs));
            }
            }
            return Status::error("unknown Z29Expr kind in substitute");
        }
    };

    DslBuildIr() = delete;

    [[nodiscard]] static Status fail(
        std::string_view rule,
        std::string message,
        const std::string& path,
        std::optional<int> lineno = std::nullopt,
        std::optional<int> col = std::nullopt) {
        return DslDiag::make(rule, std::move(message), path, lineno, col).to_status();
    }

    [[nodiscard]] static StatusOr<const DslAstNode*> find_decorator_call(
        const DslAstNode& defn,
        std::string_view deco_name) {
        const DslAstValue* list = defn.find_field("decorator_list");
        if (!list || list->type() != DslAstValue::Type::Array) {
            return static_cast<const DslAstNode*>(nullptr);
        }
        for (const DslAstValue& item : list->as_array()) {
            if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                continue;
            }
            const DslAstNode& deco = *item.as_node();
            if (deco.kind() != "Call") {
                continue;
            }
            StatusOr<std::string> name = call_name(deco);
            if (!name.ok()) {
                return name.status();
            }
            if (name.value() == deco_name) {
                return &deco;
            }
        }
        return static_cast<const DslAstNode*>(nullptr);
    }

    [[nodiscard]] static StatusOr<std::string> call_name(const DslAstNode& call) {
        const DslAstValue* func = call.find_field("func");
        if (!func || func->type() != DslAstValue::Type::Node || !func->as_node()) {
            return Status::error("Call.func missing");
        }
        const DslAstNode& f = *func->as_node();
        if (f.kind() != "Name") {
            return Status::error("Call.func must be a Name for decorator/Param");
        }
        const DslAstValue* idv = f.find_field("id");
        if (!idv || idv->type() != DslAstValue::Type::String) {
            return Status::error("Call Name.id missing");
        }
        return idv->as_string();
    }

    [[nodiscard]] static StatusOr<std::string> keyword_string(
        const DslAstNode& call,
        std::string_view key) {
        StatusOr<std::optional<std::string>> opt = optional_keyword_string(call, key);
        if (!opt.ok()) {
            return opt.status();
        }
        if (!opt.value().has_value()) {
            return Status::error(std::string("missing keyword '") + std::string(key) + "'");
        }
        return *opt.value();
    }

    [[nodiscard]] static StatusOr<std::optional<std::string>> optional_keyword_string(
        const DslAstNode& call,
        std::string_view key) {
        const DslAstValue* kws = call.find_field("keywords");
        if (!kws || kws->type() != DslAstValue::Type::Array) {
            return std::optional<std::string>{};
        }
        for (const DslAstValue& item : kws->as_array()) {
            if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                continue;
            }
            const DslAstNode& kw = *item.as_node();
            const DslAstValue* arg = kw.find_field("arg");
            const DslAstValue* value = kw.find_field("value");
            if (!arg || arg->type() != DslAstValue::Type::String || arg->as_string() != key) {
                continue;
            }
            if (!value || value->type() != DslAstValue::Type::Node || !value->as_node() ||
                value->as_node()->kind() != "Constant") {
                return Status::error(std::string("keyword '") + std::string(key) + "' must be a Constant");
            }
            const DslAstValue* cv = value->as_node()->find_field("value");
            if (!cv || cv->type() != DslAstValue::Type::String) {
                return Status::error(std::string("keyword '") + std::string(key) + "' must be a string");
            }
            return std::optional<std::string>{cv->as_string()};
        }
        return std::optional<std::string>{};
    }

    [[nodiscard]] static StatusOr<std::int64_t> keyword_int(
        const DslAstNode& call,
        std::string_view key) {
        const DslAstValue* kws = call.find_field("keywords");
        if (!kws || kws->type() != DslAstValue::Type::Array) {
            return Status::error(std::string("missing keyword '") + std::string(key) + "'");
        }
        for (const DslAstValue& item : kws->as_array()) {
            if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                continue;
            }
            const DslAstNode& kw = *item.as_node();
            const DslAstValue* arg = kw.find_field("arg");
            const DslAstValue* value = kw.find_field("value");
            if (!arg || arg->type() != DslAstValue::Type::String || arg->as_string() != key) {
                continue;
            }
            if (!value || value->type() != DslAstValue::Type::Node || !value->as_node() ||
                value->as_node()->kind() != "Constant") {
                return Status::error(std::string("keyword '") + std::string(key) + "' must be int Constant");
            }
            const DslAstValue* cv = value->as_node()->find_field("value");
            if (!cv || cv->type() != DslAstValue::Type::Int) {
                return Status::error(std::string("keyword '") + std::string(key) + "' must be an int");
            }
            return cv->as_int();
        }
        return Status::error(std::string("missing keyword '") + std::string(key) + "'");
    }

    [[nodiscard]] static StatusOr<std::vector<std::string>> function_arg_names(
        const DslAstNode& fn,
        bool skip_self) {
        const DslAstValue* args = fn.find_field("args");
        if (!args || args->type() != DslAstValue::Type::Node || !args->as_node()) {
            return Status::error("FunctionDef.args missing");
        }
        const DslAstValue* list = args->as_node()->find_field("args");
        if (!list || list->type() != DslAstValue::Type::Array) {
            return Status::error("arguments.args missing");
        }
        std::vector<std::string> out;
        for (const DslAstValue& item : list->as_array()) {
            if (item.type() != DslAstValue::Type::Node || !item.as_node()) {
                continue;
            }
            const DslAstValue* arg = item.as_node()->find_field("arg");
            if (!arg || arg->type() != DslAstValue::Type::String) {
                return Status::error("arg.arg missing");
            }
            if (skip_self && arg->as_string() == "self") {
                continue;
            }
            out.push_back(arg->as_string());
        }
        return out;
    }

    [[nodiscard]] static StatusOr<const DslAstNode*> single_return_expr(const DslAstNode& fn) {
        const DslAstValue* body = fn.find_field("body");
        if (!body || body->type() != DslAstValue::Type::Array || body->as_array().size() != 1) {
            return Status::error("function body must be a single return");
        }
        const DslAstValue& only = body->as_array().front();
        if (only.type() != DslAstValue::Type::Node || !only.as_node() || only.as_node()->kind() != "Return") {
            return Status::error("function body must be a single return");
        }
        const DslAstValue* value = only.as_node()->find_field("value");
        if (!value || value->type() != DslAstValue::Type::Node || !value->as_node()) {
            return Status::error("return value missing");
        }
        return value->as_node().get();
    }
};

#endif // DSL_BUILD_IR_HPP
