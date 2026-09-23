#ifndef THEORY_APPLY_IR_HPP
#define THEORY_APPLY_IR_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Persist / load CPU-dispatch IR for a compiled theory (`apply_ir.json`).
/// Schema: `parcae.theory_apply_ir.v0` — encrypt/decrypt Z29Expr trees + params.
/// Used by TheoryDispatch (docs/spec/theory-artifact.md).
class TheoryApplyIr {
public:
    static constexpr std::string_view schema_id = "parcae.theory_apply_ir.v0";

    [[nodiscard]] static StatusOr<nlohmann::json> to_json(
        const TheoryIr& theory,
        std::string_view cipher_var = "x") {
        if (cipher_var.empty()) {
            return Status::error("TheoryApplyIr cipher_var must be non-empty");
        }
        StatusOr<nlohmann::json> enc = expr_to_json(theory.encrypt_step());
        if (!enc.ok()) {
            return enc.status();
        }
        StatusOr<nlohmann::json> dec = expr_to_json(theory.decrypt_step());
        if (!dec.ok()) {
            return dec.status();
        }
        nlohmann::json params = nlohmann::json::array();
        for (const ParamIr& p : theory.params()) {
            params.push_back(nlohmann::json{
                {"name", p.name()},
                {"min", static_cast<std::int64_t>(p.min())},
                {"max", static_cast<std::int64_t>(p.max())},
            });
        }
        nlohmann::json out{
            {"schema", std::string(schema_id)},
            {"name", theory.name()},
            {"family", TheoryIr::family_str(theory.family())},
            {"tier", TheoryIr::tier_str(theory.tier())},
            {"interrupts", interrupt_str(theory.interrupt_mode())},
            {"cipher_var", std::string(cipher_var)},
            {"params", std::move(params)},
            {"encrypt_step", std::move(enc.value())},
            {"decrypt_step", std::move(dec.value())},
        };
        if (theory.structural_claim().has_value()) {
            out["structural_claim"] = *theory.structural_claim();
        } else {
            out["structural_claim"] = nullptr;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<TheoryIr> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("TheoryApplyIr must be a JSON object");
        }
        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != schema_id) {
            return Status::error(
                "TheoryApplyIr.schema must be " + std::string(schema_id));
        }
        if (!root.contains("name") || !root.at("name").is_string()) {
            return Status::error("TheoryApplyIr.name is required");
        }
        if (!root.contains("family") || !root.at("family").is_string()) {
            return Status::error("TheoryApplyIr.family is required");
        }
        if (!root.contains("tier") || !root.at("tier").is_string()) {
            return Status::error("TheoryApplyIr.tier is required");
        }
        StatusOr<TheoryIr::Family> family =
            TheoryIr::parse_family(root.at("family").get<std::string>());
        if (!family.ok()) {
            return family.status();
        }
        StatusOr<TheoryIr::Tier> tier =
            TheoryIr::parse_tier(root.at("tier").get<std::string>());
        if (!tier.ok()) {
            return tier.status();
        }
        TheoryIr::InterruptMode interrupts = TheoryIr::InterruptMode::ElementwiseDefault;
        if (root.contains("interrupts")) {
            if (!root.at("interrupts").is_string()) {
                return Status::error("TheoryApplyIr.interrupts must be a string");
            }
            StatusOr<TheoryIr::InterruptMode> parsed =
                parse_interrupt(root.at("interrupts").get<std::string>());
            if (!parsed.ok()) {
                return parsed.status();
            }
            interrupts = parsed.value();
        }

        std::vector<ParamIr> params;
        if (root.contains("params")) {
            if (!root.at("params").is_array()) {
                return Status::error("TheoryApplyIr.params must be an array");
            }
            for (const nlohmann::json& row : root.at("params")) {
                if (!row.is_object() || !row.contains("name") || !row.at("name").is_string() ||
                    !row.contains("min") || !row.at("min").is_number_integer() ||
                    !row.contains("max") || !row.at("max").is_number_integer()) {
                    return Status::error("TheoryApplyIr.params entries must be {name,min,max}");
                }
                StatusOr<ParamIr> p = ParamIr::make(
                    row.at("name").get<std::string>(),
                    row.at("min").get<std::int64_t>(),
                    row.at("max").get<std::int64_t>());
                if (!p.ok()) {
                    return p.status();
                }
                params.push_back(std::move(p.value()));
            }
        }

        if (!root.contains("encrypt_step") || !root.contains("decrypt_step")) {
            return Status::error("TheoryApplyIr requires encrypt_step and decrypt_step");
        }
        StatusOr<Z29Expr::Ptr> enc = expr_from_json(root.at("encrypt_step"));
        if (!enc.ok()) {
            return enc.status();
        }
        StatusOr<Z29Expr::Ptr> dec = expr_from_json(root.at("decrypt_step"));
        if (!dec.ok()) {
            return dec.status();
        }

        std::optional<std::string> claim;
        if (root.contains("structural_claim") && !root.at("structural_claim").is_null()) {
            if (!root.at("structural_claim").is_string()) {
                return Status::error("TheoryApplyIr.structural_claim must be string or null");
            }
            claim = root.at("structural_claim").get<std::string>();
        }

        return TheoryIr::make(
            root.at("name").get<std::string>(),
            family.value(),
            tier.value(),
            interrupts,
            std::move(params),
            std::move(enc.value()),
            std::move(dec.value()),
            std::move(claim));
    }

    [[nodiscard]] static StatusOr<std::string> cipher_var_from_json(const nlohmann::json& root) {
        if (!root.contains("cipher_var") || !root.at("cipher_var").is_string()) {
            return std::string("x");
        }
        const std::string v = root.at("cipher_var").get<std::string>();
        if (v.empty()) {
            return Status::error("TheoryApplyIr.cipher_var must be non-empty");
        }
        return v;
    }

    [[nodiscard]] static Status write(
        const std::filesystem::path& path,
        const TheoryIr& theory,
        std::string_view cipher_var = "x") {
        StatusOr<nlohmann::json> json = to_json(theory, cipher_var);
        if (!json.ok()) {
            return json.status();
        }
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::error("failed to write apply_ir: " + path.string());
        }
        out << json.value().dump(2) << '\n';
        if (!out) {
            return Status::error("failed while writing apply_ir: " + path.string());
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<TheoryIr> load(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("failed to open apply_ir: " + path.string());
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        if (!in && !in.eof()) {
            return Status::error("failed while reading apply_ir: " + path.string());
        }
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(ss.str());
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid TheoryApplyIr JSON: ") + ex.what());
        }
        return from_json(root);
    }

    [[nodiscard]] static StatusOr<std::string> load_cipher_var(
        const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("failed to open apply_ir: " + path.string());
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(ss.str());
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid TheoryApplyIr JSON: ") + ex.what());
        }
        return cipher_var_from_json(root);
    }

private:
    TheoryApplyIr() = delete;

    [[nodiscard]] static constexpr std::string_view interrupt_str(
        TheoryIr::InterruptMode m) noexcept {
        switch (m) {
        case TheoryIr::InterruptMode::PolicyMethod:
            return "policy_method";
        case TheoryIr::InterruptMode::NoneByDesign:
            return "none_by_design";
        case TheoryIr::InterruptMode::ElementwiseDefault:
            return "elementwise_default";
        }
        return "elementwise_default";
    }

    [[nodiscard]] static StatusOr<TheoryIr::InterruptMode> parse_interrupt(
        std::string_view text) {
        if (text == "policy_method") {
            return TheoryIr::InterruptMode::PolicyMethod;
        }
        if (text == "none_by_design") {
            return TheoryIr::InterruptMode::NoneByDesign;
        }
        if (text == "elementwise_default") {
            return TheoryIr::InterruptMode::ElementwiseDefault;
        }
        return Status::error("unknown interrupts mode '" + std::string(text) + "'");
    }

    [[nodiscard]] static StatusOr<std::string> kind_str(Z29Expr::Kind k) {
        switch (k) {
        case Z29Expr::Kind::Const:
            return std::string("const");
        case Z29Expr::Kind::Var:
            return std::string("var");
        case Z29Expr::Kind::Add:
            return std::string("add");
        case Z29Expr::Kind::Sub:
            return std::string("sub");
        case Z29Expr::Kind::Mul:
            return std::string("mul");
        case Z29Expr::Kind::Div:
            return std::string("div");
        case Z29Expr::Kind::FloorDiv:
            return std::string("floordiv");
        case Z29Expr::Kind::Mod:
            return std::string("mod");
        case Z29Expr::Kind::Pow:
            return std::string("pow");
        case Z29Expr::Kind::Neg:
            return std::string("neg");
        case Z29Expr::Kind::Inv:
            return std::string("inv");
        case Z29Expr::Kind::Atbash:
            return std::string("atbash");
        case Z29Expr::Kind::BitAnd:
            return std::string("bitand");
        case Z29Expr::Kind::BitOr:
            return std::string("bitor");
        case Z29Expr::Kind::BitXor:
            return std::string("bitxor");
        case Z29Expr::Kind::BitNot:
            return std::string("bitnot");
        case Z29Expr::Kind::LShift:
            return std::string("lshift");
        case Z29Expr::Kind::RShift:
            return std::string("rshift");
        case Z29Expr::Kind::Eq:
            return std::string("eq");
        case Z29Expr::Kind::Ne:
            return std::string("ne");
        case Z29Expr::Kind::Lt:
            return std::string("lt");
        case Z29Expr::Kind::Le:
            return std::string("le");
        case Z29Expr::Kind::Gt:
            return std::string("gt");
        case Z29Expr::Kind::Ge:
            return std::string("ge");
        case Z29Expr::Kind::BoolAnd:
            return std::string("booland");
        case Z29Expr::Kind::BoolOr:
            return std::string("boolor");
        case Z29Expr::Kind::BoolNot:
            return std::string("boolnot");
        case Z29Expr::Kind::Select:
            return std::string("select");
        case Z29Expr::Kind::Call:
            return std::string("call");
        }
        return Status::error("unknown Z29Expr kind");
    }

    [[nodiscard]] static StatusOr<Z29Expr::Kind> parse_kind(std::string_view text) {
        if (text == "const") {
            return Z29Expr::Kind::Const;
        }
        if (text == "var") {
            return Z29Expr::Kind::Var;
        }
        if (text == "add") {
            return Z29Expr::Kind::Add;
        }
        if (text == "sub") {
            return Z29Expr::Kind::Sub;
        }
        if (text == "mul") {
            return Z29Expr::Kind::Mul;
        }
        if (text == "div") {
            return Z29Expr::Kind::Div;
        }
        if (text == "floordiv") {
            return Z29Expr::Kind::FloorDiv;
        }
        if (text == "mod") {
            return Z29Expr::Kind::Mod;
        }
        if (text == "pow") {
            return Z29Expr::Kind::Pow;
        }
        if (text == "neg") {
            return Z29Expr::Kind::Neg;
        }
        if (text == "inv") {
            return Z29Expr::Kind::Inv;
        }
        if (text == "atbash") {
            return Z29Expr::Kind::Atbash;
        }
        if (text == "bitand") {
            return Z29Expr::Kind::BitAnd;
        }
        if (text == "bitor") {
            return Z29Expr::Kind::BitOr;
        }
        if (text == "bitxor") {
            return Z29Expr::Kind::BitXor;
        }
        if (text == "bitnot") {
            return Z29Expr::Kind::BitNot;
        }
        if (text == "lshift") {
            return Z29Expr::Kind::LShift;
        }
        if (text == "rshift") {
            return Z29Expr::Kind::RShift;
        }
        if (text == "eq") {
            return Z29Expr::Kind::Eq;
        }
        if (text == "ne") {
            return Z29Expr::Kind::Ne;
        }
        if (text == "lt") {
            return Z29Expr::Kind::Lt;
        }
        if (text == "le") {
            return Z29Expr::Kind::Le;
        }
        if (text == "gt") {
            return Z29Expr::Kind::Gt;
        }
        if (text == "ge") {
            return Z29Expr::Kind::Ge;
        }
        if (text == "booland") {
            return Z29Expr::Kind::BoolAnd;
        }
        if (text == "boolor") {
            return Z29Expr::Kind::BoolOr;
        }
        if (text == "boolnot") {
            return Z29Expr::Kind::BoolNot;
        }
        if (text == "select") {
            return Z29Expr::Kind::Select;
        }
        if (text == "call") {
            return Z29Expr::Kind::Call;
        }
        return Status::error("unknown Z29Expr kind '" + std::string(text) + "'");
    }

    [[nodiscard]] static StatusOr<nlohmann::json> expr_to_json(const Z29Expr::Ptr& node) {
        if (!node) {
            return Status::error("null Z29Expr in TheoryApplyIr");
        }
        StatusOr<std::string> kind = kind_str(node->kind());
        if (!kind.ok()) {
            return kind.status();
        }
        if (node->kind() == Z29Expr::Kind::Const) {
            return nlohmann::json{{"kind", kind.value()}, {"value", node->const_value()}};
        }
        if (node->kind() == Z29Expr::Kind::Var) {
            return nlohmann::json{{"kind", kind.value()}, {"name", node->name()}};
        }
        if (node->kind() == Z29Expr::Kind::Call) {
            nlohmann::json args = nlohmann::json::array();
            for (const Z29Expr::Ptr& a : node->args()) {
                StatusOr<nlohmann::json> j = expr_to_json(a);
                if (!j.ok()) {
                    return j.status();
                }
                args.push_back(std::move(j.value()));
            }
            return nlohmann::json{
                {"kind", kind.value()}, {"name", node->name()}, {"args", std::move(args)}};
        }
        if (node->kind() == Z29Expr::Kind::Select) {
            StatusOr<nlohmann::json> c = expr_to_json(node->cond());
            if (!c.ok()) {
                return c.status();
            }
            StatusOr<nlohmann::json> t = expr_to_json(node->if_true());
            if (!t.ok()) {
                return t.status();
            }
            StatusOr<nlohmann::json> f = expr_to_json(node->if_false());
            if (!f.ok()) {
                return f.status();
            }
            return nlohmann::json{
                {"kind", kind.value()},
                {"cond", std::move(c.value())},
                {"if_true", std::move(t.value())},
                {"if_false", std::move(f.value())}};
        }
        if (Z29Expr::is_unary(node->kind())) {
            StatusOr<nlohmann::json> arg = expr_to_json(node->arg());
            if (!arg.ok()) {
                return arg.status();
            }
            return nlohmann::json{{"kind", kind.value()}, {"arg", std::move(arg.value())}};
        }
        if (Z29Expr::is_binary(node->kind())) {
            StatusOr<nlohmann::json> left = expr_to_json(node->left());
            if (!left.ok()) {
                return left.status();
            }
            StatusOr<nlohmann::json> right = expr_to_json(node->right());
            if (!right.ok()) {
                return right.status();
            }
            return nlohmann::json{
                {"kind", kind.value()},
                {"left", std::move(left.value())},
                {"right", std::move(right.value())}};
        }
        return Status::error("unsupported Z29Expr kind in TheoryApplyIr");
    }

    [[nodiscard]] static StatusOr<Z29Expr::Ptr> expr_from_json(const nlohmann::json& root) {
        if (!root.is_object() || !root.contains("kind") || !root.at("kind").is_string()) {
            return Status::error("Z29Expr JSON must be an object with kind");
        }
        StatusOr<Z29Expr::Kind> kind = parse_kind(root.at("kind").get<std::string>());
        if (!kind.ok()) {
            return kind.status();
        }
        if (kind.value() == Z29Expr::Kind::Const) {
            if (!root.contains("value") || !root.at("value").is_number_integer()) {
                return Status::error("const Z29Expr requires integer value");
            }
            const std::int64_t v = root.at("value").get<std::int64_t>();
            if (v < 0 || v >= 29) {
                return Status::error("const Z29Expr value out of 0..28");
            }
            return Z29Expr::constant(static_cast<std::uint8_t>(v));
        }
        if (kind.value() == Z29Expr::Kind::Var) {
            if (!root.contains("name") || !root.at("name").is_string()) {
                return Status::error("var Z29Expr requires name");
            }
            return Z29Expr::var(root.at("name").get<std::string>());
        }
        if (kind.value() == Z29Expr::Kind::Call) {
            if (!root.contains("name") || !root.at("name").is_string() ||
                !root.contains("args") || !root.at("args").is_array()) {
                return Status::error("call Z29Expr requires name and args");
            }
            std::vector<Z29Expr::Ptr> args;
            for (const nlohmann::json& a : root.at("args")) {
                StatusOr<Z29Expr::Ptr> p = expr_from_json(a);
                if (!p.ok()) {
                    return p.status();
                }
                args.push_back(std::move(p.value()));
            }
            return Z29Expr::call(root.at("name").get<std::string>(), std::move(args));
        }
        if (kind.value() == Z29Expr::Kind::Select) {
            if (!root.contains("cond") || !root.contains("if_true") || !root.contains("if_false")) {
                return Status::error("select Z29Expr requires cond, if_true, if_false");
            }
            StatusOr<Z29Expr::Ptr> c = expr_from_json(root.at("cond"));
            if (!c.ok()) {
                return c.status();
            }
            StatusOr<Z29Expr::Ptr> t = expr_from_json(root.at("if_true"));
            if (!t.ok()) {
                return t.status();
            }
            StatusOr<Z29Expr::Ptr> f = expr_from_json(root.at("if_false"));
            if (!f.ok()) {
                return f.status();
            }
            return Z29Expr::make_select(
                std::move(c.value()), std::move(t.value()), std::move(f.value()));
        }
        if (Z29Expr::is_unary(kind.value())) {
            if (!root.contains("arg")) {
                return Status::error("unary Z29Expr requires arg");
            }
            StatusOr<Z29Expr::Ptr> arg = expr_from_json(root.at("arg"));
            if (!arg.ok()) {
                return arg.status();
            }
            return Z29Expr::make_unary_kind(kind.value(), std::move(arg.value()));
        }
        if (Z29Expr::is_binary(kind.value())) {
            if (!root.contains("left") || !root.contains("right")) {
                return Status::error("binary Z29Expr requires left and right");
            }
            StatusOr<Z29Expr::Ptr> left = expr_from_json(root.at("left"));
            if (!left.ok()) {
                return left.status();
            }
            StatusOr<Z29Expr::Ptr> right = expr_from_json(root.at("right"));
            if (!right.ok()) {
                return right.status();
            }
            return Z29Expr::make_binary(
                kind.value(), std::move(left.value()), std::move(right.value()));
        }
        return Status::error("unsupported Z29Expr kind in TheoryApplyIr");
    }
};

#endif // THEORY_APPLY_IR_HPP
