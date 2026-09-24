#ifndef HOST_GLUE_IR_HPP
#define HOST_GLUE_IR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// OuterControl host-side IR (docs/spec/dsl.md § Execution scopes).
/// Distinct from `Z29Expr` (HotLoop math). Used for setup / step_params /
/// compile-time specialization — never inside the CUDA element loop.
class HostGlueIr {
public:
    enum class Kind : std::uint8_t {
        Assign = 0,
        If,
        ForRange,
        WhileBounded,
        Call,
        Seq,
        ConstInt,
        Name,
        Pass,
    };

    /// How a loop was accepted / will be emitted.
    enum class BoundKind : std::uint8_t {
        ConstUnroll = 0, // fully constant → unroll / static IR
        HostKnown,       // Param / launch-time host-known finite range
    };

    using Ptr = std::shared_ptr<HostGlueIr>;

    [[nodiscard]] static Ptr make_pass() {
        return std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::Pass));
    }

    [[nodiscard]] static Ptr make_const_int(std::int64_t value) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::ConstInt));
        n->int_value_ = value;
        return n;
    }

    [[nodiscard]] static Ptr make_name(std::string name) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::Name));
        n->name_ = std::move(name);
        return n;
    }

    [[nodiscard]] static Ptr make_assign(std::string target, Ptr value) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::Assign));
        n->name_ = std::move(target);
        n->children_.push_back(std::move(value));
        return n;
    }

    [[nodiscard]] static Ptr make_if(Ptr cond, Ptr then_body, Ptr else_body) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::If));
        n->children_.push_back(std::move(cond));
        n->children_.push_back(std::move(then_body));
        n->children_.push_back(std::move(else_body));
        return n;
    }

    [[nodiscard]] static Ptr make_for_range(std::string target, Ptr start, Ptr stop, Ptr step,
                                            Ptr body, BoundKind bound) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::ForRange));
        n->name_ = std::move(target);
        n->bound_kind_ = bound;
        n->children_.push_back(std::move(start));
        n->children_.push_back(std::move(stop));
        n->children_.push_back(std::move(step));
        n->children_.push_back(std::move(body));
        return n;
    }

    [[nodiscard]] static Ptr make_while_bounded(Ptr cond, Ptr body, std::int64_t max_iters,
                                                BoundKind bound) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::WhileBounded));
        n->bound_kind_ = bound;
        n->int_value_ = max_iters;
        n->children_.push_back(std::move(cond));
        n->children_.push_back(std::move(body));
        return n;
    }

    [[nodiscard]] static Ptr make_call(std::string callee, std::vector<Ptr> args) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::Call));
        n->name_ = std::move(callee);
        n->children_ = std::move(args);
        return n;
    }

    [[nodiscard]] static Ptr make_seq(std::vector<Ptr> stmts) {
        auto n = std::shared_ptr<HostGlueIr>(new HostGlueIr(Kind::Seq));
        n->children_ = std::move(stmts);
        return n;
    }

    [[nodiscard]] Kind kind() const noexcept { return kind_; }

    [[nodiscard]] BoundKind bound_kind() const noexcept { return bound_kind_; }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    [[nodiscard]] std::int64_t int_value() const noexcept { return int_value_; }

    [[nodiscard]] const std::vector<Ptr>& children() const noexcept { return children_; }

    [[nodiscard]] std::optional<int> lineno() const noexcept { return lineno_; }

    [[nodiscard]] std::optional<int> col_offset() const noexcept { return col_offset_; }

    void set_location(std::optional<int> lineno, std::optional<int> col) {
        lineno_ = lineno;
        col_offset_ = col;
    }

    [[nodiscard]] std::string_view kind_string() const noexcept {
        switch (kind_) {
        case Kind::Assign:
            return "Assign";
        case Kind::If:
            return "If";
        case Kind::ForRange:
            return "ForRange";
        case Kind::WhileBounded:
            return "WhileBounded";
        case Kind::Call:
            return "Call";
        case Kind::Seq:
            return "Seq";
        case Kind::ConstInt:
            return "ConstInt";
        case Kind::Name:
            return "Name";
        case Kind::Pass:
            return "Pass";
        }
        return "Pass";
    }

    [[nodiscard]] std::string_view bound_kind_string() const noexcept {
        switch (bound_kind_) {
        case BoundKind::ConstUnroll:
            return "ConstUnroll";
        case BoundKind::HostKnown:
            return "HostKnown";
        }
        return "ConstUnroll";
    }

private:
    explicit HostGlueIr(Kind kind) : kind_(kind) {}

    Kind kind_ = Kind::Pass;
    BoundKind bound_kind_ = BoundKind::ConstUnroll;
    std::string name_;
    std::int64_t int_value_ = 0;
    std::vector<Ptr> children_;
    std::optional<int> lineno_;
    std::optional<int> col_offset_;
};

#endif // HOST_GLUE_IR_HPP
