#ifndef DSL_EXEC_SCOPE_HPP
#define DSL_EXEC_SCOPE_HPP

#include <cstddef>
#include <string_view>

/// Execution scope for a DSL AST region (docs/spec/dsl.md § Execution scopes).
///
/// OuterControl = host / setup / compile-time configuration.
/// HotLoop = per-rune math that lowers to Z29Expr / CUDA element loops.
class DslExecScope {
public:
    enum class Kind {
        OuterControl = 0,
        HotLoop,
    };

    DslExecScope() noexcept : kind_(Kind::OuterControl), loop_depth_(0) {}

    explicit DslExecScope(Kind kind, std::size_t loop_depth = 0) noexcept
        : kind_(kind), loop_depth_(loop_depth) {}

    [[nodiscard]] Kind kind() const noexcept {
        return kind_;
    }

    [[nodiscard]] std::size_t loop_depth() const noexcept {
        return loop_depth_;
    }

    [[nodiscard]] bool is_outer_control() const noexcept {
        return kind_ == Kind::OuterControl;
    }

    [[nodiscard]] bool is_hot_loop() const noexcept {
        return kind_ == Kind::HotLoop;
    }

    [[nodiscard]] bool in_loop() const noexcept {
        return loop_depth_ > 0;
    }

    [[nodiscard]] DslExecScope with_kind(Kind kind) const noexcept {
        return DslExecScope{kind, loop_depth_};
    }

    /// Enter a `for` / `while` (depth += 1), keeping Outer vs HotLoop kind.
    [[nodiscard]] DslExecScope enter_loop() const noexcept {
        return DslExecScope{kind_, loop_depth_ + 1};
    }

    [[nodiscard]] std::string_view kind_string() const noexcept {
        switch (kind_) {
        case Kind::OuterControl:
            return "OuterControl";
        case Kind::HotLoop:
            return "HotLoop";
        }
        return "OuterControl";
    }

    [[nodiscard]] bool operator==(const DslExecScope& other) const noexcept {
        return kind_ == other.kind_ && loop_depth_ == other.loop_depth_;
    }

    [[nodiscard]] bool operator!=(const DslExecScope& other) const noexcept {
        return !(*this == other);
    }

private:
    Kind kind_ = Kind::OuterControl;
    std::size_t loop_depth_ = 0;
};

#endif // DSL_EXEC_SCOPE_HPP
