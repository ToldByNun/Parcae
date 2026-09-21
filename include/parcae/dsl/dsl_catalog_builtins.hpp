#ifndef DSL_CATALOG_BUILTINS_HPP
#define DSL_CATALOG_BUILTINS_HPP

#include "parcae/core/status_or.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <vector>

/// Frozen catalog TheoryIr twins for `@ComposedTheory` steps (`atbash`, `caesar`, …).
/// Used by `DslCompile` / `DslFuse` when a compose references catalog transform ids.
class DslCatalogBuiltins {
public:
    [[nodiscard]] static TheoryIr identity() {
        const Z29Expr::Ptr x = Z29Expr::var("x");
        StatusOr<TheoryIr> th = TheoryIr::make(
            "identity",
            TheoryIr::Family::Elementwise,
            TheoryIr::Tier::A,
            TheoryIr::InterruptMode::ElementwiseDefault,
            {},
            x,
            x);
        return th.value();
    }

    [[nodiscard]] static TheoryIr atbash() {
        const Z29Expr::Ptr x = Z29Expr::var("x");
        StatusOr<TheoryIr> th = TheoryIr::make(
            "atbash",
            TheoryIr::Family::Elementwise,
            TheoryIr::Tier::A,
            TheoryIr::InterruptMode::ElementwiseDefault,
            {},
            Z29Expr::atbash(x),
            Z29Expr::atbash(x));
        return th.value();
    }

    [[nodiscard]] static TheoryIr caesar() {
        StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr s = Z29Expr::var("shift");
        StatusOr<TheoryIr> th = TheoryIr::make(
            "caesar",
            TheoryIr::Family::Elementwise,
            TheoryIr::Tier::A,
            TheoryIr::InterruptMode::ElementwiseDefault,
            {shift.value()},
            Z29Expr::add(x, s),
            Z29Expr::sub(x, s));
        return th.value();
    }

    [[nodiscard]] static TheoryIr affine() {
        StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
        StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr av = Z29Expr::var("a");
        const Z29Expr::Ptr bv = Z29Expr::var("b");
        // encrypt: a*x + b ; decrypt: inv(a)*(x - b)
        StatusOr<TheoryIr> th = TheoryIr::make(
            "affine",
            TheoryIr::Family::Elementwise,
            TheoryIr::Tier::A,
            TheoryIr::InterruptMode::ElementwiseDefault,
            {a.value(), b.value()},
            Z29Expr::add(Z29Expr::mul(av, x), bv),
            Z29Expr::mul(Z29Expr::inv(av), Z29Expr::sub(x, bv)));
        return th.value();
    }

    /// Catalog theories eligible as `@ComposedTheory` step ids.
    [[nodiscard]] static std::vector<TheoryIr> all() {
        return {identity(), atbash(), caesar(), affine()};
    }

private:
    DslCatalogBuiltins() = delete;
};

#endif // DSL_CATALOG_BUILTINS_HPP
