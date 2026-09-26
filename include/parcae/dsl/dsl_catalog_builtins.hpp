#ifndef DSL_CATALOG_BUILTINS_HPP
#define DSL_CATALOG_BUILTINS_HPP

#include "parcae/core/status_or.hpp"
#include "parcae/dsl/matrix_ir.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <optional>
#include <string_view>
#include <vector>

/// Frozen catalog TheoryIr twins for `@ComposedTheory` steps (`atbash`, `caesar`, …).
/// Used by `DslCompile` / `DslFuse` when a compose references catalog transform ids
/// or DSL-only elementwise twins (`matrix_mix`).
class DslCatalogBuiltins {
public:
    [[nodiscard]] static TheoryIr identity() {
        const Z29Expr::Ptr x = Z29Expr::var("x");
        StatusOr<TheoryIr> th =
            TheoryIr::make("identity", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                           TheoryIr::InterruptMode::ElementwiseDefault, {}, x, x);
        return th.value();
    }

    [[nodiscard]] static TheoryIr atbash() {
        const Z29Expr::Ptr x = Z29Expr::var("x");
        StatusOr<TheoryIr> th =
            TheoryIr::make("atbash", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                           TheoryIr::InterruptMode::ElementwiseDefault, {}, Z29Expr::atbash(x),
                           Z29Expr::atbash(x));
        return th.value();
    }

    [[nodiscard]] static TheoryIr caesar() {
        StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr s = Z29Expr::var("shift");
        StatusOr<TheoryIr> th =
            TheoryIr::make("caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                           TheoryIr::InterruptMode::ElementwiseDefault, {shift.value()},
                           Z29Expr::add(x, s), Z29Expr::sub(x, s));
        return th.value();
    }

    [[nodiscard]] static TheoryIr affine() {
        StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
        StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr av = Z29Expr::var("a");
        const Z29Expr::Ptr bv = Z29Expr::var("b");
        // encrypt: a*x + b ; decrypt: inv(a)*(x - b)
        StatusOr<TheoryIr> th =
            TheoryIr::make("affine", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                           TheoryIr::InterruptMode::ElementwiseDefault, {a.value(), b.value()},
                           Z29Expr::add(Z29Expr::mul(av, x), bv),
                           Z29Expr::mul(Z29Expr::inv(av), Z29Expr::sub(x, bv)));
        return th.value();
    }

    /// Elementwise key-mix via `MatrixIr::det_expr` (2×2). DSL-only compose leaf:
    /// fuse inlines; staged ComposeTransform fallback is unavailable (no TransformId).
    [[nodiscard]] static TheoryIr matrix_mix() {
        StatusOr<ParamIr> a = ParamIr::make("a", 0, 28);
        StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
        StatusOr<ParamIr> c = ParamIr::make("c", 0, 28);
        StatusOr<ParamIr> d = ParamIr::make("d", 0, 28);
        const Z29Expr::Ptr x = Z29Expr::var("x");
        StatusOr<MatrixIr> m = MatrixIr::make({Z29Expr::var("a"), Z29Expr::var("b"),
                                               Z29Expr::var("c"), Z29Expr::var("d")});
        const Z29Expr::Ptr det = m.value().det_expr();
        StatusOr<TheoryIr> th =
            TheoryIr::make("matrix_mix", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                           TheoryIr::InterruptMode::ElementwiseDefault,
                           {a.value(), b.value(), c.value(), d.value()}, Z29Expr::add(x, det),
                           Z29Expr::sub(x, det));
        return th.value();
    }

    /// Primer-less CTAK-style lag key via `z29_autokey_shift` Call (emit → AutokeyRing*).
    /// DSL-only compose leaf (fused emit only).
    [[nodiscard]] static TheoryIr autokey_lag() {
        StatusOr<ParamIr> lag = ParamIr::make("lag", 1, 28);
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr L = Z29Expr::var("lag");
        const Z29Expr::Ptr key = Z29Expr::call("z29_autokey_shift", {x, L});
        StatusOr<TheoryIr> th =
            TheoryIr::make("autokey_lag", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                           TheoryIr::InterruptMode::ElementwiseDefault, {lag.value()},
                           Z29Expr::add(x, key), Z29Expr::sub(x, key));
        return th.value();
    }

    /// Catalog theories eligible as `@ComposedTheory` step ids (incl. DSL-only leaves).
    [[nodiscard]] static std::vector<TheoryIr> all() {
        return {identity(), atbash(), caesar(), affine(), matrix_mix(), autokey_lag()};
    }

    /// Resolve a catalog step id, or nullopt if unknown.
    [[nodiscard]] static std::optional<TheoryIr> lookup(std::string_view name) {
        if (name == "identity") {
            return identity();
        }
        if (name == "atbash") {
            return atbash();
        }
        if (name == "caesar") {
            return caesar();
        }
        if (name == "affine") {
            return affine();
        }
        if (name == "matrix_mix") {
            return matrix_mix();
        }
        if (name == "autokey_lag") {
            return autokey_lag();
        }
        return std::nullopt;
    }

    /// True when `step` is a ComposeTransform / ComposeDriver staged-fallback id.
    [[nodiscard]] static bool is_staged_catalog_id(std::string_view step) noexcept {
        return step == "identity" || step == "atbash" || step == "caesar" || step == "affine" ||
               step == "vigenere_key" || step == "beaufort_key" || step == "totient_prime_stream";
    }

private:
    DslCatalogBuiltins() = delete;
};

#endif // DSL_CATALOG_BUILTINS_HPP
