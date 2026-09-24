#include <catch2/catch_test_macros.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/dsl_optimize.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <string>

TEST_CASE("DslOptimize const_fold add/mul/atbash", "[dsl][optimize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::add(Z29Expr::mul(Z29Expr::constant(3).value(), Z29Expr::constant(5).value()),
                     Z29Expr::atbash(Z29Expr::constant(2).value()));
    // 3*5=15; atbash(2)=26; 15+26=12
    const StatusOr<Z29Expr::Ptr> folded = DslOptimize::const_fold(expr);
    REQUIRE(folded.ok());
    REQUIRE(folded.value()->kind() == Z29Expr::Kind::Const);
    REQUIRE(folded.value()->const_value() == 12);
}

TEST_CASE("DslOptimize const_fold inv of nonzero constant", "[dsl][optimize]") {
    const StatusOr<Z29Expr::Ptr> folded =
        DslOptimize::const_fold(Z29Expr::inv(Z29Expr::constant(3).value()));
    REQUIRE(folded.ok());
    REQUIRE(folded.value()->kind() == Z29Expr::Kind::Const);
    REQUIRE(folded.value()->const_value() == Z29::inv(Index29{3}).value());
}

TEST_CASE("DslOptimize const_fold inv(0) is E040", "[dsl][optimize]") {
    const StatusOr<Z29Expr::Ptr> folded =
        DslOptimize::const_fold(Z29Expr::inv(Z29Expr::constant(0).value()));
    REQUIRE_FALSE(folded.ok());
    REQUIRE(folded.status().message().find("E040") != std::string::npos);
}

TEST_CASE("DslOptimize const_fold identities", "[dsl][optimize]") {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    {
        const StatusOr<Z29Expr::Ptr> f =
            DslOptimize::const_fold(Z29Expr::add(x, Z29Expr::constant(0).value()));
        REQUIRE(f.ok());
        REQUIRE(f.value()->kind() == Z29Expr::Kind::Var);
        REQUIRE(f.value()->name() == "x");
    }
    {
        const StatusOr<Z29Expr::Ptr> f =
            DslOptimize::const_fold(Z29Expr::mul(x, Z29Expr::constant(1).value()));
        REQUIRE(f.ok());
        REQUIRE(f.value()->kind() == Z29Expr::Kind::Var);
    }
    {
        const StatusOr<Z29Expr::Ptr> f =
            DslOptimize::const_fold(Z29Expr::mul(x, Z29Expr::constant(0).value()));
        REQUIRE(f.ok());
        REQUIRE(f.value()->kind() == Z29Expr::Kind::Const);
        REQUIRE(f.value()->const_value() == 0);
    }
}

TEST_CASE("DslOptimize hoists inv of affine multiplier", "[dsl][optimize]") {
    // decrypt: inv(a) * (x - b)
    const Z29Expr::Ptr body = Z29Expr::mul(Z29Expr::inv(Z29Expr::var("a")),
                                           Z29Expr::sub(Z29Expr::var("x"), Z29Expr::var("b")));
    const StatusOr<DslOptimize::Result> opt = DslOptimize::optimize(body, "x");
    REQUIRE(opt.ok());
    REQUIRE(opt.value().inv_hoists() == 1);
    REQUIRE(opt.value().hoists().size() == 1);
    REQUIRE(opt.value().hoists()[0].name() == "__parcae_inv_0");
    REQUIRE(opt.value().hoists()[0].inv_arg()->kind() == Z29Expr::Kind::Var);
    REQUIRE(opt.value().hoists()[0].inv_arg()->name() == "a");

    // Body should use the hoist var instead of Inv.
    REQUIRE(opt.value().expr()->kind() == Z29Expr::Kind::Mul);
    REQUIRE(opt.value().expr()->left()->kind() == Z29Expr::Kind::Var);
    REQUIRE(opt.value().expr()->left()->name() == "__parcae_inv_0");

    // Eval parity vs original for a sample (bind hoist manually).
    Z29Expr::Env env{{"x", Index29{10}}, {"a", Index29{3}}, {"b", Index29{2}}};
    const Index29 expect = Z29::mul(Z29::inv(Index29{3}), Z29::sub(Index29{10}, Index29{2}));
    env.emplace("__parcae_inv_0", Z29::inv(Index29{3}));
    REQUIRE(opt.value().expr()->eval(env).value() == expect);
    REQUIRE_FALSE(DslOptimize::depends_on_var(*opt.value().hoists()[0].inv_arg(), "x"));
}

TEST_CASE("DslOptimize does not hoist inv of cipher", "[dsl][optimize]") {
    const Z29Expr::Ptr body = Z29Expr::inv(Z29Expr::var("x"));
    const StatusOr<DslOptimize::Result> opt = DslOptimize::optimize(body, "x");
    REQUIRE(opt.ok());
    REQUIRE(opt.value().inv_hoists() == 0);
    REQUIRE(opt.value().hoists().empty());
    REQUIRE(opt.value().expr()->kind() == Z29Expr::Kind::Inv);
}

TEST_CASE("DslOptimize folds const inv then no hoist needed", "[dsl][optimize]") {
    const Z29Expr::Ptr body =
        Z29Expr::mul(Z29Expr::inv(Z29Expr::constant(7).value()), Z29Expr::var("x"));
    const StatusOr<DslOptimize::Result> opt = DslOptimize::optimize(body, "x");
    REQUIRE(opt.ok());
    REQUIRE(opt.value().const_folds() >= 1);
    REQUIRE(opt.value().inv_hoists() == 0);
    REQUIRE(opt.value().expr()->kind() == Z29Expr::Kind::Mul);
    REQUIRE(opt.value().expr()->left()->kind() == Z29Expr::Kind::Const);
}

TEST_CASE("DslOptimize optimize_theory rewrites affine decrypt", "[dsl][optimize]") {
    const StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
    const StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "dsl_affine", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {a.value(), b.value()},
        Z29Expr::add(Z29Expr::mul(Z29Expr::var("a"), x), Z29Expr::var("b")),
        Z29Expr::mul(Z29Expr::inv(Z29Expr::var("a")), Z29Expr::sub(x, Z29Expr::var("b"))));
    REQUIRE(theory.ok());

    const StatusOr<DslOptimize::TheoryResult> opt =
        DslOptimize::optimize_theory(theory.value(), "x");
    REQUIRE(opt.ok());
    REQUIRE(opt.value().decrypt().inv_hoists() == 1);
    REQUIRE(opt.value().theory().decrypt_step()->left()->name() == "__parcae_inv_0");
}

TEST_CASE("DslOptimize depends_on_var", "[dsl][optimize]") {
    const Z29Expr::Ptr expr = Z29Expr::add(Z29Expr::var("x"), Z29Expr::inv(Z29Expr::var("a")));
    REQUIRE(DslOptimize::depends_on_var(*expr, "x"));
    REQUIRE(DslOptimize::depends_on_var(*expr, "a"));
    REQUIRE_FALSE(DslOptimize::depends_on_var(*expr, "b"));
}

TEST_CASE("DslOptimize folds Select with const cond (dead arm)", "[dsl][optimize][select]") {
    const Z29Expr::Ptr expr =
        Z29Expr::select(Z29Expr::constant(1).value(),
                        Z29Expr::add(Z29Expr::constant(2).value(), Z29Expr::constant(3).value()),
                        Z29Expr::constant(9).value());
    const StatusOr<Z29Expr::Ptr> folded = DslOptimize::const_fold(expr);
    REQUIRE(folded.ok());
    REQUIRE(folded.value()->kind() == Z29Expr::Kind::Const);
    REQUIRE(folded.value()->const_value() == 5);
}

TEST_CASE("DslOptimize folds Select false arm when cond is 0", "[dsl][optimize][select]") {
    const Z29Expr::Ptr expr =
        Z29Expr::select(Z29Expr::eq(Z29Expr::constant(1).value(), Z29Expr::constant(2).value()),
                        Z29Expr::constant(7).value(), Z29Expr::constant(13).value());
    const StatusOr<Z29Expr::Ptr> folded = DslOptimize::const_fold(expr);
    REQUIRE(folded.ok());
    REQUIRE(folded.value()->kind() == Z29Expr::Kind::Const);
    REQUIRE(folded.value()->const_value() == 13);
}

TEST_CASE("DslOptimize Select equal arms collapses", "[dsl][optimize][select]") {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr expr =
        Z29Expr::select(x, Z29Expr::constant(4).value(), Z29Expr::constant(4).value());
    const StatusOr<Z29Expr::Ptr> folded = DslOptimize::const_fold(expr);
    REQUIRE(folded.ok());
    REQUIRE(folded.value()->kind() == Z29Expr::Kind::Const);
    REQUIRE(folded.value()->const_value() == 4);
}

TEST_CASE("DslOptimize keeps Select when cond is var", "[dsl][optimize][select]") {
    const Z29Expr::Ptr expr =
        Z29Expr::select(Z29Expr::var("flag"), Z29Expr::var("x"), Z29Expr::constant(0).value());
    const StatusOr<Z29Expr::Ptr> folded = DslOptimize::const_fold(expr);
    REQUIRE(folded.ok());
    REQUIRE(folded.value()->kind() == Z29Expr::Kind::Select);
    REQUIRE(DslOptimize::depends_on_var(*folded.value(), "flag"));
    REQUIRE(DslOptimize::depends_on_var(*folded.value(), "x"));
}

TEST_CASE("DslOptimize hoists inv under Select arms", "[dsl][optimize][select]") {
    const Z29Expr::Ptr body = Z29Expr::select(
        Z29Expr::var("flag"), Z29Expr::mul(Z29Expr::inv(Z29Expr::var("a")), Z29Expr::var("x")),
        Z29Expr::var("x"));
    const StatusOr<DslOptimize::Result> opt = DslOptimize::optimize(body, "x");
    REQUIRE(opt.ok());
    REQUIRE(opt.value().inv_hoists() == 1);
    REQUIRE(opt.value().expr()->kind() == Z29Expr::Kind::Select);
    REQUIRE(opt.value().expr()->if_true()->kind() == Z29Expr::Kind::Mul);
    REQUIRE(opt.value().expr()->if_true()->left()->name() == "__parcae_inv_0");
}
