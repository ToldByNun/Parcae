#include <catch2/catch_test_macros.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/dsl/z29_expr_normalize.hpp>

#include <cstdint>
#include <string>
#include <vector>

class Z29ExprNormalizeTestUtil {
public:
    [[nodiscard]] static Z29Expr::Ptr lit(int v) { return Z29Expr::constant(v).value(); }

    [[nodiscard]] static Z29Expr::Ptr x() { return Z29Expr::var("x"); }

    [[nodiscard]] static Z29Expr::Ptr v(const char* name) { return Z29Expr::var(name); }

    static void require_eval_parity(const Z29Expr::Ptr& original, const Z29Expr::Ptr& normalized,
                                    const std::vector<Z29Expr::Env>& envs) {
        REQUIRE(original);
        REQUIRE(normalized);
        for (const Z29Expr::Env& env : envs) {
            const StatusOr<Index29> a = original->eval(env);
            const StatusOr<Index29> b = normalized->eval(env);
            REQUIRE(a.ok());
            REQUIRE(b.ok());
            REQUIRE(a.value() == b.value());
        }
    }

    [[nodiscard]] static std::vector<Z29Expr::Env> sample_x_envs() {
        std::vector<Z29Expr::Env> out;
        for (std::uint8_t i = 0; i < 29; ++i) {
            out.push_back(Z29Expr::Env{{"x", Index29{i}}});
        }
        return out;
    }

    [[nodiscard]] static std::vector<Z29Expr::Env> sample_affine_envs() {
        std::vector<Z29Expr::Env> out;
        const std::uint8_t xs[] = {0, 1, 7, 14, 28};
        const std::uint8_t as[] = {1, 3, 10, 28};
        const std::uint8_t bs[] = {0, 5, 28};
        for (std::uint8_t xv : xs) {
            for (std::uint8_t a : as) {
                for (std::uint8_t b : bs) {
                    out.push_back(
                        Z29Expr::Env{{"x", Index29{xv}}, {"a", Index29{a}}, {"b", Index29{b}}});
                }
            }
        }
        return out;
    }

    [[nodiscard]] static std::vector<Z29Expr::Env> sample_linear_envs() {
        std::vector<Z29Expr::Env> out;
        const std::uint8_t xs[] = {0, 3, 28};
        const std::uint8_t is[] = {0, 1, 28};
        const std::uint8_t b0s[] = {0, 4};
        const std::uint8_t b1s[] = {1, 7};
        for (std::uint8_t xv : xs) {
            for (std::uint8_t iv : is) {
                for (std::uint8_t b0 : b0s) {
                    for (std::uint8_t b1 : b1s) {
                        out.push_back(Z29Expr::Env{{"x", Index29{xv}},
                                                   {"i", Index29{iv}},
                                                   {"b0", Index29{b0}},
                                                   {"b1", Index29{b1}}});
                    }
                }
            }
        }
        return out;
    }

private:
    Z29ExprNormalizeTestUtil() = delete;
};

TEST_CASE("Z29ExprNormalize atbash-as-arith Sub(28,x)", "[dsl][normalize]") {
    const Z29Expr::Ptr expr = Z29Expr::sub(Z29ExprNormalizeTestUtil::lit(28), Z29ExprNormalizeTestUtil::x());
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Atbash);
    REQUIRE(n.value()->arg()->kind() == Z29Expr::Kind::Var);
    REQUIRE(n.value()->arg()->name() == "x");
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(),
                                                  Z29ExprNormalizeTestUtil::sample_x_envs());
}

TEST_CASE("Z29ExprNormalize atbash-as-arith Add(28, Neg(x))", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::add(Z29ExprNormalizeTestUtil::lit(28), Z29Expr::neg(Z29ExprNormalizeTestUtil::x()));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Atbash);
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(),
                                                  Z29ExprNormalizeTestUtil::sample_x_envs());
}

TEST_CASE("Z29ExprNormalize atbash-as-arith Add(Neg(x), 28)", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::add(Z29Expr::neg(Z29ExprNormalizeTestUtil::x()), Z29ExprNormalizeTestUtil::lit(28));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Atbash);
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(),
                                                  Z29ExprNormalizeTestUtil::sample_x_envs());
}

TEST_CASE("Z29ExprNormalize BitNot becomes Atbash", "[dsl][normalize]") {
    const Z29Expr::Ptr expr = Z29Expr::bit_not(Z29ExprNormalizeTestUtil::x());
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Atbash);
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(),
                                                  Z29ExprNormalizeTestUtil::sample_x_envs());
}

TEST_CASE("Z29ExprNormalize z29_atbash Call becomes Atbash", "[dsl][normalize]") {
    const Z29Expr::Ptr expr = Z29Expr::call("z29_atbash", {Z29ExprNormalizeTestUtil::x()});
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Atbash);
    for (const Z29Expr::Env& env : Z29ExprNormalizeTestUtil::sample_x_envs()) {
        REQUIRE(n.value()->eval(env).value() == Z29::atbash(env.at("x")));
    }
}

TEST_CASE("Z29ExprNormalize Atbash node is stable", "[dsl][normalize]") {
    const Z29Expr::Ptr expr = Z29Expr::atbash(Z29ExprNormalizeTestUtil::x());
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Atbash);
    REQUIRE(Z29ExprNormalize::structurally_equal(*expr, *n.value()));
}

TEST_CASE("Z29ExprNormalize Add commute puts cipher on left", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::add(Z29ExprNormalizeTestUtil::v("shift"), Z29ExprNormalizeTestUtil::x());
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Add);
    REQUIRE(n.value()->left()->kind() == Z29Expr::Kind::Var);
    REQUIRE(n.value()->left()->name() == "x");
    REQUIRE(n.value()->right()->name() == "shift");
    std::vector<Z29Expr::Env> envs;
    const std::uint8_t xs[] = {0, 5, 28};
    const std::uint8_t ss[] = {0, 3, 10};
    for (std::uint8_t xv : xs) {
        for (std::uint8_t s : ss) {
            envs.push_back(Z29Expr::Env{{"x", Index29{xv}}, {"shift", Index29{s}}});
        }
    }
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(), envs);
}

TEST_CASE("Z29ExprNormalize Caesar Sub becomes Add Neg", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::sub(Z29ExprNormalizeTestUtil::x(), Z29ExprNormalizeTestUtil::v("shift"));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Add);
    REQUIRE(n.value()->left()->name() == "x");
    REQUIRE(n.value()->right());
    std::vector<Z29Expr::Env> envs;
    const std::uint8_t xs[] = {0, 9, 28};
    const std::uint8_t ss[] = {0, 1, 28};
    for (std::uint8_t xv : xs) {
        for (std::uint8_t s : ss) {
            envs.push_back(Z29Expr::Env{{"x", Index29{xv}}, {"shift", Index29{s}}});
        }
    }
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(), envs);
}

TEST_CASE("Z29ExprNormalize Mul commute prefers coeff left cipher right", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::mul(Z29ExprNormalizeTestUtil::x(), Z29ExprNormalizeTestUtil::v("a"));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Mul);
    REQUIRE(n.value()->left()->name() == "a");
    REQUIRE(n.value()->right()->name() == "x");
    std::vector<Z29Expr::Env> envs;
    const std::uint8_t xs[] = {0, 4, 28};
    const std::uint8_t as[] = {1, 3, 10};
    for (std::uint8_t xv : xs) {
        for (std::uint8_t a : as) {
            envs.push_back(Z29Expr::Env{{"x", Index29{xv}}, {"a", Index29{a}}});
        }
    }
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(), envs);
}

TEST_CASE("Z29ExprNormalize affine Add(b, Mul(a,x)) orders Mul then b", "[dsl][normalize]") {
    const Z29Expr::Ptr expr = Z29Expr::add(
        Z29ExprNormalizeTestUtil::v("b"),
        Z29Expr::mul(Z29ExprNormalizeTestUtil::x(), Z29ExprNormalizeTestUtil::v("a")));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Add);
    REQUIRE(n.value()->left()->kind() == Z29Expr::Kind::Mul);
    REQUIRE(n.value()->left()->left()->name() == "a");
    REQUIRE(n.value()->left()->right()->name() == "x");
    REQUIRE(n.value()->right()->name() == "b");
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(),
                                                  Z29ExprNormalizeTestUtil::sample_affine_envs());
}

TEST_CASE("Z29ExprNormalize linear keystream operand order", "[dsl][normalize]") {
    const Z29Expr::Ptr ks = Z29Expr::add(
        Z29Expr::mul(Z29ExprNormalizeTestUtil::v("i"), Z29ExprNormalizeTestUtil::v("b1")),
        Z29ExprNormalizeTestUtil::v("b0"));
    const Z29Expr::Ptr expr = Z29Expr::add(ks, Z29ExprNormalizeTestUtil::x());
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Add);
    REQUIRE(n.value()->left()->name() == "x");
    REQUIRE(n.value()->right()->kind() == Z29Expr::Kind::Add);
    REQUIRE(n.value()->right()->left()->name() == "b0");
    REQUIRE(n.value()->right()->right()->kind() == Z29Expr::Kind::Mul);
    REQUIRE(n.value()->right()->right()->right()->name() == "i");
    Z29ExprNormalizeTestUtil::require_eval_parity(expr, n.value(),
                                                  Z29ExprNormalizeTestUtil::sample_linear_envs());
}

TEST_CASE("Z29ExprNormalize idempotent on atbash-as-arith", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::sub(Z29ExprNormalizeTestUtil::lit(28), Z29ExprNormalizeTestUtil::x());
    const StatusOr<Z29Expr::Ptr> n1 = Z29ExprNormalize::normalize(expr);
    REQUIRE(n1.ok());
    const StatusOr<Z29Expr::Ptr> n2 = Z29ExprNormalize::normalize(n1.value());
    REQUIRE(n2.ok());
    REQUIRE(Z29ExprNormalize::structurally_equal(*n1.value(), *n2.value()));
}

TEST_CASE("Z29ExprNormalize idempotent on affine commute", "[dsl][normalize]") {
    const Z29Expr::Ptr expr = Z29Expr::add(
        Z29ExprNormalizeTestUtil::v("b"),
        Z29Expr::mul(Z29ExprNormalizeTestUtil::x(), Z29ExprNormalizeTestUtil::v("a")));
    const StatusOr<Z29Expr::Ptr> n1 = Z29ExprNormalize::normalize(expr);
    REQUIRE(n1.ok());
    const StatusOr<Z29Expr::Ptr> n2 = Z29ExprNormalize::normalize(n1.value());
    REQUIRE(n2.ok());
    REQUIRE(Z29ExprNormalize::structurally_equal(*n1.value(), *n2.value()));
}

TEST_CASE("Z29ExprNormalize does not rewrite z29_autokey_shift Call", "[dsl][normalize]") {
    const Z29Expr::Ptr lag =
        Z29Expr::add(Z29ExprNormalizeTestUtil::v("L"), Z29ExprNormalizeTestUtil::lit(0));
    const Z29Expr::Ptr expr = Z29Expr::sub(
        Z29ExprNormalizeTestUtil::x(),
        Z29Expr::call("z29_autokey_shift", {Z29ExprNormalizeTestUtil::x(), lag}));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    // Both sides depend on cipher → Caesar Sub→Add does not apply (fail closed).
    // Call node stays; lag Add(L,0) const-folds to L.
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Sub);
    REQUIRE(n.value()->right()->kind() == Z29Expr::Kind::Call);
    REQUIRE(n.value()->right()->name() == "z29_autokey_shift");
    REQUIRE(n.value()->right()->args().size() == 2);
    REQUIRE(n.value()->right()->args()[1]->kind() == Z29Expr::Kind::Var);
    REQUIRE(n.value()->right()->args()[1]->name() == "L");
}

TEST_CASE("Z29ExprNormalize prefer_branch Select keeps flag; normalizes arms", "[dsl][normalize]") {
    const Z29Expr::Ptr arm =
        Z29Expr::sub(Z29ExprNormalizeTestUtil::lit(28), Z29ExprNormalizeTestUtil::x());
    const Z29Expr::Ptr expr = Z29Expr::select(Z29ExprNormalizeTestUtil::v("c"), arm,
                                              Z29ExprNormalizeTestUtil::x(),
                                              /*prefer_branch=*/true);
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Select);
    REQUIRE(n.value()->prefer_branch());
    REQUIRE(n.value()->if_true()->kind() == Z29Expr::Kind::Atbash);
}

TEST_CASE("Z29ExprNormalize const_fold still applies", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::add(Z29ExprNormalizeTestUtil::lit(3), Z29ExprNormalizeTestUtil::lit(5));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr);
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Const);
    REQUIRE(n.value()->const_value() == 8);
}

TEST_CASE("Z29ExprNormalize null / empty cipher_var errors", "[dsl][normalize]") {
    REQUIRE_FALSE(Z29ExprNormalize::normalize(nullptr).ok());
    REQUIRE_FALSE(Z29ExprNormalize::normalize(Z29ExprNormalizeTestUtil::x(), "").ok());
}

TEST_CASE("Z29ExprNormalize name-irrelevant: unrelated var names still atbash", "[dsl][normalize]") {
    const Z29Expr::Ptr expr =
        Z29Expr::sub(Z29ExprNormalizeTestUtil::lit(28), Z29ExprNormalizeTestUtil::v("cipher"));
    const StatusOr<Z29Expr::Ptr> n = Z29ExprNormalize::normalize(expr, "cipher");
    REQUIRE(n.ok());
    REQUIRE(n.value()->kind() == Z29Expr::Kind::Atbash);
    REQUIRE(n.value()->arg()->name() == "cipher");
}
