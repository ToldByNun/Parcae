#include <catch2/catch_test_macros.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_shape_match.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/dsl/z29_expr_normalize.hpp>

#include <cstdint>
#include <string>
#include <vector>

class TheoryShapeMatchTestUtil {
public:
    [[nodiscard]] static Z29Expr::Ptr lit(int v) { return Z29Expr::constant(v).value(); }

    [[nodiscard]] static Z29Expr::Ptr x() { return Z29Expr::var("x"); }

    [[nodiscard]] static Z29Expr::Ptr v(const char* name) { return Z29Expr::var(name); }

    [[nodiscard]] static TheoryIr make_theory(std::string name, Z29Expr::Ptr decrypt) {
        StatusOr<TheoryIr> t = TheoryIr::make(
            std::move(name), TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
            TheoryIr::InterruptMode::ElementwiseDefault, /*params=*/{},
            /*encrypt_step=*/Z29Expr::var("x"), std::move(decrypt));
        REQUIRE(t.ok());
        return t.value();
    }

private:
    TheoryShapeMatchTestUtil() = delete;
};

TEST_CASE("TheoryShapeMatch Atbash from Sub(28,x) after normalize", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::sub(TheoryShapeMatchTestUtil::lit(28), TheoryShapeMatchTestUtil::x());
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Atbash);
    REQUIRE(m.value().normalized()->kind() == Z29Expr::Kind::Atbash);
}

TEST_CASE("TheoryShapeMatch Atbash from z29_atbash Call", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::call("z29_atbash", {TheoryShapeMatchTestUtil::x()});
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Atbash);
}

TEST_CASE("TheoryShapeMatch Atbash from BitNot", "[dsl][shape]") {
    const StatusOr<TheoryShapeMatch::Match> m =
        TheoryShapeMatch::match(Z29Expr::bit_not(TheoryShapeMatchTestUtil::x()));
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Atbash);
}

TEST_CASE("TheoryShapeMatch Caesar Add(shift,x) name-irrelevant cipher order", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::add(TheoryShapeMatchTestUtil::v("shift"), TheoryShapeMatchTestUtil::x());
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Caesar);
    REQUIRE(m.value().shift_name() == "shift");
}

TEST_CASE("TheoryShapeMatch Caesar Sub(x,shift) via normalize", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::sub(TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("shift"));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Caesar);
    REQUIRE(m.value().shift_name() == "shift");
}

TEST_CASE("TheoryShapeMatch Affine Add(b, Mul(x,a))", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("b"),
        Z29Expr::mul(TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("a")));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Affine);
    REQUIRE(m.value().a_name() == "a");
    REQUIRE(m.value().b_name() == "b");
    REQUIRE_FALSE(m.value().affine_decrypt());
}

TEST_CASE("TheoryShapeMatch Affine decrypt Mul(Inv(a), Sub(x,b))", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::mul(
        Z29Expr::inv(TheoryShapeMatchTestUtil::v("a")),
        Z29Expr::sub(TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("b")));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Affine);
    REQUIRE(m.value().affine_decrypt());
    REQUIRE(m.value().a_name() == "a");
    REQUIRE(m.value().b_name() == "b");
}

TEST_CASE("TheoryShapeMatch Affine Mul(const,x) with invertible a", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::mul(TheoryShapeMatchTestUtil::lit(3), TheoryShapeMatchTestUtil::x());
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Affine);
    REQUIRE(m.value().has_const_a());
    REQUIRE(m.value().const_a() == 3);
    REQUIRE(m.value().has_const_b());
    REQUIRE(m.value().const_b() == 0);
}

TEST_CASE("TheoryShapeMatch non-inv affine Const(0)*x is not Affine", "[dsl][shape]") {
    // const_fold may collapse Mul(0,x)→0; either way must not report Affine.
    const Z29Expr::Ptr expr = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("b"),
        Z29Expr::mul(TheoryShapeMatchTestUtil::lit(0), TheoryShapeMatchTestUtil::x()));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() != TheoryShapeMatch::ShapeId::Affine);
}

TEST_CASE("TheoryShapeMatch LinearKeystream progressive form", "[dsl][shape]") {
    const Z29Expr::Ptr ks = Z29Expr::add(
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("i"), TheoryShapeMatchTestUtil::v("b1")),
        TheoryShapeMatchTestUtil::v("b0"));
    const Z29Expr::Ptr expr = Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::LinearKeystream);
    REQUIRE(m.value().b0_name() == "b0");
    REQUIRE(m.value().b1_name() == "b1");
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch LinearKeystream Add(ks,x) after commute", "[dsl][shape]") {
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("b0"),
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("b1"), TheoryShapeMatchTestUtil::v("i")));
    const Z29Expr::Ptr expr = Z29Expr::add(ks, TheoryShapeMatchTestUtil::x());
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::LinearKeystream);
    REQUIRE_FALSE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch LinearKeystream bare b1*i implies b0=0", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::sub(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("b1"), TheoryShapeMatchTestUtil::v("i")));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::LinearKeystream);
    REQUIRE(m.value().b1_name() == "b1");
    REQUIRE(m.value().b0_name().empty());
    REQUIRE(m.value().has_const_b0());
    REQUIRE(m.value().const_b0() == 0);
    REQUIRE(m.value().cipher_minus_ks());
    REQUIRE(m.value().linear_coeffs_ok());
}

TEST_CASE("TheoryShapeMatch LinearKeystream Mul(i,b1) commute bare term", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::add(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("i"), TheoryShapeMatchTestUtil::v("b1")));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::LinearKeystream);
    REQUIRE(m.value().b1_name() == "b1");
    REQUIRE(m.value().has_const_b0());
    REQUIRE_FALSE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch LinearKeystream const b0 Lit + b1*i", "[dsl][shape]") {
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::lit(5),
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("b1"), TheoryShapeMatchTestUtil::v("i")));
    const Z29Expr::Ptr expr = Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::LinearKeystream);
    REQUIRE(m.value().has_const_b0());
    REQUIRE(m.value().const_b0() == 5);
    REQUIRE(m.value().b1_name() == "b1");
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch LinearKeystream b0 + const*i", "[dsl][shape]") {
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("b0"),
        Z29Expr::mul(TheoryShapeMatchTestUtil::lit(3), TheoryShapeMatchTestUtil::v("i")));
    const Z29Expr::Ptr expr = Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::LinearKeystream);
    REQUIRE(m.value().b0_name() == "b0");
    REQUIRE(m.value().has_const_b1());
    REQUIRE(m.value().const_b1() == 3);
}

TEST_CASE("TheoryShapeMatch PolyKeystream quadratic bitmask_blend class", "[dsl][shape][s5]") {
    const Z29Expr::Ptr i = TheoryShapeMatchTestUtil::v("i");
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("b0"),
        Z29Expr::add(Z29Expr::mul(TheoryShapeMatchTestUtil::v("b1"), i),
                     Z29Expr::mul(TheoryShapeMatchTestUtil::v("b2"), Z29Expr::mul(i, i))));
    const Z29Expr::Ptr expr = Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(m.value().poly_coeffs_ok());
    REQUIRE(m.value().b0_name() == "b0");
    REQUIRE(m.value().b1_name() == "b1");
    REQUIRE(m.value().b2_name() == "b2");
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch PolyKeystream left-assoc Mul(Mul(b2,i),i)", "[dsl][shape][s5]") {
    const Z29Expr::Ptr i = TheoryShapeMatchTestUtil::v("i");
    const Z29Expr::Ptr b2ii =
        Z29Expr::mul(Z29Expr::mul(TheoryShapeMatchTestUtil::v("c2"), i), i);
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("c0"),
        Z29Expr::add(Z29Expr::mul(TheoryShapeMatchTestUtil::v("c1"), i), b2ii));
    const Z29Expr::Ptr expr = Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(m.value().b0_name() == "c0");
    REQUIRE(m.value().b1_name() == "c1");
    REQUIRE(m.value().b2_name() == "c2");
}

TEST_CASE("TheoryShapeMatch PolyKeystream name-irrelevant custom still poly",
          "[dsl][shape][s5]") {
    const Z29Expr::Ptr i = TheoryShapeMatchTestUtil::v("i");
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("alpha"),
        Z29Expr::add(
            Z29Expr::mul(TheoryShapeMatchTestUtil::v("beta"), i),
            Z29Expr::mul(TheoryShapeMatchTestUtil::v("gamma"), Z29Expr::mul(i, i))));
    const TheoryIr theory = TheoryShapeMatchTestUtil::make_theory(
        "bitmask_blend_custom", Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match_theory(theory);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(m.value().b0_name() == "alpha");
    REQUIRE(m.value().b1_name() == "beta");
    REQUIRE(m.value().b2_name() == "gamma");
}

TEST_CASE("TheoryShapeMatch PolyKeystream bare b2*i*i implies b0=b1=0", "[dsl][shape][s5]") {
    const Z29Expr::Ptr i = TheoryShapeMatchTestUtil::v("i");
    const Z29Expr::Ptr expr = Z29Expr::sub(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("b2"), Z29Expr::mul(i, i)));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(m.value().poly_coeffs_ok());
    REQUIRE(m.value().b2_name() == "b2");
    REQUIRE(m.value().has_const_b0());
    REQUIRE(m.value().const_b0() == 0);
    REQUIRE(m.value().has_const_b1());
    REQUIRE(m.value().const_b1() == 0);
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch PolyKeystream b0 + b2*i*i implies b1=0", "[dsl][shape][s5]") {
    const Z29Expr::Ptr i = TheoryShapeMatchTestUtil::v("i");
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("b0"),
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("b2"), Z29Expr::mul(i, i)));
    const Z29Expr::Ptr expr = Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(m.value().b0_name() == "b0");
    REQUIRE(m.value().b2_name() == "b2");
    REQUIRE(m.value().has_const_b1());
    REQUIRE(m.value().const_b1() == 0);
}

TEST_CASE("TheoryShapeMatch PolyKeystream const Lit coeffs", "[dsl][shape][s5]") {
    const Z29Expr::Ptr i = TheoryShapeMatchTestUtil::v("i");
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::lit(2),
        Z29Expr::add(Z29Expr::mul(TheoryShapeMatchTestUtil::lit(3), i),
                     Z29Expr::mul(TheoryShapeMatchTestUtil::lit(4), Z29Expr::mul(i, i))));
    const Z29Expr::Ptr expr = Z29Expr::sub(TheoryShapeMatchTestUtil::x(), ks);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(m.value().has_const_b0());
    REQUIRE(m.value().const_b0() == 2);
    REQUIRE(m.value().has_const_b1());
    REQUIRE(m.value().const_b1() == 3);
    REQUIRE(m.value().has_const_b2());
    REQUIRE(m.value().const_b2() == 4);
}

TEST_CASE("TheoryShapeMatch PolyKeystream Add+Neg decrypt form", "[dsl][shape][s5]") {
    const Z29Expr::Ptr i = TheoryShapeMatchTestUtil::v("i");
    const Z29Expr::Ptr ks = Z29Expr::add(
        TheoryShapeMatchTestUtil::v("b0"),
        Z29Expr::add(Z29Expr::mul(TheoryShapeMatchTestUtil::v("b1"), i),
                     Z29Expr::mul(TheoryShapeMatchTestUtil::v("b2"), Z29Expr::mul(i, i))));
    const Z29Expr::Ptr expr =
        Z29Expr::add(TheoryShapeMatchTestUtil::x(), Z29Expr::neg(ks));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(m.value().cipher_minus_ks());
    REQUIRE(m.value().b0_name() == "b0");
}

TEST_CASE("TheoryShapeMatch LinearKeystream bare i is b0=0 b1=1", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::sub(TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("i"));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::LinearKeystream);
    REQUIRE(m.value().has_const_b0());
    REQUIRE(m.value().const_b0() == 0);
    REQUIRE(m.value().has_const_b1());
    REQUIRE(m.value().const_b1() == 1);
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch LinearKeystream x+const stays Caesar not S2", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::add(TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::lit(7));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Caesar);
    REQUIRE(m.value().has_const_shift());
    REQUIRE(m.value().const_shift() == 7);
}

TEST_CASE("TheoryShapeMatch Affine decrypt Mul(Inv(a), x) is b=0", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::mul(Z29Expr::inv(TheoryShapeMatchTestUtil::v("a")), TheoryShapeMatchTestUtil::x());
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Affine);
    REQUIRE(m.value().affine_decrypt());
    REQUIRE(m.value().a_name() == "a");
    REQUIRE(m.value().has_const_b());
    REQUIRE(m.value().const_b() == 0);
}

TEST_CASE("TheoryShapeMatch FxOnly custom f(x)", "[dsl][shape]") {
    // x^2-style: Mul(x,x) — not Affine/Caesar.
    const Z29Expr::Ptr expr =
        Z29Expr::mul(TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::x());
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::FxOnly);
}

TEST_CASE("TheoryShapeMatch KeyedGeneral uses i non-linear", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::add(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::mul(TheoryShapeMatchTestUtil::v("i"), TheoryShapeMatchTestUtil::v("i")));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::KeyedGeneral);
}

TEST_CASE("TheoryShapeMatch Autokey vigenere_lag binds lag", "[dsl][shape][s4]") {
    const Z29Expr::Ptr expr = Z29Expr::sub(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::call("z29_autokey_shift",
                      {TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("L")}));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Autokey);
    REQUIRE(m.value().autokey_lag_ok());
    REQUIRE(m.value().lag_name() == "L");
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch Autokey Add+Neg decrypt form", "[dsl][shape][s4]") {
    const Z29Expr::Ptr prior = Z29Expr::call(
        "z29_autokey_shift",
        {TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("delay")});
    const Z29Expr::Ptr expr =
        Z29Expr::add(TheoryShapeMatchTestUtil::x(), Z29Expr::neg(prior));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Autokey);
    REQUIRE(m.value().autokey_lag_ok());
    REQUIRE(m.value().lag_name() == "delay");
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch Autokey Add encrypt form (cipher + shift)", "[dsl][shape][s4]") {
    const Z29Expr::Ptr prior = Z29Expr::call(
        "z29_autokey_shift",
        {TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("lag")});
    const Z29Expr::Ptr expr = Z29Expr::add(TheoryShapeMatchTestUtil::x(), prior);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Autokey);
    REQUIRE(m.value().autokey_lag_ok());
    REQUIRE_FALSE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch Autokey const lag binds", "[dsl][shape][s4]") {
    const Z29Expr::Ptr expr = Z29Expr::sub(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::call("z29_autokey_shift",
                      {TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::lit(3)}));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Autokey);
    REQUIRE(m.value().autokey_lag_ok());
    REQUIRE(m.value().has_const_lag());
    REQUIRE(m.value().const_lag() == 3);
    REQUIRE(m.value().lag_name().empty());
    REQUIRE(m.value().cipher_minus_ks());
}

TEST_CASE("TheoryShapeMatch Autokey lag depending on i is unbound Autokey",
          "[dsl][shape][s4]") {
    // Still Autokey shape (contains Call), but no lag bind → emit soft S0.
    const Z29Expr::Ptr expr = Z29Expr::sub(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::call("z29_autokey_shift",
                      {TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("i")}));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Autokey);
    REQUIRE_FALSE(m.value().autokey_lag_ok());
}

TEST_CASE("TheoryShapeMatch Autokey name-irrelevant custom still Autokey",
          "[dsl][shape][s4]") {
    // Theory name is unrelated to catalog autokey_lag / vigenere — math wins.
    const TheoryIr theory = TheoryShapeMatchTestUtil::make_theory(
        "foo_stream_custom",
        Z29Expr::sub(TheoryShapeMatchTestUtil::x(),
                     Z29Expr::call("z29_autokey_shift",
                                   {TheoryShapeMatchTestUtil::x(),
                                    TheoryShapeMatchTestUtil::v("lag")})));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match_theory(theory);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Autokey);
    REQUIRE(m.value().lag_name() == "lag");
}

TEST_CASE("TheoryShapeMatch prefer_branch Select is Unknown until measured twin",
          "[dsl][shape][prefer_branch]") {
    const Z29Expr::Ptr expr = Z29Expr::select(
        TheoryShapeMatchTestUtil::v("c"),
        Z29Expr::sub(TheoryShapeMatchTestUtil::lit(28), TheoryShapeMatchTestUtil::x()),
        TheoryShapeMatchTestUtil::x(), /*prefer_branch=*/true);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Unknown);
    REQUIRE(m.value().reason().find("prefer_branch") != std::string::npos);
    REQUIRE(m.value().reason().find("measured twin") != std::string::npos);
}

TEST_CASE("TheoryShapeMatch mux Select without prefer_branch is not Unknown-by-flag",
          "[dsl][shape][prefer_branch]") {
    const Z29Expr::Ptr expr = Z29Expr::select(
        TheoryShapeMatchTestUtil::v("c"),
        Z29Expr::sub(TheoryShapeMatchTestUtil::lit(28), TheoryShapeMatchTestUtil::x()),
        TheoryShapeMatchTestUtil::x(), /*prefer_branch=*/false);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() != TheoryShapeMatch::ShapeId::Unknown);
    // Select root is not Atbash/Caesar; falls to FxOnly (no stream i).
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::FxOnly);
}

TEST_CASE("TheoryShapeMatch name-irrelevant: theory name foo_bar still Atbash", "[dsl][shape]") {
    const TheoryIr theory = TheoryShapeMatchTestUtil::make_theory(
        "foo_bar", Z29Expr::sub(TheoryShapeMatchTestUtil::lit(28), TheoryShapeMatchTestUtil::x()));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match_theory(theory);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Atbash);
}

TEST_CASE("TheoryShapeMatch name-irrelevant: family string caesar but math Affine", "[dsl][shape]") {
    // Theory name suggests caesar; decrypt is affine — math wins.
    const TheoryIr theory = TheoryShapeMatchTestUtil::make_theory(
        "my_caesar_stream",
        Z29Expr::add(Z29Expr::mul(TheoryShapeMatchTestUtil::v("a"), TheoryShapeMatchTestUtil::x()),
                     TheoryShapeMatchTestUtil::v("b")));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match_theory(theory);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Affine);
}

TEST_CASE("TheoryShapeMatch cipher_var override", "[dsl][shape]") {
    const Z29Expr::Ptr expr =
        Z29Expr::sub(TheoryShapeMatchTestUtil::lit(28), TheoryShapeMatchTestUtil::v("cipher"));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr, "cipher");
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Atbash);
}

TEST_CASE("TheoryShapeMatch already_normalized skips second normalize", "[dsl][shape]") {
    const Z29Expr::Ptr raw =
        Z29Expr::sub(TheoryShapeMatchTestUtil::lit(28), TheoryShapeMatchTestUtil::x());
    const StatusOr<Z29Expr::Ptr> norm = Z29ExprNormalize::normalize(raw);
    REQUIRE(norm.ok());
    const StatusOr<TheoryShapeMatch::Match> m =
        TheoryShapeMatch::match(norm.value(), "x", /*already_normalized=*/true);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Atbash);
}

TEST_CASE("TheoryShapeMatch null / empty cipher_var errors", "[dsl][shape]") {
    REQUIRE_FALSE(TheoryShapeMatch::match(nullptr).ok());
    REQUIRE_FALSE(TheoryShapeMatch::match(TheoryShapeMatchTestUtil::x(), "").ok());
}

TEST_CASE("TheoryShapeMatch shape_str covers ladder", "[dsl][shape]") {
    REQUIRE(std::string(TheoryShapeMatch::shape_str(TheoryShapeMatch::ShapeId::Atbash)) ==
            "Atbash");
    REQUIRE(std::string(TheoryShapeMatch::shape_str(TheoryShapeMatch::ShapeId::Unknown)) ==
            "Unknown");
    REQUIRE(std::string(TheoryShapeMatch::shape_str(TheoryShapeMatch::ShapeId::LinearKeystream)) ==
            "LinearKeystream");
}
