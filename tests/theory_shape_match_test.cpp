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

TEST_CASE("TheoryShapeMatch Autokey negative/special", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::sub(
        TheoryShapeMatchTestUtil::x(),
        Z29Expr::call("z29_autokey_shift",
                      {TheoryShapeMatchTestUtil::x(), TheoryShapeMatchTestUtil::v("L")}));
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Autokey);
}

TEST_CASE("TheoryShapeMatch prefer_branch Select is Unknown", "[dsl][shape]") {
    const Z29Expr::Ptr expr = Z29Expr::select(
        TheoryShapeMatchTestUtil::v("c"),
        Z29Expr::sub(TheoryShapeMatchTestUtil::lit(28), TheoryShapeMatchTestUtil::x()),
        TheoryShapeMatchTestUtil::x(), /*prefer_branch=*/true);
    const StatusOr<TheoryShapeMatch::Match> m = TheoryShapeMatch::match(expr);
    REQUIRE(m.ok());
    REQUIRE(m.value().shape() == TheoryShapeMatch::ShapeId::Unknown);
    REQUIRE(m.value().reason().find("prefer_branch") != std::string::npos);
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
