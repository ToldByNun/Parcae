#include <catch2/catch_test_macros.hpp>

#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/dsl_optimize.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_shape_match.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <optional>
#include <string>

namespace {

[[nodiscard]] TheoryIr make_caesar_theory() {
    const StatusOr<ParamIr> shift_p = ParamIr::make("shift", 0, 28);
    REQUIRE(shift_p.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr shift = Z29Expr::var("shift");
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("emit_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {shift_p.value()},
                       Z29Expr::add(x, shift), Z29Expr::sub(x, shift));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_progressive_theory() {
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr s =
        Z29Expr::add(Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), i));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_progressive", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, s),
        Z29Expr::sub(x, s), std::string("S2 candidate progressive."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_autokey_theory() {
    const StatusOr<ParamIr> lag = ParamIr::make("lag", 1, 28);
    REQUIRE(lag.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr lag_v = Z29Expr::var("lag");
    const Z29Expr::Ptr prior = Z29Expr::call("z29_autokey_shift", {x, lag_v});
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("emit_autokey", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
                       TheoryIr::InterruptMode::NoneByDesign, {lag.value()},
                       Z29Expr::add(x, prior), Z29Expr::sub(x, prior),
                       std::string("Autokey S4 AutokeyRing."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_autokey_const_lag_theory() {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<Z29Expr::Ptr> lag = Z29Expr::constant(5);
    REQUIRE(lag.ok());
    const Z29Expr::Ptr prior = Z29Expr::call("z29_autokey_shift", {x, lag.value()});
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_autokey_const_lag", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {}, Z29Expr::add(x, prior),
        Z29Expr::sub(x, prior), std::string("Autokey S4 with folded const lag."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_autokey_unbound_lag_theory() {
    // lag = i → Autokey shape but no bindable lag → hard S0.
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr prior = Z29Expr::call("z29_autokey_shift", {x, Z29Expr::var("i")});
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_autokey_bad_lag", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {}, Z29Expr::add(x, prior),
        Z29Expr::sub(x, prior), std::string("Autokey without lag bind → S0."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_poly_keystream_theory() {
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    const StatusOr<ParamIr> b2 = ParamIr::make("b2", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    REQUIRE(b2.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr ks =
        Z29Expr::add(Z29Expr::var("b0"),
                     Z29Expr::add(Z29Expr::mul(Z29Expr::var("b1"), i),
                                  Z29Expr::mul(Z29Expr::var("b2"), Z29Expr::mul(i, i))));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_poly_stream", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value(), b2.value()},
        Z29Expr::add(x, ks), Z29Expr::sub(x, ks),
        std::string("S5 poly / bitmask_blend-class keystream."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_complex_i_theory() {
    // Uses i but root is Mul — not simple ± keystream → S3.
    const StatusOr<ParamIr> k = ParamIr::make("k", 0, 28);
    REQUIRE(k.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr body = Z29Expr::mul(x, Z29Expr::add(Z29Expr::var("k"), Z29Expr::var("i")));
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("emit_complex_i", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
                       TheoryIr::InterruptMode::NoneByDesign, {k.value()}, body, body,
                       std::string("S3 candidate."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_affine_theory() {
    const StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
    const StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr av = Z29Expr::var("a");
    const Z29Expr::Ptr bv = Z29Expr::var("b");
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_affine", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {a.value(), b.value()},
        Z29Expr::add(Z29Expr::mul(av, x), bv),
        Z29Expr::mul(Z29Expr::inv(av), Z29Expr::sub(x, bv)),
        std::string("Affine decrypt ShapeInline candidate."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_atbash_arith_theory() {
    // Self-written Atbash as pure arith — name-irrelevant ShapeInline.
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr twenty_eight = Z29Expr::constant(28).value();
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_foo_bar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {},
        Z29Expr::sub(twenty_eight, x), Z29Expr::sub(twenty_eight, x),
        std::string("Atbash arith ShapeInline."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_nonlinear_s2_theory() {
    // x - (b0 + b1*i*i) — classified S2 (±g(i)) but not linear b0+b1*i.
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr s = Z29Expr::add(
        Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), Z29Expr::mul(i, i)));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_quad_stream", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, s),
        Z29Expr::sub(x, s), std::string("S2 shape, non-linear g."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_bare_b1i_theory() {
    // x - (b1*i) — widened LinearKeystream (implied b0=0).
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr ks =
        Z29Expr::mul(Z29Expr::var("b1"), Z29Expr::var("i"));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_bare_b1i", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b1.value()}, Z29Expr::add(x, ks),
        Z29Expr::sub(x, ks), std::string("S2 bare b1*i."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_const_b0_linear_theory() {
    // x - (5 + b1*i) — const b0 edge form.
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr ks = Z29Expr::add(
        Z29Expr::constant(5).value(),
        Z29Expr::mul(Z29Expr::var("b1"), Z29Expr::var("i")));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_const_b0_lin", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b1.value()}, Z29Expr::add(x, ks),
        Z29Expr::sub(x, ks), std::string("S2 const b0 + b1*i."));
    REQUIRE(theory.ok());
    return theory.value();
}

} // namespace

TEST_CASE("TheoryHistChi2Emit selects ShapeInline for caesar-shaped decrypt",
          "[dsl][emit][hist][chi2][shape]") {
    const TheoryIr theory = make_caesar_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(sel.is_specialized());
    REQUIRE(sel.shape().has_value());
    REQUIRE(sel.shape()->shape() == TheoryShapeMatch::ShapeId::Caesar);
}

TEST_CASE("TheoryHistChi2Emit selects ShapeInline for self-written Atbash arith",
          "[dsl][emit][hist][chi2][shape]") {
    const TheoryIr theory = make_atbash_arith_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(sel.shape().has_value());
    REQUIRE(sel.shape()->shape() == TheoryShapeMatch::ShapeId::Atbash);
    REQUIRE(std::string(theory.name()).find("atbash") == std::string::npos);
}

TEST_CASE("TheoryHistChi2Emit selects S2 for progressive x±g(i)", "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_progressive_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
}

TEST_CASE("TheoryHistChi2Emit selects S3 for non-± keystream with i",
          "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_complex_i_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S3ScalarInline);
}

TEST_CASE("TheoryHistChi2Emit selects S4 AutokeyRing for vigenere_lag",
          "[dsl][emit][hist][chi2][s4]") {
    const TheoryIr theory = make_autokey_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S4AutokeyRing);
    REQUIRE(sel.is_specialized());
    REQUIRE(sel.shape().has_value());
    REQUIRE(sel.shape()->autokey_lag_ok());
    REQUIRE(sel.shape()->lag_name() == "lag");

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S4AutokeyRing);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s4_autokey().has_value());
    REQUIRE(bundle.value().s4_autokey()->lag_name() == "lag");
    REQUIRE(bundle.value().s4_autokey()->cipher_minus_ks());
    REQUIRE_FALSE(bundle.value().header_text().empty());
}

TEST_CASE("TheoryHistChi2Emit S4 AutokeyRing for const lag",
          "[dsl][emit][hist][chi2][s4]") {
    const TheoryIr theory = make_autokey_const_lag_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S4AutokeyRing);
    REQUIRE(sel.shape()->has_const_lag());
    REQUIRE(sel.shape()->const_lag() == 5);

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S4AutokeyRing);
    REQUIRE(bundle.value().s4_autokey().has_value());
    REQUIRE(bundle.value().s4_autokey()->has_const_lag());
    REQUIRE(bundle.value().s4_autokey()->const_lag() == 5);
    REQUIRE(bundle.value().header_text().find("TheoryHistChi2S4") != std::string::npos);
}

TEST_CASE("TheoryHistChi2Emit Autokey without lag bind soft-falls S0",
          "[dsl][emit][hist][chi2][s4]") {
    const TheoryIr theory = make_autokey_unbound_lag_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(sel.is_specialized());

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(bundle.value().specialized());
}

TEST_CASE("TheoryHistChi2Emit selects S5 for poly / bitmask_blend keystream",
          "[dsl][emit][hist][chi2][s5]") {
    const TheoryIr theory = make_poly_keystream_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S5PolyKeystream);
    REQUIRE(sel.is_specialized());
    REQUIRE(sel.shape().has_value());
    REQUIRE(sel.shape()->shape() == TheoryShapeMatch::ShapeId::PolyKeystream);
    REQUIRE(sel.shape()->poly_coeffs_ok());

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S5PolyKeystream);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s5_poly().has_value());
    REQUIRE(bundle.value().s5_poly()->b0_name() == "b0");
    REQUIRE(bundle.value().s5_poly()->b1_name() == "b1");
    REQUIRE(bundle.value().s5_poly()->b2_name() == "b2");
    REQUIRE(bundle.value().s5_poly()->cipher_minus_ks());
    REQUIRE(bundle.value().header_text().find("TheoryHistChi2S5") != std::string::npos);
    REQUIRE_FALSE(bundle.value().header_text().empty());
}

TEST_CASE("TheoryHistChi2Emit S5 for bare b2*i*i (implied b0=b1=0)",
          "[dsl][emit][hist][chi2][s5]") {
    const StatusOr<ParamIr> b2 = ParamIr::make("b2", 0, 28);
    REQUIRE(b2.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr ks = Z29Expr::mul(Z29Expr::var("b2"), Z29Expr::mul(i, i));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_poly_bare_b2", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b2.value()}, Z29Expr::add(x, ks),
        Z29Expr::sub(x, ks), std::string("S5 bare quadratic term."));
    REQUIRE(theory.ok());

    const TheoryHistChi2Emit::Selection sel =
        TheoryHistChi2Emit::select_strategy(theory.value());
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S5PolyKeystream);
    REQUIRE(sel.shape()->has_const_b0());
    REQUIRE(sel.shape()->const_b0() == 0);
    REQUIRE(sel.shape()->has_const_b1());
    REQUIRE(sel.shape()->const_b1() == 0);
    REQUIRE(sel.shape()->b2_name() == "b2");

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory.value());
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S5PolyKeystream);
    REQUIRE(bundle.value().s5_poly()->has_const_b0());
    REQUIRE(bundle.value().s5_poly()->has_const_b1());
    REQUIRE(bundle.value().s5_poly()->b2_name() == "b2");
}

TEST_CASE("TheoryHistChi2Emit S2 linear progressive emits uchar4 sources",
          "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_progressive_theory();
    std::optional<TheoryHistChi2Emit::S2LinearPlan> plan =
        TheoryHistChi2Emit::match_s2_linear(theory);
    REQUIRE(plan.has_value());
    REQUIRE(plan->b0_name() == "b0");
    REQUIRE(plan->b1_name() == "b1");
    REQUIRE(plan->cipher_minus_ks());

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s2_linear().has_value());
    REQUIRE(bundle.value().kernel_symbol() == "emit_progressive_s2_hist_kernel");
    REQUIRE(bundle.value().cu_text().find("uchar4") != std::string::npos);
    REQUIRE(bundle.value().cu_text().find("emit_progressive_s2_hist_kernel") != std::string::npos);
    REQUIRE(bundle.value().cu_text().find("namespace {") == std::string::npos);
    REQUIRE(bundle.value().header_text().find("TheoryHistChi2S2") != std::string::npos);

    StatusOr<DslLaunchPlan::Plan> launch = TheoryHistChi2Emit::hist_launch_plan(29, 1024);
    REQUIRE(launch.ok());
    REQUIRE(launch.value().kind() == DslLaunchPlan::Kind::HistChi2_2D);
    REQUIRE(launch.value().grid_x() == 29);
}

TEST_CASE("TheoryHistChi2Emit S2 widen bare b1*i specializes (not soft S0)",
          "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_bare_b1i_theory();
    REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
            TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    std::optional<TheoryHistChi2Emit::S2LinearPlan> plan =
        TheoryHistChi2Emit::match_s2_linear(theory);
    REQUIRE(plan.has_value());
    REQUIRE(plan->has_const_b0());
    REQUIRE(plan->const_b0() == 0);
    REQUIRE(plan->b1_name() == "b1");
    REQUIRE(plan->coeffs_ok());

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    REQUIRE(bundle.value().s2_linear().has_value());
    REQUIRE(bundle.value().s2_linear()->has_const_b0());
}

TEST_CASE("TheoryHistChi2Emit S2 widen const b0 + b1*i specializes",
          "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_const_b0_linear_theory();
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    REQUIRE(bundle.value().s2_linear()->has_const_b0());
    REQUIRE(bundle.value().s2_linear()->const_b0() == 5);
    REQUIRE(bundle.value().s2_linear()->b1_name() == "b1");
}

TEST_CASE("TheoryHistChi2Emit non-linear keystream specializes S3",
          "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_nonlinear_s2_theory();
    REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
            TheoryHistChi2Emit::Strategy::S3ScalarInline);
    REQUIRE_FALSE(TheoryHistChi2Emit::match_s2_linear(theory).has_value());

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S3ScalarInline);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S3ScalarInline);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s3_scalar().has_value());
    REQUIRE(bundle.value().s3_scalar()->within_caps());
    REQUIRE(bundle.value().s3_scalar()->binds_index_i());
    REQUIRE(bundle.value().kernel_symbol() == "emit_quad_stream_s3_hist_kernel");
    REQUIRE(bundle.value().cu_text().find("namespace {") == std::string::npos);
}

TEST_CASE("TheoryHistChi2Emit ShapeInline caesar uses HistFast shape twin",
          "[dsl][emit][hist][chi2][shape]") {
    const TheoryIr theory = make_caesar_theory();
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().has_shape_caesar_kernel());
    REQUIRE_FALSE(bundle.value().has_s1_soft_path());
    REQUIRE(bundle.value().shape().has_value());
    REQUIRE(bundle.value().shape()->shape() == TheoryShapeMatch::ShapeId::Caesar);
    REQUIRE(bundle.value().shape()->shift_name() == "shift");
    REQUIRE(bundle.value().kernel_symbol() == "theory_hist_chi2_shape_caesar_kernel");
}

TEST_CASE("TheoryHistChi2Emit ShapeInline Atbash uses HistFast shape twin",
          "[dsl][emit][hist][chi2][shape]") {
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(make_atbash_arith_theory());
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(bundle.value().shape()->shape() == TheoryShapeMatch::ShapeId::Atbash);
    REQUIRE(bundle.value().has_shape_atbash_kernel());
    REQUIRE_FALSE(bundle.value().has_s1_soft_path());
    REQUIRE(bundle.value().kernel_symbol() == "theory_hist_chi2_shape_atbash_kernel");
}

TEST_CASE("TheoryHistChi2Emit ShapeInline affine decrypt uses HistFast shape twin",
          "[dsl][emit][hist][chi2][shape]") {
    const TheoryIr theory = make_affine_theory();
    REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
            TheoryHistChi2Emit::Strategy::ShapeInline);
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(bundle.value().has_shape_affine_kernel());
    REQUIRE(bundle.value().shape()->a_name() == "a");
    REQUIRE(bundle.value().shape()->b_name() == "b");
    REQUIRE(bundle.value().kernel_symbol() == "theory_hist_chi2_shape_affine_kernel");
}

TEST_CASE("TheoryHistChi2Emit S3 ExprLower specializes Mul(x,i)-shaped",
          "[dsl][emit][hist][chi2]") {
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(make_complex_i_theory());
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S3ScalarInline);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S3ScalarInline);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s3_scalar().has_value());
}

TEST_CASE("TheoryHistChi2Emit prefer_branch stays hard S0 until measured twin",
          "[dsl][emit][hist][chi2][prefer_branch]") {
    const StatusOr<ParamIr> c = ParamIr::make("c", 0, 1);
    REQUIRE(c.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr expr = Z29Expr::select(
        Z29Expr::var("c"), Z29Expr::sub(Z29Expr::constant(28).value(), x), x,
        /*prefer_branch=*/true);
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_prefer_branch", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {c.value()}, expr, expr,
        std::string("prefer_branch → S0."));
    REQUIRE(theory.ok());

    const TheoryHistChi2Emit::Selection sel =
        TheoryHistChi2Emit::select_strategy(theory.value());
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE(sel.reason().find("measured twin") != std::string::npos);

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory.value());
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(bundle.value().specialized());
    REQUIRE(bundle.value().reason().find("measured twin") != std::string::npos);
}

TEST_CASE("TheoryHistChi2Emit Affine ShapeInline stays specialized despite inv hoists",
          "[dsl][emit][hist][chi2][hoist]") {
    // Policy: ShapeInline twins ignore HotLoop temps; keep specialized when
    // DslOptimize surfaces decrypt inv-hoists (export/plan_writer path).
    const TheoryIr theory = make_affine_theory();
    StatusOr<DslOptimize::TheoryResult> opt = DslOptimize::optimize_theory(theory);
    REQUIRE(opt.ok());
    REQUIRE(opt.value().decrypt().inv_hoists() >= 1);

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist_optimized(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().has_shape_affine_kernel());
}

TEST_CASE("TheoryHistChi2Emit S1 soft-falls S0 when decrypt inv hoists present",
          "[dsl][emit][hist][chi2][hoist]") {
    // FxOnly body-eval path: Add(x, inv(k)) — not Affine decrypt / Caesar.
    // Hoist prelude is not wired into S1 hist; must soft-fall S0 (no wrong scores).
    const StatusOr<ParamIr> k = ParamIr::make("k", 1, 28);
    REQUIRE(k.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr body = Z29Expr::add(x, Z29Expr::inv(Z29Expr::var("k")));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "emit_fx_plus_inv", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {k.value()}, body, body,
        std::string("FxOnly + inv → S1 classify, hoist soft S0."));
    REQUIRE(theory.ok());

    REQUIRE(TheoryHistChi2Emit::select_strategy(theory.value()).strategy() ==
            TheoryHistChi2Emit::Strategy::S1Lut29);

    StatusOr<TheoryHistChi2Emit::EmitBundle> no_hoist =
        TheoryHistChi2Emit::emit_decrypt_hist(theory.value());
    REQUIRE(no_hoist.ok());
    REQUIRE(no_hoist.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S1Lut29);
    REQUIRE(no_hoist.value().specialized());

    StatusOr<TheoryHistChi2Emit::EmitBundle> with_hoist =
        TheoryHistChi2Emit::emit_decrypt_hist_optimized(theory.value());
    REQUIRE(with_hoist.ok());
    REQUIRE(with_hoist.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S1Lut29);
    REQUIRE(with_hoist.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(with_hoist.value().specialized());
    REQUIRE(with_hoist.value().reason().find("hoists") != std::string::npos);
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_launch.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        out[i] = indices[i].value();
    }
    return out;
}

} // namespace

TEST_CASE("TheoryHistChi2Emit S2 effective_strategy is specialized",
          "[dsl][emit][hist][chi2][cuda]") {
    REQUIRE(TheoryHistChi2Launch::effective_strategy(
                TheoryHistChi2Emit::emit_decrypt_hist(make_progressive_theory()).value()) ==
            TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
}

TEST_CASE("TheoryHistChi2Emit ShapeInline effective_strategy is specialized",
          "[dsl][emit][hist][chi2][cuda]") {
    REQUIRE(TheoryHistChi2Launch::effective_strategy(
                TheoryHistChi2Emit::emit_decrypt_hist(make_caesar_theory()).value()) ==
            TheoryHistChi2Emit::Strategy::ShapeInline);
}

TEST_CASE("TheoryHistChi2 Shape Caesar+Affine golden: bytecode ≡ shape ≡ catalog",
          "[dsl][emit][hist][chi2][cuda][golden][shape]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    std::vector<Index29> cipher;
    for (std::uint8_t i = 0; i < 64; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>((i * 3u + 5u) % 29u)});
    }
    const std::vector<std::uint8_t> host_in = to_bytes(cipher);

    SECTION("Caesar shape twin") {
        const TheoryIr theory = make_caesar_theory();
        REQUIRE(TheoryHistChi2Emit::emit_decrypt_hist(theory).value().has_shape_caesar_kernel());
        const StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
        REQUIRE(prog.ok());

        std::vector<nlohmann::json> params_list;
        std::vector<std::uint8_t> host_shifts;
        for (int shift = 0; shift < 29; ++shift) {
            params_list.push_back(nlohmann::json{{"shift", shift}});
            host_shifts.push_back(static_cast<std::uint8_t>(shift));
        }
        const std::size_t C = params_list.size();
        const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.value().slot_names.size());
        std::vector<std::uint8_t> ops;
        for (Z29Bytecode::Op op : prog.value().ops) {
            ops.push_back(Z29Bytecode::op_as_u8(op));
        }
        std::vector<std::uint8_t> slots_flat(C * slot_count, 0);
        for (std::size_t c = 0; c < C; ++c) {
            StatusOr<std::vector<Index29>> bound =
                Z29Bytecode::bind_theory_slots(prog.value(), theory, params_list[c]);
            REQUIRE(bound.ok());
            for (std::uint16_t s = 0; s < slot_count; ++s) {
                slots_flat[c * slot_count + s] = bound.value()[s].value();
            }
        }

        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        REQUIRE(device_in.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_ops =
            DeviceBuffer<std::uint8_t>::from_host(ops);
        REQUIRE(device_ops.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_imm =
            DeviceBuffer<std::uint8_t>::from_host(prog.value().imm);
        REQUIRE(device_imm.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_slots =
            DeviceBuffer<std::uint8_t>::from_host(slots_flat);
        REQUIRE(device_slots.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(host_shifts);
        REQUIRE(device_shifts.ok());
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.value().probabilities().data(),
                                    freqs.value().probabilities().size()));
        REQUIRE(device_probs.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> counts_bc =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(counts_bc.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> counts_shape =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(counts_shape.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> counts_cat =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(counts_cat.ok());
        StatusOr<DeviceBuffer<double>> scores_bc = DeviceBuffer<double>::allocate(C);
        REQUIRE(scores_bc.ok());
        StatusOr<DeviceBuffer<double>> scores_shape = DeviceBuffer<double>::allocate(C);
        REQUIRE(scores_shape.ok());
        StatusOr<DeviceBuffer<double>> scores_cat = DeviceBuffer<double>::allocate(C);
        REQUIRE(scores_cat.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_err =
            DeviceBuffer<std::uint8_t>::allocate(C);
        REQUIRE(device_err.ok());
        const std::uint16_t max_stack =
            prog.value().max_stack == 0 ? 8 : prog.value().max_stack;

        REQUIRE(TheoryHistChi2Launch::launch_bytecode_async(
                    device_in.value().data(), device_ops.value().data(),
                    device_imm.value().data(), static_cast<std::uint32_t>(ops.size()),
                    device_slots.value().data(), slot_count, prog.value().cipher_slot,
                    prog.value().index_slot, 0u, max_stack, device_probs.value().data(),
                    counts_bc.value().data(), scores_bc.value().data(), device_err.value().data(),
                    C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "caesar shape bc sync").ok());
        REQUIRE(TheoryHistChi2Launch::launch_shape_caesar_async(
                    device_in.value().data(), device_shifts.value().data(),
                    device_probs.value().data(), counts_shape.value().data(),
                    scores_shape.value().data(), C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "caesar shape twin sync").ok());
        REQUIRE(CaesarChi2Batch::launch_decrypt_async(
                    device_in.value().data(), device_shifts.value().data(),
                    device_probs.value().data(), counts_cat.value().data(),
                    scores_cat.value().data(), C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "caesar catalog sync").ok());

        std::vector<double> bc(C), sh(C), cat(C);
        REQUIRE(scores_bc.value().copy_to_host(bc).ok());
        REQUIRE(scores_shape.value().copy_to_host(sh).ok());
        REQUIRE(scores_cat.value().copy_to_host(cat).ok());
        for (std::size_t c = 0; c < C; ++c) {
            REQUIRE(bc[c] == sh[c]);
            REQUIRE(sh[c] == cat[c]);
        }
    }

    SECTION("Affine decrypt shape twin") {
        const TheoryIr theory = make_affine_theory();
        REQUIRE(TheoryHistChi2Emit::emit_decrypt_hist(theory).value().has_shape_affine_kernel());
        const StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
        REQUIRE(prog.ok());

        std::vector<nlohmann::json> params_list;
        std::vector<std::uint8_t> host_a;
        std::vector<std::uint8_t> host_b;
        for (int a = 1; a <= 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                params_list.push_back(nlohmann::json{{"a", a}, {"b", b}});
                host_a.push_back(static_cast<std::uint8_t>(a));
                host_b.push_back(static_cast<std::uint8_t>(b));
            }
        }
        const std::size_t C = params_list.size();
        const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.value().slot_names.size());
        std::vector<std::uint8_t> ops;
        for (Z29Bytecode::Op op : prog.value().ops) {
            ops.push_back(Z29Bytecode::op_as_u8(op));
        }
        std::vector<std::uint8_t> slots_flat(C * slot_count, 0);
        for (std::size_t c = 0; c < C; ++c) {
            StatusOr<std::vector<Index29>> bound =
                Z29Bytecode::bind_theory_slots(prog.value(), theory, params_list[c]);
            REQUIRE(bound.ok());
            for (std::uint16_t s = 0; s < slot_count; ++s) {
                slots_flat[c * slot_count + s] = bound.value()[s].value();
            }
        }

        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        REQUIRE(device_in.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_ops =
            DeviceBuffer<std::uint8_t>::from_host(ops);
        REQUIRE(device_ops.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_imm =
            DeviceBuffer<std::uint8_t>::from_host(prog.value().imm);
        REQUIRE(device_imm.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_slots =
            DeviceBuffer<std::uint8_t>::from_host(slots_flat);
        REQUIRE(device_slots.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_a =
            DeviceBuffer<std::uint8_t>::from_host(host_a);
        REQUIRE(device_a.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_b =
            DeviceBuffer<std::uint8_t>::from_host(host_b);
        REQUIRE(device_b.ok());
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.value().probabilities().data(),
                                    freqs.value().probabilities().size()));
        REQUIRE(device_probs.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> counts_bc =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(counts_bc.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> counts_shape =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(counts_shape.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> counts_cat =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(counts_cat.ok());
        StatusOr<DeviceBuffer<double>> scores_bc = DeviceBuffer<double>::allocate(C);
        REQUIRE(scores_bc.ok());
        StatusOr<DeviceBuffer<double>> scores_shape = DeviceBuffer<double>::allocate(C);
        REQUIRE(scores_shape.ok());
        StatusOr<DeviceBuffer<double>> scores_cat = DeviceBuffer<double>::allocate(C);
        REQUIRE(scores_cat.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_err =
            DeviceBuffer<std::uint8_t>::allocate(C);
        REQUIRE(device_err.ok());
        const std::uint16_t max_stack =
            prog.value().max_stack == 0 ? 8 : prog.value().max_stack;

        REQUIRE(TheoryHistChi2Launch::launch_bytecode_async(
                    device_in.value().data(), device_ops.value().data(),
                    device_imm.value().data(), static_cast<std::uint32_t>(ops.size()),
                    device_slots.value().data(), slot_count, prog.value().cipher_slot,
                    prog.value().index_slot, 0u, max_stack, device_probs.value().data(),
                    counts_bc.value().data(), scores_bc.value().data(), device_err.value().data(),
                    C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "affine shape bc sync").ok());
        REQUIRE(TheoryHistChi2Launch::launch_shape_affine_async(
                    device_in.value().data(), device_a.value().data(), device_b.value().data(),
                    device_probs.value().data(), counts_shape.value().data(),
                    scores_shape.value().data(), C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "affine shape twin sync").ok());
        REQUIRE(FamilyChi2Batch::launch_affine_async(
                    device_in.value().data(), device_a.value().data(), device_b.value().data(),
                    device_probs.value().data(), counts_cat.value().data(),
                    scores_cat.value().data(), C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "affine catalog sync").ok());

        std::vector<double> bc(C), sh(C), cat(C);
        REQUIRE(scores_bc.value().copy_to_host(bc).ok());
        REQUIRE(scores_shape.value().copy_to_host(sh).ok());
        REQUIRE(scores_cat.value().copy_to_host(cat).ok());
        for (std::size_t c = 0; c < C; ++c) {
            REQUIRE(bc[c] == sh[c]);
            REQUIRE(sh[c] == cat[c]);
        }
    }
}

TEST_CASE("TheoryHistChi2 S2 linear golden: bytecode χ² == specialized χ²",
          "[dsl][emit][hist][chi2][cuda][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const TheoryIr theory = make_progressive_theory();
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s2_linear().has_value());
    REQUIRE(bundle.value().s2_linear()->cipher_minus_ks());

    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().binds_index_i);

    std::vector<Index29> cipher;
    for (std::uint8_t i = 0; i < 48; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 2u) % 29u)});
    }
    const std::vector<std::uint8_t> host_in = to_bytes(cipher);

    // Fixed 3×3 grid (b0,b1 ∈ {0,1,2}) — same shape as TheoryChi2Batch progressive golden.
    std::vector<nlohmann::json> params_list;
    std::vector<std::uint8_t> host_b0;
    std::vector<std::uint8_t> host_b1;
    for (int b0 = 0; b0 < 3; ++b0) {
        for (int b1 = 0; b1 < 3; ++b1) {
            params_list.push_back(nlohmann::json{{"b0", b0}, {"b1", b1}});
            host_b0.push_back(static_cast<std::uint8_t>(b0));
            host_b1.push_back(static_cast<std::uint8_t>(b1));
        }
    }
    const std::size_t C = params_list.size();

    // Pack bytecode slots.
    const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.value().slot_names.size());
    std::vector<std::uint8_t> ops;
    ops.reserve(prog.value().ops.size());
    for (Z29Bytecode::Op op : prog.value().ops) {
        ops.push_back(Z29Bytecode::op_as_u8(op));
    }
    std::vector<std::uint8_t> slots(C * slot_count, 0);
    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<std::vector<Index29>> bound =
            Z29Bytecode::bind_theory_slots(prog.value(), theory, params_list[c]);
        REQUIRE(bound.ok());
        for (std::uint16_t s = 0; s < slot_count; ++s) {
            slots[c * slot_count + s] = bound.value()[s].value();
        }
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_ops = DeviceBuffer<std::uint8_t>::from_host(ops);
    REQUIRE(device_ops.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_imm =
        DeviceBuffer<std::uint8_t>::from_host(prog.value().imm);
    REQUIRE(device_imm.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_slots =
        DeviceBuffer<std::uint8_t>::from_host(slots);
    REQUIRE(device_slots.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b0 =
        DeviceBuffer<std::uint8_t>::from_host(host_b0);
    REQUIRE(device_b0.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_b1 =
        DeviceBuffer<std::uint8_t>::from_host(host_b1);
    REQUIRE(device_b1.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(),
                                freqs.value().probabilities().size()));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts_bc =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts_bc.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts_s2 =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts_s2.ok());
    StatusOr<DeviceBuffer<double>> device_scores_bc = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores_bc.ok());
    StatusOr<DeviceBuffer<double>> device_scores_s2 = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores_s2.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_err = DeviceBuffer<std::uint8_t>::allocate(C);
    REQUIRE(device_err.ok());

    const std::uint16_t max_stack =
        prog.value().max_stack == 0 ? 8 : prog.value().max_stack;

    REQUIRE(TheoryHistChi2Launch::launch_bytecode_async(
                device_in.value().data(), device_ops.value().data(), device_imm.value().data(),
                static_cast<std::uint32_t>(ops.size()), device_slots.value().data(), slot_count,
                prog.value().cipher_slot, prog.value().index_slot,
                prog.value().binds_index_i ? 1u : 0u, max_stack, device_probs.value().data(),
                device_counts_bc.value().data(), device_scores_bc.value().data(),
                device_err.value().data(), C, host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S2 golden bytecode sync").ok());

    REQUIRE(TheoryHistChi2Launch::launch_s2_linear_async(
                device_in.value().data(), device_b0.value().data(), device_b1.value().data(),
                device_probs.value().data(), device_counts_s2.value().data(),
                device_scores_s2.value().data(), C, host_in.size(),
                /*cipher_minus_ks=*/true)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S2 golden specialized sync").ok());

    std::vector<double> scores_bc(C, 0.0);
    std::vector<double> scores_s2(C, 0.0);
    REQUIRE(device_scores_bc.value().copy_to_host(scores_bc).ok());
    REQUIRE(device_scores_s2.value().copy_to_host(scores_s2).ok());
    REQUIRE(scores_bc.size() == scores_s2.size());
    for (std::size_t c = 0; c < C; ++c) {
        REQUIRE(scores_bc[c] == scores_s2[c]);
    }
}

TEST_CASE("TheoryHistChi2 S3 scalar golden: bytecode χ² == specialized χ²",
          "[dsl][emit][hist][chi2][cuda][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const TheoryIr theory = make_nonlinear_s2_theory();
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s3_scalar().has_value());
    REQUIRE(bundle.value().s3_scalar()->within_caps());

    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().binds_index_i);

    std::vector<Index29> cipher;
    for (std::uint8_t i = 0; i < 48; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 2u) % 29u)});
    }
    const std::vector<std::uint8_t> host_in = to_bytes(cipher);

    std::vector<nlohmann::json> params_list;
    for (int b0 = 0; b0 < 3; ++b0) {
        for (int b1 = 0; b1 < 3; ++b1) {
            params_list.push_back(nlohmann::json{{"b0", b0}, {"b1", b1}});
        }
    }
    const std::size_t C = params_list.size();

    const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.value().slot_names.size());
    std::vector<std::uint8_t> ops;
    ops.reserve(prog.value().ops.size());
    for (Z29Bytecode::Op op : prog.value().ops) {
        ops.push_back(Z29Bytecode::op_as_u8(op));
    }
    std::vector<std::uint8_t> slots(C * slot_count, 0);
    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<std::vector<Index29>> bound =
            Z29Bytecode::bind_theory_slots(prog.value(), theory, params_list[c]);
        REQUIRE(bound.ok());
        for (std::uint16_t s = 0; s < slot_count; ++s) {
            slots[c * slot_count + s] = bound.value()[s].value();
        }
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_ops = DeviceBuffer<std::uint8_t>::from_host(ops);
    REQUIRE(device_ops.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_imm =
        DeviceBuffer<std::uint8_t>::from_host(prog.value().imm);
    REQUIRE(device_imm.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_slots = DeviceBuffer<std::uint8_t>::from_host(slots);
    REQUIRE(device_slots.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(),
                                freqs.value().probabilities().size()));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts_bc =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts_bc.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts_s3 =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts_s3.ok());
    StatusOr<DeviceBuffer<double>> device_scores_bc = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores_bc.ok());
    StatusOr<DeviceBuffer<double>> device_scores_s3 = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores_s3.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_err = DeviceBuffer<std::uint8_t>::allocate(C);
    REQUIRE(device_err.ok());

    const std::uint16_t max_stack =
        prog.value().max_stack == 0 ? 8 : prog.value().max_stack;

    REQUIRE(TheoryHistChi2Launch::launch_bytecode_async(
                device_in.value().data(), device_ops.value().data(), device_imm.value().data(),
                static_cast<std::uint32_t>(ops.size()), device_slots.value().data(), slot_count,
                prog.value().cipher_slot, prog.value().index_slot,
                prog.value().binds_index_i ? 1u : 0u, max_stack, device_probs.value().data(),
                device_counts_bc.value().data(), device_scores_bc.value().data(),
                device_err.value().data(), C, host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S3 golden bytecode sync").ok());

    REQUIRE(TheoryHistChi2Launch::launch_s3_scalar_async(
                device_in.value().data(), device_ops.value().data(), device_imm.value().data(),
                static_cast<std::uint32_t>(ops.size()), device_slots.value().data(), slot_count,
                prog.value().cipher_slot, prog.value().index_slot, device_probs.value().data(),
                device_counts_s3.value().data(), device_scores_s3.value().data(),
                device_err.value().data(), C, host_in.size(), max_stack)
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S3 golden specialized sync").ok());

    std::vector<double> scores_bc(C, 0.0);
    std::vector<double> scores_s3(C, 0.0);
    REQUIRE(device_scores_bc.value().copy_to_host(scores_bc).ok());
    REQUIRE(device_scores_s3.value().copy_to_host(scores_s3).ok());
    for (std::size_t c = 0; c < C; ++c) {
        REQUIRE(scores_bc[c] == scores_s3[c]);
    }
}

TEST_CASE("TheoryHistChi2 Shape Atbash golden: bytecode ≡ shape ≡ catalog atbash",
          "[dsl][emit][hist][chi2][cuda][golden][shape]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const TheoryIr theory = make_atbash_arith_theory();
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().has_shape_atbash_kernel());

    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE_FALSE(prog.value().binds_index_i);

    // Fixed cipher grid (same length class as other hist goldens).
    std::vector<Index29> cipher;
    for (std::uint8_t i = 0; i < 64; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>((i * 7u + 11u) % 29u)});
    }
    const std::vector<std::uint8_t> host_in = to_bytes(cipher);

    // Param-free Atbash: three identical candidate rows.
    constexpr std::size_t C = 3;
    std::vector<nlohmann::json> params_list(C, nlohmann::json::object());

    const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.value().slot_names.size());
    std::vector<std::uint8_t> ops;
    ops.reserve(prog.value().ops.size());
    for (Z29Bytecode::Op op : prog.value().ops) {
        ops.push_back(Z29Bytecode::op_as_u8(op));
    }
    std::vector<std::uint8_t> slots_flat(slot_count == 0 ? 1 : C * slot_count, 0);
    for (std::size_t c = 0; c < C; ++c) {
        StatusOr<std::vector<Index29>> bound =
            Z29Bytecode::bind_theory_slots(prog.value(), theory, params_list[c]);
        REQUIRE(bound.ok());
        REQUIRE(bound.value().size() == slot_count);
        for (std::uint16_t s = 0; s < slot_count; ++s) {
            slots_flat[c * slot_count + s] = bound.value()[s].value();
        }
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_ops = DeviceBuffer<std::uint8_t>::from_host(ops);
    REQUIRE(device_ops.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_imm =
        DeviceBuffer<std::uint8_t>::from_host(prog.value().imm.empty()
                                                  ? std::vector<std::uint8_t>{0}
                                                  : prog.value().imm);
    REQUIRE(device_imm.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_slots =
        DeviceBuffer<std::uint8_t>::from_host(slots_flat);
    REQUIRE(device_slots.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(),
                                freqs.value().probabilities().size()));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts_bc =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts_bc.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts_shape =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts_shape.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts_cat =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts_cat.ok());
    StatusOr<DeviceBuffer<double>> device_scores_bc = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores_bc.ok());
    StatusOr<DeviceBuffer<double>> device_scores_shape = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores_shape.ok());
    StatusOr<DeviceBuffer<double>> device_scores_cat = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores_cat.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_err = DeviceBuffer<std::uint8_t>::allocate(C);
    REQUIRE(device_err.ok());

    const std::uint16_t max_stack = prog.value().max_stack == 0 ? 8 : prog.value().max_stack;

    REQUIRE(TheoryHistChi2Launch::launch_bytecode_async(
                device_in.value().data(), device_ops.value().data(), device_imm.value().data(),
                static_cast<std::uint32_t>(ops.size()), device_slots.value().data(), slot_count,
                prog.value().cipher_slot, prog.value().index_slot, 0u, max_stack,
                device_probs.value().data(), device_counts_bc.value().data(),
                device_scores_bc.value().data(), device_err.value().data(), C, host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "shape atbash bytecode sync").ok());

    REQUIRE(TheoryHistChi2Launch::launch_shape_atbash_async(
                device_in.value().data(), device_probs.value().data(),
                device_counts_shape.value().data(), device_scores_shape.value().data(), C,
                host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "shape atbash twin sync").ok());

    REQUIRE(FamilyChi2Batch::launch_atbash_async(
                device_in.value().data(), device_probs.value().data(),
                device_counts_cat.value().data(), device_scores_cat.value().data(), C,
                host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "catalog atbash sync").ok());

    std::vector<double> scores_bc(C, 0.0);
    std::vector<double> scores_shape(C, 0.0);
    std::vector<double> scores_cat(C, 0.0);
    REQUIRE(device_scores_bc.value().copy_to_host(scores_bc).ok());
    REQUIRE(device_scores_shape.value().copy_to_host(scores_shape).ok());
    REQUIRE(device_scores_cat.value().copy_to_host(scores_cat).ok());
    for (std::size_t c = 0; c < C; ++c) {
        REQUIRE(scores_bc[c] == scores_shape[c]);
        REQUIRE(scores_shape[c] == scores_cat[c]);
    }
}

#endif
