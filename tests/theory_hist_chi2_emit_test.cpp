#include <catch2/catch_test_macros.hpp>

#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>

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
                       std::string("Autokey forces S0."));
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

} // namespace

TEST_CASE("TheoryHistChi2Emit selects S1 for caesar-shaped decrypt",
          "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_caesar_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S1Lut29);
    REQUIRE(sel.is_specialized());
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

TEST_CASE("TheoryHistChi2Emit forces S0 for autokey", "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_autokey_theory();
    const TheoryHistChi2Emit::Selection sel = TheoryHistChi2Emit::select_strategy(theory);
    REQUIRE(sel.strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(sel.is_specialized());
}

TEST_CASE("TheoryHistChi2Emit skeleton emit falls back to S0", "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_progressive_theory();
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(bundle.value().specialized());
    REQUIRE(bundle.value().header_text().empty());
    REQUIRE(bundle.value().cu_text().empty());
    REQUIRE(bundle.value().reason().find("skeleton") != std::string::npos);

    StatusOr<DslLaunchPlan::Plan> plan = TheoryHistChi2Emit::hist_launch_plan(29, 1024);
    REQUIRE(plan.ok());
    REQUIRE(plan.value().kind() == DslLaunchPlan::Kind::HistChi2_2D);
    REQUIRE(plan.value().grid_x() == 29);
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"
#include "theory_hist_chi2_launch.hpp"

TEST_CASE("TheoryHistChi2Launch façade has no specialized twins yet",
          "[dsl][emit][hist][chi2][cuda]") {
    REQUIRE_FALSE(TheoryHistChi2Launch::has_specialized("emit_progressive"));
    REQUIRE(TheoryHistChi2Launch::effective_strategy(
                TheoryHistChi2Emit::emit_decrypt_hist(make_progressive_theory()).value()) ==
            TheoryHistChi2Emit::Strategy::S0Bytecode);
}

#endif
