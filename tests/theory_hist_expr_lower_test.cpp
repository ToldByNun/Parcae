#include <catch2/catch_test_macros.hpp>

#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_hist_expr_lower.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_bytecode.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

[[nodiscard]] TheoryIr make_quad_stream() {
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr s =
        Z29Expr::add(Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), Z29Expr::mul(i, i)));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "lower_quad", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, s),
        Z29Expr::sub(x, s), std::string("ExprLower S3."));
    REQUIRE(theory.ok());
    return theory.value();
}

} // namespace

TEST_CASE("TheoryHistExprLower lowers bounded i-expr within S3 caps", "[dsl][lower][s3]") {
    const TheoryIr theory = make_quad_stream();
    StatusOr<TheoryHistExprLower::Lowered> low = TheoryHistExprLower::lower_decrypt(theory);
    REQUIRE(low.ok());
    REQUIRE(low.value().binds_index_i());
    REQUIRE(low.value().op_count() <= TheoryHistExprLower::kMaxProgramOps);
    REQUIRE(low.value().max_stack() <= TheoryHistExprLower::kMaxDeviceStack);
    REQUIRE(low.value().slot_count() <= TheoryHistExprLower::kMaxSlots);
    REQUIRE_FALSE(low.value().device_cpp().empty());
    REQUIRE(TheoryHistExprLower::within_caps(low.value().program()));
}

TEST_CASE("TheoryHistExprLower rejects missing i (S1 territory)", "[dsl][lower][s3]") {
    const StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "lower_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {shift.value()},
        Z29Expr::add(x, Z29Expr::var("shift")), Z29Expr::sub(x, Z29Expr::var("shift")));
    REQUIRE(theory.ok());
    StatusOr<TheoryHistExprLower::Lowered> low =
        TheoryHistExprLower::lower_decrypt(theory.value());
    REQUIRE_FALSE(low.ok());
    REQUIRE(low.status().message().find("no stream index") != std::string::npos);
}

TEST_CASE("TheoryHistExprLower check_caps soft-fails oversized program", "[dsl][lower][s3]") {
    Z29Bytecode::Program prog;
    prog.ops.assign(TheoryHistExprLower::kMaxProgramOps + 1, Z29Bytecode::Op::Const);
    prog.imm.assign(TheoryHistExprLower::kMaxProgramOps + 1, 0);
    prog.slot_names = {"x", "i"};
    prog.binds_index_i = true;
    prog.max_stack = 2;
    REQUIRE_FALSE(TheoryHistExprLower::within_caps(prog));
    REQUIRE_FALSE(TheoryHistExprLower::check_caps(prog).ok());
}

TEST_CASE("TheoryHistChi2Emit S3 caps stay aligned with ExprLower", "[dsl][lower][s3]") {
    REQUIRE(TheoryHistChi2Emit::kS3MaxProgramOps == TheoryHistExprLower::kMaxProgramOps);
    REQUIRE(TheoryHistChi2Emit::kS3MaxDeviceStack == TheoryHistExprLower::kMaxDeviceStack);
    REQUIRE(TheoryHistChi2Emit::kS3MaxSlots == TheoryHistExprLower::kMaxSlots);
}
