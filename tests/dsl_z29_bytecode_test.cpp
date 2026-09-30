#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_apply_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_bytecode.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <vector>

namespace {

TheoryIr make_caesar_theory() {
    const StatusOr<ParamIr> shift_p = ParamIr::make("shift", 0, 28);
    REQUIRE(shift_p.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr shift = Z29Expr::var("shift");
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("bc_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {shift_p.value()},
                       Z29Expr::add(x, shift), Z29Expr::sub(x, shift));
    REQUIRE(theory.ok());
    return theory.value();
}

TheoryIr make_progressive_theory() {
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr s =
        Z29Expr::add(Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), i));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "bc_progressive", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()},
        Z29Expr::add(x, s), Z29Expr::sub(x, s),
        std::string("Speculative. progressive stream bytecode parity."));
    REQUIRE(theory.ok());
    return theory.value();
}

} // namespace

TEST_CASE("Z29Bytecode caesar matches DslIrApplicator", "[dsl][bytecode]") {
    const TheoryIr theory = make_caesar_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().ops.size() >= 3);
    REQUIRE_FALSE(prog.value().binds_index_i);

    const std::vector<Index29> cipher{Index29{4}, Index29{15}, Index29{27}, Index29{0}};
    const nlohmann::json params{{"shift", 9}};

    const StatusOr<std::vector<Index29>> via_ir =
        DslIrApplicator::apply(theory, cipher, params, TransformDirection::Decrypt);
    REQUIRE(via_ir.ok());
    const StatusOr<std::vector<Index29>> via_bc =
        Z29Bytecode::apply_theory(prog.value(), theory, params, cipher);
    REQUIRE(via_bc.ok());
    REQUIRE(via_bc.value() == via_ir.value());
}

TEST_CASE("Z29Bytecode progressive stream binds i", "[dsl][bytecode]") {
    const TheoryIr theory = make_progressive_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().binds_index_i);

    const std::vector<Index29> cipher{Index29{3}, Index29{10}, Index29{28}, Index29{7},
                                      Index29{1}};
    const nlohmann::json params{{"b0", 5}, {"b1", 2}};

    const StatusOr<std::vector<Index29>> via_ir =
        DslIrApplicator::apply(theory, cipher, params, TransformDirection::Decrypt);
    REQUIRE(via_ir.ok());
    const StatusOr<std::vector<Index29>> via_bc =
        Z29Bytecode::apply_theory(prog.value(), theory, params, cipher);
    REQUIRE(via_bc.ok());
    REQUIRE(via_bc.value() == via_ir.value());
}

TEST_CASE("Z29Bytecode xor progressive parity", "[dsl][bytecode]") {
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr s =
        Z29Expr::add(Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), i));
    const Z29Expr::Ptr step = Z29Expr::bit_xor(x, s);
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "bc_xor_prog", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, step, step,
        std::string("Speculative. xor progressive bytecode."));
    REQUIRE(theory.ok());

    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory.value(), TransformDirection::Decrypt);
    REQUIRE(prog.ok());

    std::vector<Index29> cipher;
    for (std::uint8_t v = 0; v < 20; ++v) {
        cipher.push_back(Index29{static_cast<std::uint8_t>(v % 29)});
    }
    const nlohmann::json params{{"b0", 26}, {"b1", 8}};

    const StatusOr<std::vector<Index29>> via_ir =
        DslIrApplicator::apply(theory.value(), cipher, params, TransformDirection::Decrypt);
    REQUIRE(via_ir.ok());
    const StatusOr<std::vector<Index29>> via_bc =
        Z29Bytecode::apply_theory(prog.value(), theory.value(), params, cipher);
    REQUIRE(via_bc.ok());
    REQUIRE(via_bc.value() == via_ir.value());
}

TEST_CASE("Z29Bytecode Select mux parity", "[dsl][bytecode][select]") {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr step =
        Z29Expr::select(Z29Expr::eq(x, Z29Expr::constant(0).value()), Z29Expr::constant(5).value(),
                        Z29Expr::add(x, Z29Expr::constant(1).value()));
    const StatusOr<Z29Bytecode::Program> prog = Z29Bytecode::compile_expr(step, "x", {});
    REQUIRE(prog.ok());

    const std::vector<Index29> input{Index29{0}, Index29{3}, Index29{28}};
    std::vector<Index29> ir_out(input.size());
    std::vector<Index29> bc_out(input.size());
    REQUIRE(DslIrApplicator::apply_into(step, "x", {}, input, ir_out).ok());
    std::vector<Index29> slots(prog.value().slot_names.size(), Index29{0});
    REQUIRE(Z29Bytecode::apply_into(prog.value(), slots, input, bc_out).ok());
    REQUIRE(bc_out == ir_out);
}

TEST_CASE("Z29Bytecode interrupt pass-through", "[dsl][bytecode]") {
    const TheoryIr theory = make_caesar_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());

    const std::vector<Index29> cipher{Index29{0}, Index29{1}, Index29{2}};
    const nlohmann::json params{{"shift", 1}};
    const StatusOr<InterruptPolicy> irq = InterruptPolicy::from_skip_indices({1});
    REQUIRE(irq.ok());

    const StatusOr<std::vector<Index29>> via_ir =
        DslIrApplicator::apply(theory, cipher, params, TransformDirection::Decrypt, irq.value());
    REQUIRE(via_ir.ok());
    const StatusOr<std::vector<Index29>> via_bc =
        Z29Bytecode::apply_theory(prog.value(), theory, params, cipher, irq.value());
    REQUIRE(via_bc.ok());
    REQUIRE(via_bc.value() == via_ir.value());
}

TEST_CASE("Z29Bytecode autokey_shift parity", "[dsl][bytecode][autokey]") {
    const StatusOr<ParamIr> lag = ParamIr::make("lag", 1, 5);
    const StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(lag.ok());
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr auto_s =
        Z29Expr::call("z29_autokey_shift", {x, Z29Expr::var("lag")});
    const Z29Expr::Ptr step = Z29Expr::sub(Z29Expr::sub(x, auto_s), Z29Expr::var("shift"));
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "bc_autokey", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {lag.value(), shift.value()},
        Z29Expr::add(Z29Expr::add(x, auto_s), Z29Expr::var("shift")), step,
        std::string("Speculative. autokey bytecode."));
    REQUIRE(theory.ok());

    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory.value(), TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    bool saw_autokey = false;
    for (Z29Bytecode::Op op : prog.value().ops) {
        if (op == Z29Bytecode::Op::AutokeyShift) {
            saw_autokey = true;
        }
    }
    REQUIRE(saw_autokey);

    const std::vector<Index29> cipher{Index29{5}, Index29{8}, Index29{12}, Index29{3},
                                      Index29{20}, Index29{1}};
    const nlohmann::json params{{"lag", 2}, {"shift", 7}};

    const StatusOr<std::vector<Index29>> via_ir =
        DslIrApplicator::apply(theory.value(), cipher, params, TransformDirection::Decrypt);
    REQUIRE(via_ir.ok());
    const StatusOr<std::vector<Index29>> via_bc =
        Z29Bytecode::apply_theory(prog.value(), theory.value(), params, cipher);
    REQUIRE(via_bc.ok());
    REQUIRE(via_bc.value() == via_ir.value());
}

TEST_CASE("Z29Bytecode compile_apply_ir round-trip", "[dsl][bytecode][apply_ir]") {
    const TheoryIr theory = make_progressive_theory();
    const StatusOr<nlohmann::json> json = TheoryApplyIr::to_json(theory);
    REQUIRE(json.ok());

    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_apply_ir(json.value(), TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().binds_index_i);

    const std::vector<Index29> cipher{Index29{9}, Index29{2}, Index29{14}};
    const nlohmann::json params{{"b0", 1}, {"b1", 3}};
    const StatusOr<std::vector<Index29>> via_ir =
        DslIrApplicator::apply(theory, cipher, params, TransformDirection::Decrypt);
    const StatusOr<std::vector<Index29>> via_bc =
        Z29Bytecode::apply_theory(prog.value(), theory, params, cipher);
    REQUIRE(via_ir.ok());
    REQUIRE(via_bc.ok());
    REQUIRE(via_bc.value() == via_ir.value());
}

TEST_CASE("Z29Bytecode rejects none_by_design with interrupt", "[dsl][bytecode]") {
    const TheoryIr theory = make_progressive_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    const std::vector<Index29> cipher{Index29{1}, Index29{2}};
    const nlohmann::json params{{"b0", 0}, {"b1", 0}};
    const StatusOr<InterruptPolicy> irq = InterruptPolicy::from_skip_indices({0});
    REQUIRE(irq.ok());
    const StatusOr<std::vector<Index29>> rejected =
        Z29Bytecode::apply_theory(prog.value(), theory, params, cipher, irq.value());
    REQUIRE_FALSE(rejected.ok());
    REQUIRE(rejected.status().message().find("none_by_design") != std::string::npos);
}

TEST_CASE("Z29Bytecode det expand parity", "[dsl][bytecode][matrix]") {
    // s = det([[a,b],[c,d]]); p = x - s
    const StatusOr<ParamIr> a = ParamIr::make("a", 0, 28);
    const StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
    const StatusOr<ParamIr> c = ParamIr::make("c", 0, 28);
    const StatusOr<ParamIr> d = ParamIr::make("d", 0, 28);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    REQUIRE(c.ok());
    REQUIRE(d.ok());
    const Z29Expr::Ptr det = Z29Expr::call(
        "z29_det", {Z29Expr::var("a"), Z29Expr::var("b"), Z29Expr::var("c"), Z29Expr::var("d")});
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "bc_det", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {a.value(), b.value(), c.value(), d.value()}, Z29Expr::add(x, det), Z29Expr::sub(x, det));
    REQUIRE(theory.ok());

    // Tree eval of Call z29_det fails in Z29Expr::eval — applicator path uses expanded
    // trees from BuildIr. Expand via bytecode encode which inlines det_expr.
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory.value(), TransformDirection::Decrypt);
    REQUIRE(prog.ok());

    // Manual expanded IR for applicator oracle: x - (a*d - b*c)
    const Z29Expr::Ptr expanded = Z29Expr::sub(
        x, Z29Expr::sub(Z29Expr::mul(Z29Expr::var("a"), Z29Expr::var("d")),
                        Z29Expr::mul(Z29Expr::var("b"), Z29Expr::var("c"))));
    const StatusOr<TheoryIr> expanded_theory = TheoryIr::make(
        "bc_det_exp", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {a.value(), b.value(), c.value(), d.value()}, Z29Expr::add(x, Z29Expr::sub(Z29Expr::mul(Z29Expr::var("a"), Z29Expr::var("d")),
                                                                                    Z29Expr::mul(Z29Expr::var("b"), Z29Expr::var("c")))),
        expanded);
    REQUIRE(expanded_theory.ok());

    const std::vector<Index29> cipher{Index29{10}, Index29{20}, Index29{0}};
    const nlohmann::json params{{"a", 3}, {"b", 4}, {"c", 5}, {"d", 6}};
    const StatusOr<std::vector<Index29>> via_ir = DslIrApplicator::apply(
        expanded_theory.value(), cipher, params, TransformDirection::Decrypt);
    REQUIRE(via_ir.ok());
    const StatusOr<std::vector<Index29>> via_bc =
        Z29Bytecode::apply_theory(prog.value(), theory.value(), params, cipher);
    REQUIRE(via_bc.ok());
    REQUIRE(via_bc.value() == via_ir.value());
}
