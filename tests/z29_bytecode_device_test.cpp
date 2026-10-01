#include <catch2/catch_test_macros.hpp>

#include "parcae/core/index29.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "z29_bytecode_device.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

namespace {

[[nodiscard]] TheoryIr make_caesar_theory() {
    const StatusOr<ParamIr> shift_p = ParamIr::make("shift", 0, 28);
    REQUIRE(shift_p.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr shift = Z29Expr::var("shift");
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("dev_bc_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
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
        "dev_bc_progressive", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, s),
        Z29Expr::sub(x, s), std::string("Device bytecode progressive parity."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_select_theory() {
    const StatusOr<ParamIr> t = ParamIr::make("t", 0, 28);
    const StatusOr<ParamIr> f = ParamIr::make("f", 0, 28);
    REQUIRE(t.ok());
    REQUIRE(f.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    // decrypt: select(x == 0, t, f) — eager both arms
    const Z29Expr::Ptr step =
        Z29Expr::select(Z29Expr::eq(x, Z29Expr::constant(0).value()), Z29Expr::var("t"),
                        Z29Expr::var("f"));
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("dev_bc_select", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {t.value(), f.value()}, step,
                       step);
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
        TheoryIr::make("dev_bc_autokey", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
                       TheoryIr::InterruptMode::NoneByDesign, {lag.value()},
                       Z29Expr::add(x, prior), Z29Expr::sub(x, prior),
                       std::string("Device bytecode autokey parity."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] TheoryIr make_xor_theory() {
    const StatusOr<ParamIr> k = ParamIr::make("k", 0, 28);
    REQUIRE(k.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr key = Z29Expr::var("k");
    const Z29Expr::Ptr step = Z29Expr::bit_xor(x, key);
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("dev_bc_xor", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {k.value()}, step, step);
    REQUIRE(theory.ok());
    return theory.value();
}

struct PackedProgram {
    std::vector<std::uint8_t> ops;
    std::vector<std::uint8_t> imm;
    std::vector<std::uint8_t> slots;
    std::uint16_t cipher_slot = 0;
    std::uint16_t index_slot = 0;
    std::uint8_t binds_index_i = 0;
    std::uint16_t max_stack = 0;
};

[[nodiscard]] PackedProgram pack(const Z29Bytecode::Program& prog,
                                 const std::vector<Index29>& bound_slots) {
    REQUIRE(bound_slots.size() == prog.slot_names.size());
    PackedProgram out;
    out.ops.reserve(prog.ops.size());
    out.imm.reserve(prog.imm.size());
    for (Z29Bytecode::Op op : prog.ops) {
        out.ops.push_back(Z29Bytecode::op_as_u8(op));
    }
    out.imm = prog.imm;
    out.slots.resize(bound_slots.size());
    for (std::size_t i = 0; i < bound_slots.size(); ++i) {
        out.slots[i] = bound_slots[i].value();
    }
    out.cipher_slot = prog.cipher_slot;
    out.index_slot = prog.index_slot;
    out.binds_index_i = prog.binds_index_i ? 1u : 0u;
    out.max_stack = prog.max_stack == 0 ? 8 : prog.max_stack;
    REQUIRE(out.max_stack <= Z29BytecodeDevice::kMaxDeviceStack);
    return out;
}

[[nodiscard]] bool device_eval_index(const PackedProgram& packed,
                                     const std::vector<std::uint8_t>& stream, std::size_t i,
                                     std::uint8_t* out_byte, std::uint8_t* err_flag) {
    std::vector<std::uint8_t> slots = packed.slots;
    std::vector<std::uint8_t> stack(Z29BytecodeDevice::kMaxDeviceStack, 0);
    return Z29BytecodeDevice::eval_at(
        packed.ops.data(), packed.imm.data(), static_cast<std::uint32_t>(packed.ops.size()),
        slots.data(), static_cast<std::uint16_t>(slots.size()), packed.cipher_slot,
        packed.index_slot, packed.binds_index_i, stream.data(), stream.size(), i, stack.data(),
        packed.max_stack, out_byte, err_flag);
}

[[nodiscard]] bool device_eval_index_trusted(const PackedProgram& packed,
                                             const std::vector<std::uint8_t>& stream, std::size_t i,
                                             std::uint8_t* out_byte, std::uint8_t* err_flag) {
    std::vector<std::uint8_t> slots = packed.slots;
    std::vector<std::uint8_t> stack(Z29BytecodeDevice::kMaxDeviceStack, 0);
    return Z29BytecodeDevice::eval_at_trusted(
        packed.ops.data(), packed.imm.data(), static_cast<std::uint32_t>(packed.ops.size()),
        slots.data(), static_cast<std::uint16_t>(slots.size()), packed.cipher_slot,
        packed.index_slot, packed.binds_index_i, stream.data(), stream.size(), i, stack.data(),
        packed.max_stack, out_byte, err_flag);
}

void require_stream_parity(const Z29Bytecode::Program& prog, const TheoryIr& theory,
                           const nlohmann::json& params, const std::vector<Index29>& cipher) {
    StatusOr<std::vector<Index29>> host_slots =
        Z29Bytecode::bind_theory_slots(prog, theory, params);
    REQUIRE(host_slots.ok());

    PackedProgram packed = pack(prog, host_slots.value());
    std::vector<std::uint8_t> stream(cipher.size());
    for (std::size_t i = 0; i < cipher.size(); ++i) {
        stream[i] = cipher[i].value();
    }

    std::vector<Index29> mutable_slots = host_slots.value();
    for (std::size_t i = 0; i < cipher.size(); ++i) {
        StatusOr<Index29> host = Z29Bytecode::eval_at(prog, mutable_slots, cipher, i);
        REQUIRE(host.ok());

        std::uint8_t out = 0xFF;
        std::uint8_t err = 0xFF;
        REQUIRE(device_eval_index(packed, stream, i, &out, &err));
        REQUIRE(err == Z29BytecodeDevice::kErrOk);
        REQUIRE(out == host.value().value());

        std::uint8_t out_t = 0xFF;
        std::uint8_t err_t = 0xFF;
        REQUIRE(device_eval_index_trusted(packed, stream, i, &out_t, &err_t));
        REQUIRE(err_t == Z29BytecodeDevice::kErrOk);
        REQUIRE(out_t == host.value().value());
    }
}

} // namespace

TEST_CASE("Z29BytecodeDevice Op numerics match Z29Bytecode::Op", "[cuda][bytecode][parity]") {
    REQUIRE(Z29Bytecode::kOpCount == 29);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Const) == Z29BytecodeDevice::kOpConst);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Load) == Z29BytecodeDevice::kOpLoad);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Add) == Z29BytecodeDevice::kOpAdd);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Sub) == Z29BytecodeDevice::kOpSub);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Mul) == Z29BytecodeDevice::kOpMul);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Div) == Z29BytecodeDevice::kOpDiv);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::FloorDiv) == Z29BytecodeDevice::kOpFloorDiv);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Mod) == Z29BytecodeDevice::kOpMod);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Pow) == Z29BytecodeDevice::kOpPow);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::BitAnd) == Z29BytecodeDevice::kOpBitAnd);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::BitOr) == Z29BytecodeDevice::kOpBitOr);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::BitXor) == Z29BytecodeDevice::kOpBitXor);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::LShift) == Z29BytecodeDevice::kOpLShift);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::RShift) == Z29BytecodeDevice::kOpRShift);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Eq) == Z29BytecodeDevice::kOpEq);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Ne) == Z29BytecodeDevice::kOpNe);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Lt) == Z29BytecodeDevice::kOpLt);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Le) == Z29BytecodeDevice::kOpLe);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Gt) == Z29BytecodeDevice::kOpGt);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Ge) == Z29BytecodeDevice::kOpGe);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::BoolAnd) == Z29BytecodeDevice::kOpBoolAnd);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::BoolOr) == Z29BytecodeDevice::kOpBoolOr);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Neg) == Z29BytecodeDevice::kOpNeg);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Inv) == Z29BytecodeDevice::kOpInv);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Atbash) == Z29BytecodeDevice::kOpAtbash);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::BitNot) == Z29BytecodeDevice::kOpBitNot);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::BoolNot) == Z29BytecodeDevice::kOpBoolNot);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::Select) == Z29BytecodeDevice::kOpSelect);
    REQUIRE(Z29Bytecode::op_as_u8(Z29Bytecode::Op::AutokeyShift) ==
            Z29BytecodeDevice::kOpAutokeyShift);
}

TEST_CASE("Z29BytecodeDevice caesar decrypt matches host eval_at", "[cuda][bytecode][parity]") {
    const TheoryIr theory = make_caesar_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    const std::vector<Index29> cipher{Index29{4}, Index29{15}, Index29{27}, Index29{0},
                                      Index29{11}};
    require_stream_parity(prog.value(), theory, nlohmann::json{{"shift", 9}}, cipher);
}

TEST_CASE("Z29BytecodeDevice progressive binds i like host", "[cuda][bytecode][parity]") {
    const TheoryIr theory = make_progressive_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().binds_index_i);
    const std::vector<Index29> cipher{Index29{3}, Index29{10}, Index29{28}, Index29{7},
                                      Index29{1}, Index29{22}};
    require_stream_parity(prog.value(), theory, nlohmann::json{{"b0", 5}, {"b1", 2}}, cipher);
}

TEST_CASE("Z29BytecodeDevice xor / select / autokey match host", "[cuda][bytecode][parity]") {
    {
        const TheoryIr theory = make_xor_theory();
        const StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
        REQUIRE(prog.ok());
        require_stream_parity(prog.value(), theory, nlohmann::json{{"k", 17}},
                              {Index29{0}, Index29{8}, Index29{28}, Index29{3}});
    }
    {
        const TheoryIr theory = make_select_theory();
        const StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
        REQUIRE(prog.ok());
        require_stream_parity(prog.value(), theory, nlohmann::json{{"t", 3}, {"f", 11}},
                              {Index29{0}, Index29{5}, Index29{0}, Index29{12}});
    }
    {
        const TheoryIr theory = make_autokey_theory();
        const StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
        REQUIRE(prog.ok());
        require_stream_parity(prog.value(), theory, nlohmann::json{{"lag", 2}},
                              {Index29{4}, Index29{7}, Index29{8}, Index29{14}, Index29{16}});
    }
}

TEST_CASE("Z29BytecodeDevice domain errors set err_flag", "[cuda][bytecode][parity]") {
    // Program: Load cipher, Const 0, Div  → div by 0
    std::vector<std::uint8_t> ops{Z29BytecodeDevice::kOpLoad, Z29BytecodeDevice::kOpConst,
                                  Z29BytecodeDevice::kOpDiv};
    std::vector<std::uint8_t> imm{0, 0, 0};
    std::vector<std::uint8_t> slots{7};
    std::vector<std::uint8_t> stream{7};
    std::vector<std::uint8_t> stack(Z29BytecodeDevice::kMaxDeviceStack, 0);
    std::uint8_t out = 0;
    std::uint8_t err = 0;
    REQUIRE_FALSE(Z29BytecodeDevice::eval_at(
        ops.data(), imm.data(), static_cast<std::uint32_t>(ops.size()), slots.data(), 1, 0, 0, 0,
        stream.data(), stream.size(), 0, stack.data(), 8, &out, &err));
    REQUIRE(err == Z29BytecodeDevice::kErrDomain);

    err = 0;
    REQUIRE_FALSE(Z29BytecodeDevice::eval_at_trusted(
        ops.data(), imm.data(), static_cast<std::uint32_t>(ops.size()), slots.data(), 1, 0, 0, 0,
        stream.data(), stream.size(), 0, stack.data(), 8, &out, &err));
    REQUIRE(err == Z29BytecodeDevice::kErrDomain);

    // Inv(0): Const 0, Inv
    ops = {Z29BytecodeDevice::kOpConst, Z29BytecodeDevice::kOpInv};
    imm = {0, 0};
    err = 0;
    REQUIRE_FALSE(Z29BytecodeDevice::eval_at(
        ops.data(), imm.data(), static_cast<std::uint32_t>(ops.size()), slots.data(), 1, 0, 0, 0,
        stream.data(), stream.size(), 0, stack.data(), 8, &out, &err));
    REQUIRE(err == Z29BytecodeDevice::kErrDomain);

    err = 0;
    REQUIRE_FALSE(Z29BytecodeDevice::eval_at_trusted(
        ops.data(), imm.data(), static_cast<std::uint32_t>(ops.size()), slots.data(), 1, 0, 0, 0,
        stream.data(), stream.size(), 0, stack.data(), 8, &out, &err));
    REQUIRE(err == Z29BytecodeDevice::kErrDomain);
}

TEST_CASE("Z29BytecodeDevice rejects stack overflow and OOB index", "[cuda][bytecode][parity]") {
    std::vector<std::uint8_t> ops{Z29BytecodeDevice::kOpConst, Z29BytecodeDevice::kOpConst};
    std::vector<std::uint8_t> imm{1, 2};
    std::vector<std::uint8_t> slots{0};
    std::vector<std::uint8_t> stream{0};
    std::vector<std::uint8_t> stack(Z29BytecodeDevice::kMaxDeviceStack, 0);
    std::uint8_t out = 0;
    std::uint8_t err = 0;
    // stack_cap=1 → second Const overflows
    REQUIRE_FALSE(Z29BytecodeDevice::eval_at(
        ops.data(), imm.data(), static_cast<std::uint32_t>(ops.size()), slots.data(), 1, 0, 0, 0,
        stream.data(), stream.size(), 0, stack.data(), 1, &out, &err));
    REQUIRE(err == Z29BytecodeDevice::kErrStack);

    ops = {Z29BytecodeDevice::kOpLoad};
    imm = {0};
    err = 0;
    REQUIRE_FALSE(Z29BytecodeDevice::eval_at(
        ops.data(), imm.data(), 1u, slots.data(), 1, 0, 0, 0, stream.data(), stream.size(), 99,
        stack.data(), 8, &out, &err));
    REQUIRE(err == Z29BytecodeDevice::kErrIndex);
}
