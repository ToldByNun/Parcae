#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/score/chi2_english_gp.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"
#include "theory_chi2_batch.hpp"
#include "z29_bytecode_device.hpp"

#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] TheoryIr make_caesar_theory() {
    const StatusOr<ParamIr> shift_p = ParamIr::make("shift", 0, 28);
    REQUIRE(shift_p.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr shift = Z29Expr::var("shift");
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("tchi2_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
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
        "tchi2_progressive", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, s),
        Z29Expr::sub(x, s), std::string("TheoryChi2Batch progressive."));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        out[i] = indices[i].value();
    }
    return out;
}

struct Packed {
    std::vector<std::uint8_t> ops;
    std::vector<std::uint8_t> imm;
    std::vector<std::uint8_t> slots; // C * S
    std::uint16_t slot_count = 0;
    std::uint16_t cipher_slot = 0;
    std::uint16_t index_slot = 0;
    std::uint8_t binds_index_i = 0;
    std::uint16_t max_stack = 0;
    std::size_t C = 0;
};

[[nodiscard]] Packed pack_grid(const Z29Bytecode::Program& prog, const TheoryIr& theory,
                               const std::vector<nlohmann::json>& params_list) {
    Packed out;
    out.C = params_list.size();
    out.slot_count = static_cast<std::uint16_t>(prog.slot_names.size());
    out.cipher_slot = prog.cipher_slot;
    out.index_slot = prog.index_slot;
    out.binds_index_i = prog.binds_index_i ? 1u : 0u;
    out.max_stack = prog.max_stack == 0 ? 8 : prog.max_stack;
    REQUIRE(out.max_stack <= TheoryChi2Batch::kMaxDeviceStack);
    REQUIRE(out.slot_count <= TheoryChi2Batch::kMaxSlots);

    out.ops.reserve(prog.ops.size());
    for (Z29Bytecode::Op op : prog.ops) {
        out.ops.push_back(Z29Bytecode::op_as_u8(op));
    }
    out.imm = prog.imm;
    out.slots.assign(out.C * out.slot_count, 0);

    for (std::size_t c = 0; c < out.C; ++c) {
        StatusOr<std::vector<Index29>> bound =
            Z29Bytecode::bind_theory_slots(prog, theory, params_list[c]);
        REQUIRE(bound.ok());
        REQUIRE(bound.value().size() == out.slot_count);
        for (std::uint16_t s = 0; s < out.slot_count; ++s) {
            out.slots[c * out.slot_count + s] = bound.value()[s].value();
        }
    }
    return out;
}

[[nodiscard]] std::vector<double> launch_scores(const Packed& packed,
                                                const std::vector<std::uint8_t>& host_in,
                                                const ExpectedFrequencyTable& freqs) {
    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_ops =
        DeviceBuffer<std::uint8_t>::from_host(packed.ops);
    REQUIRE(device_ops.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_imm =
        DeviceBuffer<std::uint8_t>::from_host(packed.imm);
    REQUIRE(device_imm.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_slots =
        DeviceBuffer<std::uint8_t>::from_host(packed.slots);
    REQUIRE(device_slots.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(packed.C * TheoryChi2Batch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(packed.C);
    REQUIRE(device_scores.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_err =
        DeviceBuffer<std::uint8_t>::allocate(packed.C);
    REQUIRE(device_err.ok());

    REQUIRE(TheoryChi2Batch::launch_async(
                device_in.value().data(), device_ops.value().data(), device_imm.value().data(),
                static_cast<std::uint32_t>(packed.ops.size()), device_slots.value().data(),
                packed.slot_count, packed.cipher_slot, packed.index_slot, packed.binds_index_i,
                packed.max_stack, device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), device_err.value().data(), packed.C, host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "TheoryChi2Batch sync").ok());

    std::vector<double> scores(packed.C, 0.0);
    REQUIRE(device_scores.value().copy_to_host(scores).ok());
    return scores;
}

} // namespace

TEST_CASE("TheoryChi2Batch caesar grid matches CPU Chi2EnglishGp",
          "[cuda][batch][chi2][theory]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const TheoryIr theory = make_caesar_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());

    std::vector<Index29> cipher;
    cipher.reserve(64);
    for (std::uint8_t i = 0; i < 64; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>((i * 3u + 5u) % 29u)});
    }

    std::vector<nlohmann::json> params_list;
    params_list.reserve(29);
    for (int shift = 0; shift < 29; ++shift) {
        params_list.push_back(nlohmann::json{{"shift", shift}});
    }

    const Packed packed = pack_grid(prog.value(), theory, params_list);
    const std::vector<double> gpu = launch_scores(packed, to_bytes(cipher), freqs.value());
    REQUIRE(gpu.size() == 29);

    for (int shift = 0; shift < 29; ++shift) {
        StatusOr<std::vector<Index29>> plain =
            Z29Bytecode::apply_theory(prog.value(), theory, params_list[static_cast<std::size_t>(shift)],
                                      cipher);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu[static_cast<std::size_t>(shift)] == cpu.value());
    }
}

TEST_CASE("TheoryChi2Batch progressive grid matches CPU Chi2EnglishGp",
          "[cuda][batch][chi2][theory]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const TheoryIr theory = make_progressive_theory();
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().binds_index_i);

    std::vector<Index29> cipher;
    for (std::uint8_t i = 0; i < 48; ++i) {
        cipher.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 2u) % 29u)});
    }

    // Small 3x3 grid (b0,b1 in {0,1,2}) — enough for binds_index_i coverage.
    std::vector<nlohmann::json> params_list;
    for (int b0 = 0; b0 < 3; ++b0) {
        for (int b1 = 0; b1 < 3; ++b1) {
            params_list.push_back(nlohmann::json{{"b0", b0}, {"b1", b1}});
        }
    }

    const Packed packed = pack_grid(prog.value(), theory, params_list);
    const std::vector<double> gpu = launch_scores(packed, to_bytes(cipher), freqs.value());
    REQUIRE(gpu.size() == params_list.size());

    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain =
            Z29Bytecode::apply_theory(prog.value(), theory, params_list[c], cipher);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu[c] == cpu.value());
    }
}

TEST_CASE("TheoryChi2Batch domain error patches score to +inf", "[cuda][batch][chi2][theory]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    // Program: Load cipher, Const 0, Div → always div0.
    std::vector<std::uint8_t> ops{Z29BytecodeDevice::kOpLoad, Z29BytecodeDevice::kOpConst,
                                  Z29BytecodeDevice::kOpDiv};
    std::vector<std::uint8_t> imm{0, 0, 0};
    std::vector<std::uint8_t> slots{0}; // one candidate, one cipher slot
    std::vector<std::uint8_t> host_in{1, 2, 3, 4};

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    REQUIRE(device_in.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_ops = DeviceBuffer<std::uint8_t>::from_host(ops);
    REQUIRE(device_ops.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_imm = DeviceBuffer<std::uint8_t>::from_host(imm);
    REQUIRE(device_imm.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_slots =
        DeviceBuffer<std::uint8_t>::from_host(slots);
    REQUIRE(device_slots.ok());
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.value().probabilities().data(), 29));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(TheoryChi2Batch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(1);
    REQUIRE(device_scores.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_err = DeviceBuffer<std::uint8_t>::allocate(1);
    REQUIRE(device_err.ok());

    REQUIRE(TheoryChi2Batch::launch_async(
                device_in.value().data(), device_ops.value().data(), device_imm.value().data(),
                static_cast<std::uint32_t>(ops.size()), device_slots.value().data(),
                /*slot_count=*/1, /*cipher_slot=*/0, /*index_slot=*/0, /*binds_index_i=*/0,
                /*max_stack=*/8, device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), device_err.value().data(), /*C=*/1, host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "TheoryChi2Batch inf sync").ok());

    std::vector<double> scores(1, 0.0);
    REQUIRE(device_scores.value().copy_to_host(scores).ok());
    REQUIRE(std::isinf(scores[0]));
    REQUIRE(scores[0] > 0.0);

    std::vector<std::uint8_t> err(1, 0);
    REQUIRE(device_err.value().copy_to_host(err).ok());
    REQUIRE(err[0] == 1);
}

TEST_CASE("TheoryChi2Batch rejects bad caps", "[cuda][batch][chi2][theory]") {
    REQUIRE(ParcaeCuda::available());
    std::uint8_t dummy_u8 = 0;
    double dummy_d = 0.0;
    std::uint32_t dummy_c = 0;
    REQUIRE_FALSE(TheoryChi2Batch::launch_async(
                      &dummy_u8, &dummy_u8, &dummy_u8, TheoryChi2Batch::kMaxProgramOps + 1,
                      &dummy_u8, 1, 0, 0, 0, 8, &dummy_d, &dummy_c, &dummy_d, &dummy_u8, 1, 4)
                      .ok());
    REQUIRE_FALSE(TheoryChi2Batch::launch_async(&dummy_u8, &dummy_u8, &dummy_u8, 1, &dummy_u8,
                                                TheoryChi2Batch::kMaxSlots + 1, 0, 0, 0, 8,
                                                &dummy_d, &dummy_c, &dummy_d, &dummy_u8, 1, 4)
                      .ok());
}

#else

TEST_CASE("TheoryChi2Batch skipped (PARCAE_HAS_CUDA unset)", "[cuda][batch][chi2][theory]") {
    SUCCEED("CUDA not built");
}

#endif
