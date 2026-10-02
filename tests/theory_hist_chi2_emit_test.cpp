#include <catch2/catch_test_macros.hpp>

#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_ir.hpp>
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
        std::string("S1 candidate affine."));
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

TEST_CASE("TheoryHistChi2Emit S2 non-linear keystream falls back to S0",
          "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_nonlinear_s2_theory();
    REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
            TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    REQUIRE_FALSE(TheoryHistChi2Emit::match_s2_linear(theory).has_value());

    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(bundle.value().specialized());
    REQUIRE(bundle.value().reason().find("not linear") != std::string::npos);
}

TEST_CASE("TheoryHistChi2Emit S1 caesar emits LUT-29 sources", "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_caesar_theory();
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S1Lut29);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S1Lut29);
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s1_lut().has_value());
    REQUIRE(bundle.value().s1_lut()->param_count() == 1);
    REQUIRE(bundle.value().s1_lut()->param_names()[0] == "shift");
    REQUIRE(bundle.value().kernel_symbol() == "emit_caesar_s1_hist_kernel");
    REQUIRE(bundle.value().cu_text().find("lut[HistFast::alphabet]") != std::string::npos);
    REQUIRE(bundle.value().cu_text().find("uchar4") != std::string::npos);
    REQUIRE(bundle.value().cu_text().find("namespace {") == std::string::npos);
    REQUIRE(bundle.value().header_text().find("TheoryHistChi2S1") != std::string::npos);
}

TEST_CASE("TheoryHistChi2Emit S1 affine emits LUT-29 sources", "[dsl][emit][hist][chi2]") {
    const TheoryIr theory = make_affine_theory();
    REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
            TheoryHistChi2Emit::Strategy::S1Lut29);
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(theory);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().specialized());
    REQUIRE(bundle.value().s1_lut().has_value());
    REQUIRE(bundle.value().s1_lut()->param_count() == 2);
    REQUIRE(bundle.value().cu_text().find("Z29Device::inv") != std::string::npos);
    REQUIRE(bundle.value().cu_text().find("emit_affine_s1_hist_kernel") != std::string::npos);
}

TEST_CASE("TheoryHistChi2Emit S3 still soft-falls back to S0", "[dsl][emit][hist][chi2]") {
    StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
        TheoryHistChi2Emit::emit_decrypt_hist(make_complex_i_theory());
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().intended_strategy() == TheoryHistChi2Emit::Strategy::S3ScalarInline);
    REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE_FALSE(bundle.value().specialized());
    REQUIRE(bundle.value().reason().find("skeleton") != std::string::npos);
}

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
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

TEST_CASE("TheoryHistChi2Emit S1 effective_strategy is specialized",
          "[dsl][emit][hist][chi2][cuda]") {
    REQUIRE(TheoryHistChi2Launch::effective_strategy(
                TheoryHistChi2Emit::emit_decrypt_hist(make_caesar_theory()).value()) ==
            TheoryHistChi2Emit::Strategy::S1Lut29);
}

TEST_CASE("TheoryHistChi2 S1 LUT golden: bytecode χ² == specialized χ² (caesar+affine)",
          "[dsl][emit][hist][chi2][cuda][golden]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    auto run_parity = [&](const TheoryIr& theory, const std::vector<nlohmann::json>& params_list) {
        StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
            TheoryHistChi2Emit::emit_decrypt_hist(theory);
        REQUIRE(bundle.ok());
        REQUIRE(bundle.value().specialized());
        REQUIRE(bundle.value().emitted_strategy() == TheoryHistChi2Emit::Strategy::S1Lut29);
        REQUIRE(bundle.value().s1_lut().has_value());

        const StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
        REQUIRE(prog.ok());
        REQUIRE_FALSE(prog.value().binds_index_i);

        std::vector<Index29> cipher;
        for (std::uint8_t i = 0; i < 64; ++i) {
            cipher.push_back(Index29{static_cast<std::uint8_t>((i * 3u + 5u) % 29u)});
        }
        const std::vector<std::uint8_t> host_in = to_bytes(cipher);
        const std::size_t C = params_list.size();

        // Host-fill C×29 LUTs via bytecode eval (same oracle as CPU apply).
        std::vector<std::uint8_t> host_luts(C * TheoryHistChi2Launch::alphabet_size, 0);
        const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.value().slot_names.size());
        for (std::size_t c = 0; c < C; ++c) {
            StatusOr<std::vector<Index29>> bound =
                Z29Bytecode::bind_theory_slots(prog.value(), theory, params_list[c]);
            REQUIRE(bound.ok());
            std::vector<Index29> slots = bound.value();
            for (std::uint8_t sym = 0; sym < 29; ++sym) {
                const std::vector<Index29> one{Index29{sym}};
                StatusOr<Index29> out =
                    Z29Bytecode::eval_at(prog.value(), std::span<Index29>(slots),
                                         std::span<const Index29>(one), 0);
                REQUIRE(out.ok());
                host_luts[c * 29u + sym] = out.value().value();
            }
        }

        std::vector<std::uint8_t> ops;
        ops.reserve(prog.value().ops.size());
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
        StatusOr<DeviceBuffer<std::uint8_t>> device_luts =
            DeviceBuffer<std::uint8_t>::from_host(host_luts);
        REQUIRE(device_luts.ok());
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.value().probabilities().data(),
                                    freqs.value().probabilities().size()));
        REQUIRE(device_probs.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts_bc =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(device_counts_bc.ok());
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts_s1 =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        REQUIRE(device_counts_s1.ok());
        StatusOr<DeviceBuffer<double>> device_scores_bc = DeviceBuffer<double>::allocate(C);
        REQUIRE(device_scores_bc.ok());
        StatusOr<DeviceBuffer<double>> device_scores_s1 = DeviceBuffer<double>::allocate(C);
        REQUIRE(device_scores_s1.ok());
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
                    device_counts_bc.value().data(), device_scores_bc.value().data(),
                    device_err.value().data(), C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S1 golden bytecode sync").ok());

        REQUIRE(TheoryHistChi2Launch::launch_s1_lut_async(
                    device_in.value().data(), device_luts.value().data(),
                    device_probs.value().data(), device_counts_s1.value().data(),
                    device_scores_s1.value().data(), C, host_in.size())
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "S1 golden specialized sync").ok());

        std::vector<double> scores_bc(C, 0.0);
        std::vector<double> scores_s1(C, 0.0);
        REQUIRE(device_scores_bc.value().copy_to_host(scores_bc).ok());
        REQUIRE(device_scores_s1.value().copy_to_host(scores_s1).ok());
        for (std::size_t c = 0; c < C; ++c) {
            REQUIRE(scores_bc[c] == scores_s1[c]);
        }
    };

    {
        std::vector<nlohmann::json> caesar_grid;
        for (int shift = 0; shift < 29; ++shift) {
            caesar_grid.push_back(nlohmann::json{{"shift", shift}});
        }
        run_parity(make_caesar_theory(), caesar_grid);
    }
    {
        // Small affine grid (a∈{1,2,3}, b∈{0,1,2}) — Affine-ähnliche Parity.
        std::vector<nlohmann::json> affine_grid;
        for (int a = 1; a <= 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                affine_grid.push_back(nlohmann::json{{"a", a}, {"b", b}});
            }
        }
        run_parity(make_affine_theory(), affine_grid);
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

#endif
