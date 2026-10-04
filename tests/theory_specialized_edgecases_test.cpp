#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/batch/batch_runner.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_apply_ir.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_dispatch.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_bytecode.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/theory_export_cache.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"
#include "theory_hist_chi2_launch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

constexpr const char* kSha =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

[[nodiscard]] Status install_theory(const std::filesystem::path& theories_root,
                                    const TheoryIr& theory) {
    TheoryArtifact::Paths paths;
    paths.set_apply_ir(std::string("apply_ir.json"));
    std::vector<TheoryArtifact::Param> params;
    for (const ParamIr& p : theory.params()) {
        params.emplace_back(p.name(), static_cast<std::int64_t>(p.min()),
                            static_cast<std::int64_t>(p.max()));
    }
    StatusOr<TheoryArtifact> art = TheoryArtifact::make(
        theory.name(), 1, theory.tier(), theory.family(), kSha,
        TheoryArtifact::Verification{DslVerifier::Mode::Exhaustive, true, std::nullopt,
                                     "1970-01-01T00:00:00Z"},
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::interrupt_mode_from_theory(theory.interrupt_mode()), std::move(params), {},
        theory.structural_claim(), std::string("inline_edge_test"), std::move(paths));
    if (!art.ok()) {
        return art.status();
    }
    Status stored = art.value().store(theories_root);
    if (!stored.ok()) {
        return stored;
    }
    return TheoryApplyIr::write(art.value().artifact_dir(theories_root) / "apply_ir.json", theory);
}

[[nodiscard]] TheoryIr make_caesar() {
    const StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr s = Z29Expr::var("shift");
    const StatusOr<TheoryIr> th =
        TheoryIr::make("edge_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {shift.value()},
                       Z29Expr::add(x, s), Z29Expr::sub(x, s));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_progressive() {
    const StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
    const StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
    REQUIRE(b0.ok());
    REQUIRE(b1.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr i = Z29Expr::var("i");
    const Z29Expr::Ptr ks =
        Z29Expr::add(Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), i));
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "edge_progressive", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, ks),
        Z29Expr::sub(x, ks), std::string("Edge S2 progressive."));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_autokey() {
    const StatusOr<ParamIr> lag = ParamIr::make("lag", 1, 28);
    REQUIRE(lag.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr prior = Z29Expr::call("z29_autokey_shift", {x, Z29Expr::var("lag")});
    const StatusOr<TheoryIr> th =
        TheoryIr::make("edge_autokey", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
                       TheoryIr::InterruptMode::NoneByDesign, {lag.value()},
                       Z29Expr::add(x, prior), Z29Expr::sub(x, prior),
                       std::string("Autokey S4 AutokeyRing."));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_div0_const() {
    // decrypt always divides by 0 → domain error / +inf on S0.
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<Z29Expr::Ptr> zero = Z29Expr::constant(0);
    REQUIRE(zero.ok());
    const StatusOr<TheoryIr> th =
        TheoryIr::make("edge_div0", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {}, x,
                       Z29Expr::div(x, zero.value()));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_div_by_param() {
    // Mixed grid: d==0 → +inf via S1 bake lane_err patch; d!=0 → finite.
    const StatusOr<ParamIr> d = ParamIr::make("d", 0, 28);
    REQUIRE(d.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr dv = Z29Expr::var("d");
    const StatusOr<TheoryIr> th =
        TheoryIr::make("edge_div_param", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {d.value()},
                       Z29Expr::mul(x, dv), Z29Expr::div(x, dv));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_affine() {
    const StatusOr<ParamIr> a = ParamIr::make("a", 0, 28);
    const StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr av = Z29Expr::var("a");
    const Z29Expr::Ptr bv = Z29Expr::var("b");
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "edge_affine", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {a.value(), b.value()},
        Z29Expr::add(Z29Expr::mul(av, x), bv),
        Z29Expr::mul(Z29Expr::inv(av), Z29Expr::sub(x, bv)),
        std::string("Edge affine decrypt; a=0 is inv domain."));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] std::vector<Index29> make_cipher(std::size_t n) {
    std::vector<Index29> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(Index29{static_cast<std::uint8_t>((i * 7u + 4u) % 29u)});
    }
    return out;
}

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        out[i] = indices[i].value();
    }
    return out;
}

[[nodiscard]] std::vector<double>
launch_bytecode_scores(const TheoryIr& theory, const std::vector<Index29>& cipher,
                       const std::vector<nlohmann::json>& params_list,
                       const ExpectedFrequencyTable& freqs) {
    const StatusOr<Z29Bytecode::Program> prog =
        Z29Bytecode::compile_theory(theory, TransformDirection::Decrypt);
    REQUIRE(prog.ok());
    const std::size_t C = params_list.size();
    const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.value().slot_names.size());
    const std::uint16_t max_stack = prog.value().max_stack == 0 ? 8 : prog.value().max_stack;

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

    const std::vector<std::uint8_t> host_in = to_bytes(cipher);
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
    StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
        std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
    REQUIRE(device_probs.ok());
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
    REQUIRE(device_counts.ok());
    StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
    REQUIRE(device_scores.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_err = DeviceBuffer<std::uint8_t>::allocate(C);
    REQUIRE(device_err.ok());

    REQUIRE(TheoryHistChi2Launch::launch_bytecode_async(
                device_in.value().data(), device_ops.value().data(), device_imm.value().data(),
                static_cast<std::uint32_t>(ops.size()), device_slots.value().data(), slot_count,
                prog.value().cipher_slot, prog.value().index_slot,
                prog.value().binds_index_i ? 1u : 0u, max_stack, device_probs.value().data(),
                device_counts.value().data(), device_scores.value().data(),
                device_err.value().data(), C, host_in.size())
                .ok());
    REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "edge bytecode sync").ok());

    std::vector<double> scores(C, 0.0);
    REQUIRE(device_scores.value().copy_to_host(scores).ok());
    return scores;
}

void require_same_topk_order(const std::vector<double>& a, const std::vector<double>& b,
                             std::size_t k) {
    REQUIRE(a.size() == b.size());
    REQUIRE(k <= a.size());
    std::vector<std::size_t> ia(a.size());
    std::vector<std::size_t> ib(b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        ia[i] = i;
        ib[i] = i;
    }
    auto by_score = [](const std::vector<double>& scores) {
        return [&](std::size_t x, std::size_t y) { return scores[x] < scores[y]; };
    };
    std::stable_sort(ia.begin(), ia.end(), by_score(a));
    std::stable_sort(ib.begin(), ib.end(), by_score(b));
    for (std::size_t r = 0; r < k; ++r) {
        REQUIRE(ia[r] == ib[r]);
        REQUIRE(a[ia[r]] == b[ib[r]]);
    }
}

} // namespace

TEST_CASE("specialized S1 vs bytecode: scores + top-k order match",
          "[cuda][theory][edge][specialized]") {
    REQUIRE(ParcaeCuda::available());

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_theory_edge_s1_topk";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_caesar();
    REQUIRE(install_theory(root / "theories", theory).ok());
    REQUIRE(TheoryHistChi2Emit::emit_decrypt_hist(theory).value().specialized());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(96);
    std::vector<nlohmann::json> params_list;
    for (int shift = 0; shift < 29; ++shift) {
        params_list.push_back(nlohmann::json{{"shift", shift}});
    }

    TheoryExportCache cache;
    StatusOr<std::vector<double>> specialized = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/edge_caesar@1", params_list,
        TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(specialized.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(TheoryHistChi2Emit::emit_decrypt_hist(theory).value().has_shape_caesar_kernel());

    const std::vector<double> bytecode =
        launch_bytecode_scores(theory, cipher, params_list, freqs.value());
    REQUIRE(specialized.value().size() == bytecode.size());
    for (std::size_t c = 0; c < bytecode.size(); ++c) {
        REQUIRE(specialized.value()[c] == bytecode[c]);
    }
    require_same_topk_order(specialized.value(), bytecode, /*k=*/5);

    StatusOr<GpuCandidateExport::Result> topk = GpuCandidateExport::theory_explicit_params(
        cipher, freqs.value(), root / "theories", "parcae://theories/edge_caesar@1", params_list,
        /*k=*/5, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(topk.ok());
    REQUIRE(topk.value().backend() == Backend::Cuda);
    REQUIRE(topk.value().size() == 5);
    for (std::size_t i = 1; i < topk.value().size(); ++i) {
        REQUIRE(topk.value().rows()[i - 1].score() <= topk.value().rows()[i].score());
    }

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("specialized S2 vs bytecode: scores + top-k order match",
          "[cuda][theory][edge][specialized]") {
    REQUIRE(ParcaeCuda::available());

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_theory_edge_s2_topk";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_progressive();
    REQUIRE(install_theory(root / "theories", theory).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(64);
    std::vector<nlohmann::json> params_list;
    for (int b0 = 0; b0 < 4; ++b0) {
        for (int b1 = 0; b1 < 4; ++b1) {
            params_list.push_back(nlohmann::json{{"b0", b0}, {"b1", b1}});
        }
    }

    TheoryExportCache cache;
    StatusOr<std::vector<double>> specialized = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/edge_progressive@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(specialized.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);

    const std::vector<double> bytecode =
        launch_bytecode_scores(theory, cipher, params_list, freqs.value());
    for (std::size_t c = 0; c < bytecode.size(); ++c) {
        REQUIRE(specialized.value()[c] == bytecode[c]);
    }
    require_same_topk_order(specialized.value(), bytecode, /*k=*/4);

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("Autokey theory prefers S4 AutokeyRing; χ² ≡ bytecode",
          "[cuda][theory][edge][autokey][s4]") {
    REQUIRE(ParcaeCuda::available());

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_theory_edge_autokey";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_autokey();
    REQUIRE(install_theory(root / "theories", theory).ok());
    REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
            TheoryHistChi2Emit::Strategy::S4AutokeyRing);
    REQUIRE(TheoryHistChi2Emit::emit_decrypt_hist(theory).value().specialized());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(32);
    std::vector<nlohmann::json> params_list{nlohmann::json{{"lag", 1}}, nlohmann::json{{"lag", 3}},
                                            nlohmann::json{{"lag", 5}}};

    TheoryExportCache cache;
    StatusOr<const TheoryExportCache::Entry*> prepared =
        cache.ensure(root / "theories", "parcae://theories/edge_autokey@1",
                     TransformDirection::Decrypt);
    REQUIRE(prepared.ok());
    REQUIRE(prepared.value()->hist_plan().specialized());
    REQUIRE(prepared.value()->hist_plan().emitted_strategy() ==
            TheoryHistChi2Emit::Strategy::S4AutokeyRing);
    REQUIRE(prepared.value()->hist_plan().s4_autokey().has_value());

    StatusOr<std::vector<double>> gpu = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/edge_autokey@1", params_list,
        TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S4AutokeyRing);

    const std::vector<double> bytecode =
        launch_bytecode_scores(theory, cipher, params_list, freqs.value());
    for (std::size_t c = 0; c < bytecode.size(); ++c) {
        REQUIRE(gpu.value()[c] == bytecode[c]);
    }

    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain =
            TheoryDispatch::apply(theory, cipher, params_list[c], TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu.value()[c] == cpu.value());
    }

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("domain error Div0 patches fused theory score to +inf",
          "[cuda][theory][edge][domain]") {
    REQUIRE(ParcaeCuda::available());

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_theory_edge_div0";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());
    const std::vector<Index29> cipher = make_cipher(16);

    SECTION("constant div0 theory (S1 device bake → +inf)") {
        const TheoryIr theory = make_div0_const();
        REQUIRE(install_theory(root / "theories", theory).ok());
        REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
                TheoryHistChi2Emit::Strategy::S1Lut29);

        TheoryExportCache cache;
        std::vector<nlohmann::json> params_list{nlohmann::json::object()};
        StatusOr<std::vector<double>> scores = GpuCandidateExport::theory_scores_only(
            cipher, freqs.value(), root / "theories", "parcae://theories/edge_div0@1", params_list,
            TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
        REQUIRE(scores.ok());
        REQUIRE(scores.value().size() == 1);
        REQUIRE(std::isinf(scores.value()[0]));
        REQUIRE(scores.value()[0] > 0.0);
        // Residual FxOnly stays on S1; bake lane_err patches +inf.
        REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S1Lut29);
    }

    SECTION("param d=0 lane is +inf; d!=0 stays finite") {
        const TheoryIr theory = make_div_by_param();
        REQUIRE(install_theory(root / "theories", theory).ok());

        TheoryExportCache cache;
        std::vector<nlohmann::json> params_list{nlohmann::json{{"d", 1}}, nlohmann::json{{"d", 0}},
                                                nlohmann::json{{"d", 2}}};
        StatusOr<std::vector<double>> scores = GpuCandidateExport::theory_scores_only(
            cipher, freqs.value(), root / "theories", "parcae://theories/edge_div_param@1",
            params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
        REQUIRE(scores.ok());
        REQUIRE(scores.value().size() == 3);
        REQUIRE(std::isfinite(scores.value()[0]));
        REQUIRE(std::isinf(scores.value()[1]));
        REQUIRE(scores.value()[1] > 0.0);
        REQUIRE(std::isfinite(scores.value()[2]));
        REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S1Lut29);
    }

    SECTION("affine a=0 inv domain soft-falls ShapeInline → S0 (+inf lane)") {
        const TheoryIr theory = make_affine();
        REQUIRE(install_theory(root / "theories", theory).ok());
        REQUIRE(TheoryHistChi2Emit::emit_decrypt_hist(theory).value().has_shape_affine_kernel());

        TheoryExportCache cache;
        // Mix: invertible a then a==0 — shape path rejects domain → soft S0.
        std::vector<nlohmann::json> params_list{nlohmann::json{{"a", 1}, {"b", 0}},
                                                nlohmann::json{{"a", 0}, {"b", 0}},
                                                nlohmann::json{{"a", 2}, {"b", 1}}};
        StatusOr<std::vector<double>> scores = GpuCandidateExport::theory_scores_only(
            cipher, freqs.value(), root / "theories", "parcae://theories/edge_affine@1",
            params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
        REQUIRE(scores.ok());
        REQUIRE(scores.value().size() == 3);
        REQUIRE(std::isfinite(scores.value()[0]));
        REQUIRE(std::isinf(scores.value()[1]));
        REQUIRE(scores.value()[1] > 0.0);
        REQUIRE(std::isfinite(scores.value()[2]));
        REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    }

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("fused theory export rejects non-empty interrupt",
          "[cuda][theory][edge][interrupt]") {
    REQUIRE(ParcaeCuda::available());

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_theory_edge_irq";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_caesar();
    REQUIRE(install_theory(root / "theories", theory).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(24);
    std::vector<nlohmann::json> params_list{nlohmann::json{{"shift", 0}},
                                            nlohmann::json{{"shift", 1}}};
    StatusOr<InterruptPolicy> irq = InterruptPolicy::from_skip_indices({0});
    REQUIRE(irq.ok());

    REQUIRE_FALSE(GpuCandidateExport::theory_explicit_params(
                      cipher, freqs.value(), root / "theories", "parcae://theories/edge_caesar@1",
                      params_list, /*k=*/1, TransformDirection::Decrypt, BatchRunner::Progress{},
                      irq.value())
                      .ok());
    REQUIRE_FALSE(GpuCandidateExport::theory_scores_only(
                      cipher, freqs.value(), root / "theories", "parcae://theories/edge_caesar@1",
                      params_list, TransformDirection::Decrypt, BatchRunner::Progress{},
                      irq.value())
                      .ok());

    // Empty interrupt still accepted on specialized path.
    StatusOr<GpuCandidateExport::Result> ok = GpuCandidateExport::theory_explicit_params(
        cipher, freqs.value(), root / "theories", "parcae://theories/edge_caesar@1", params_list,
        /*k=*/1, TransformDirection::Decrypt, BatchRunner::Progress{}, InterruptPolicy::none());
    REQUIRE(ok.ok());
    REQUIRE(ok.value().backend() == Backend::Cuda);

    std::filesystem::remove_all(root, ec);
}

#else

TEST_CASE("theory specialized edgecases skipped without CUDA",
          "[cuda][theory][edge][specialized]") {
    SUCCEED("PARCAE_HAS_CUDA unset");
}

#endif
