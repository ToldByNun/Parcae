#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/core/index29.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_apply_ir.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_dispatch.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_shape_match.hpp>
#include <parcae/dsl/theory_uri.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/theory_export_cache.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
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
        theory.structural_claim(), std::string("inline_test"), std::move(paths));
    if (!art.ok()) {
        return art.status();
    }
    Status stored = art.value().store(theories_root);
    if (!stored.ok()) {
        return stored;
    }
    const std::filesystem::path dir = art.value().artifact_dir(theories_root);
    return TheoryApplyIr::write(dir / "apply_ir.json", theory);
}

[[nodiscard]] TheoryIr make_caesar() {
    const StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr s = Z29Expr::var("shift");
    const StatusOr<TheoryIr> th =
        TheoryIr::make("export_prefer_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {shift.value()},
                       Z29Expr::add(x, s), Z29Expr::sub(x, s));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_atbash_arith() {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr twenty_eight = Z29Expr::constant(28).value();
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "export_prefer_atbash_arith", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {}, Z29Expr::sub(twenty_eight, x),
        Z29Expr::sub(twenty_eight, x), std::string("Self-written Atbash arith."));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_affine_decrypt() {
    const StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
    const StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr av = Z29Expr::var("a");
    const Z29Expr::Ptr bv = Z29Expr::var("b");
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "export_prefer_affine", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {a.value(), b.value()},
        Z29Expr::add(Z29Expr::mul(av, x), bv),
        Z29Expr::mul(Z29Expr::inv(av), Z29Expr::sub(x, bv)),
        std::string("Affine decrypt ShapeInline."));
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
        "export_prefer_progressive", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {b0.value(), b1.value()}, Z29Expr::add(x, ks),
        Z29Expr::sub(x, ks), std::string("S2 export prefer progressive."));
    REQUIRE(th.ok());
    return th.value();
}

/// Residual FxOnly: mul(x,x) — not Affine/Caesar; forces S1 device LUT bake.
[[nodiscard]] TheoryIr make_fx_square() {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr body = Z29Expr::mul(x, x);
    const StatusOr<TheoryIr> th =
        TheoryIr::make("export_prefer_fx_square", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {}, body, body,
                       std::string("FxOnly residual S1 bake."));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] std::vector<Index29> make_cipher(std::size_t n) {
    std::vector<Index29> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 3u) % 29u)});
    }
    return out;
}

} // namespace

TEST_CASE("GpuCandidateExport prefers ShapeInline Caesar twin; export_backend=cuda",
          "[search][export][theory][specialized][cuda][shape]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_shape";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_caesar();
    REQUIRE(install_theory(root / "theories", theory).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(64);
    std::vector<nlohmann::json> params_list;
    for (int shift = 0; shift < 8; ++shift) {
        params_list.push_back(nlohmann::json{{"shift", shift}});
    }

    TheoryExportCache cache;
    StatusOr<const TheoryExportCache::Entry*> prepared = cache.ensure(
        root / "theories", "parcae://theories/export_prefer_caesar@1", TransformDirection::Decrypt);
    REQUIRE(prepared.ok());
    REQUIRE(prepared.value()->hist_plan().specialized());
    REQUIRE(prepared.value()->hist_plan().emitted_strategy() ==
            TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(prepared.value()->hist_plan().has_shape_caesar_kernel());
    REQUIRE_FALSE(prepared.value()->hist_plan().has_s1_soft_path());
    REQUIRE(prepared.value()->hist_plan().shape().has_value());
    REQUIRE(prepared.value()->hist_plan().shape()->shape() == TheoryShapeMatch::ShapeId::Caesar);

    StatusOr<GpuCandidateExport::Result> exported = GpuCandidateExport::theory_explicit_params(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_caesar@1",
        params_list, /*k=*/3, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().backend() == Backend::Cuda);
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::ShapeInline);
    REQUIRE(exported.value().size() == 3);

    // Parity vs CPU χ² ordering for top-1.
    StatusOr<std::vector<double>> gpu_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_caesar@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu_scores.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::ShapeInline);

    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain = TheoryDispatch::apply(
            theory, cipher, params_list[c], TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu_scores.value()[c] == cpu.value());
    }
}

TEST_CASE("GpuCandidateExport prefers ShapeInline Affine decrypt twin",
          "[search][export][theory][specialized][cuda][shape]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_affine";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_affine_decrypt();
    REQUIRE(install_theory(root / "theories", theory).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(48);
    std::vector<nlohmann::json> params_list;
    for (int a = 1; a <= 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            params_list.push_back(nlohmann::json{{"a", a}, {"b", b}});
        }
    }

    TheoryExportCache cache;
    StatusOr<const TheoryExportCache::Entry*> prepared = cache.ensure(
        root / "theories", "parcae://theories/export_prefer_affine@1", TransformDirection::Decrypt);
    REQUIRE(prepared.ok());
    REQUIRE(prepared.value()->hist_plan().has_shape_affine_kernel());

    StatusOr<std::vector<double>> gpu_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_affine@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu_scores.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::ShapeInline);
    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain =
            TheoryDispatch::apply(theory, cipher, params_list[c], TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu_scores.value()[c] == cpu.value());
    }
}

TEST_CASE("GpuCandidateExport prefers ShapeInline Atbash hist twin",
          "[search][export][theory][specialized][cuda][shape]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_atbash";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_atbash_arith();
    REQUIRE(install_theory(root / "theories", theory).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(64);
    std::vector<nlohmann::json> params_list{nlohmann::json::object(), nlohmann::json::object(),
                                            nlohmann::json::object()};

    TheoryExportCache cache;
    StatusOr<const TheoryExportCache::Entry*> prepared =
        cache.ensure(root / "theories", "parcae://theories/export_prefer_atbash_arith@1",
                     TransformDirection::Decrypt);
    REQUIRE(prepared.ok());
    REQUIRE(prepared.value()->hist_plan().specialized());
    REQUIRE(prepared.value()->hist_plan().has_shape_atbash_kernel());
    REQUIRE(prepared.value()->hist_plan().emitted_strategy() ==
            TheoryHistChi2Emit::Strategy::ShapeInline);

    StatusOr<std::vector<double>> gpu_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_atbash_arith@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu_scores.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::ShapeInline);

    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain =
            TheoryDispatch::apply(theory, cipher, params_list[c], TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu_scores.value()[c] == cpu.value());
    }
}

TEST_CASE("GpuCandidateExport prefers S1 device-bake for residual FxOnly; χ² parity",
          "[search][export][theory][specialized][cuda][s1]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_s1_fx";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_fx_square();
    REQUIRE(install_theory(root / "theories", theory).ok());
    REQUIRE(TheoryHistChi2Emit::select_strategy(theory).strategy() ==
            TheoryHistChi2Emit::Strategy::S1Lut29);

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(96);
    // Param-free FxOnly — one lane is enough for parity; pad a few identical rows.
    std::vector<nlohmann::json> params_list(8, nlohmann::json::object());

    TheoryExportCache cache;
    StatusOr<std::vector<double>> gpu_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_fx_square@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu_scores.ok());
    REQUIRE(gpu_scores.value().size() == params_list.size());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S1Lut29);

    StatusOr<std::vector<Index29>> plain =
        TheoryDispatch::apply(theory, cipher, params_list[0], TransformDirection::Decrypt);
    REQUIRE(plain.ok());
    StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
    REQUIRE(cpu.ok());
    for (double s : gpu_scores.value()) {
        REQUIRE(s == cpu.value());
    }
}

/// Name/catalog-irrelevant vigenere_lag custom (not `autokey_lag` catalog id).
[[nodiscard]] TheoryIr make_vigenere_lag_custom() {
    const StatusOr<ParamIr> lag = ParamIr::make("delay", 1, 28);
    REQUIRE(lag.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr prior = Z29Expr::call("z29_autokey_shift", {x, Z29Expr::var("delay")});
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "export_prefer_lag_stream", TheoryIr::Family::KeyedStream, TheoryIr::Tier::B,
        TheoryIr::InterruptMode::NoneByDesign, {lag.value()}, Z29Expr::add(x, prior),
        Z29Expr::sub(x, prior), std::string("Custom vigenere_lag without catalog API."));
    REQUIRE(th.ok());
    return th.value();
}

TEST_CASE("GpuCandidateExport prefers S4 AutokeyRing for custom vigenere_lag",
          "[search][export][theory][specialized][cuda][s4]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_s4";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_vigenere_lag_custom();
    REQUIRE(install_theory(root / "theories", theory).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(48);
    std::vector<nlohmann::json> params_list;
    for (int lag = 1; lag <= 6; ++lag) {
        params_list.push_back(nlohmann::json{{"delay", lag}});
    }

    TheoryExportCache cache;
    StatusOr<const TheoryExportCache::Entry*> prepared = cache.ensure(
        root / "theories", "parcae://theories/export_prefer_lag_stream@1",
        TransformDirection::Decrypt);
    REQUIRE(prepared.ok());
    REQUIRE(prepared.value()->hist_plan().emitted_strategy() ==
            TheoryHistChi2Emit::Strategy::S4AutokeyRing);

    StatusOr<std::vector<double>> gpu_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_lag_stream@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu_scores.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S4AutokeyRing);

    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain =
            TheoryDispatch::apply(theory, cipher, params_list[c], TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu_scores.value()[c] == cpu.value());
    }

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("GpuCandidateExport prefers S2 linear hist; soft S0 for non-specialized",
          "[search][export][theory][specialized][cuda]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_s2";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr progressive = make_progressive();
    REQUIRE(install_theory(root / "theories", progressive).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> cipher = make_cipher(48);
    std::vector<nlohmann::json> params_list;
    for (int b0 = 0; b0 < 3; ++b0) {
        for (int b1 = 0; b1 < 3; ++b1) {
            params_list.push_back(nlohmann::json{{"b0", b0}, {"b1", b1}});
        }
    }

    TheoryExportCache cache;
    StatusOr<GpuCandidateExport::Result> exported = GpuCandidateExport::theory_explicit_params(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_progressive@1",
        params_list, /*k=*/2, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().backend() == Backend::Cuda);
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline);

    StatusOr<std::vector<double>> gpu_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_progressive@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu_scores.ok());
    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain = TheoryDispatch::apply(
            progressive, cipher, params_list[c], TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu_scores.value()[c] == cpu.value());
    }
}

#else

TEST_CASE("GpuCandidateExport specialized prefer skipped without CUDA",
          "[search][export][theory][specialized][cuda]") {
    SUCCEED("PARCAE_HAS_CUDA unset");
}

#endif
