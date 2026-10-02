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

[[nodiscard]] std::vector<Index29> make_cipher(std::size_t n) {
    std::vector<Index29> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(Index29{static_cast<std::uint8_t>((i * 5u + 3u) % 29u)});
    }
    return out;
}

} // namespace

TEST_CASE("GpuCandidateExport prefers S1 LUT hist; export_backend=cuda",
          "[search][export][theory][specialized][cuda]") {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_s1";
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
            TheoryHistChi2Emit::Strategy::S1Lut29);

    StatusOr<GpuCandidateExport::Result> exported = GpuCandidateExport::theory_explicit_params(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_caesar@1",
        params_list, /*k=*/3, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(exported.ok());
    REQUIRE(exported.value().backend() == Backend::Cuda);
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S1Lut29);
    REQUIRE(exported.value().size() == 3);

    // Parity vs CPU χ² ordering for top-1.
    StatusOr<std::vector<double>> gpu_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", "parcae://theories/export_prefer_caesar@1",
        params_list, TransformDirection::Decrypt, {}, InterruptPolicy::none(), &cache);
    REQUIRE(gpu_scores.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S1Lut29);

    for (std::size_t c = 0; c < params_list.size(); ++c) {
        StatusOr<std::vector<Index29>> plain = TheoryDispatch::apply(
            theory, cipher, params_list[c], TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
        REQUIRE(cpu.ok());
        REQUIRE(gpu_scores.value()[c] == cpu.value());
    }
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
