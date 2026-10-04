#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include <parcae/core/index29.hpp>
#include <parcae/core/sha256.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_apply_ir.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_dispatch.hpp>
#include <parcae/dsl/theory_hist_chi2_emit.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/chi2_english_gp.hpp>
#include <parcae/score/expected_frequency_loader.hpp>
#include <parcae/search/gpu_candidate_export.hpp>
#include <parcae/search/theory_export_cache.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>

#include "parcae_cuda.hpp"
#include "theory_hist_chi2_launch.hpp"
#include "theory_hist_module.hpp"

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
        theory.structural_claim(), std::string("inline_module_test"), std::move(paths));
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
        TheoryIr::make("module_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {shift.value()},
                       Z29Expr::add(x, s), Z29Expr::sub(x, s));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] std::vector<Index29> make_cipher(std::size_t n) {
    std::vector<Index29> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(Index29{static_cast<std::uint8_t>((i * 7u + 1u) % 29u)});
    }
    return out;
}

[[nodiscard]] std::string proxy_digest(std::string_view label) {
    return Sha256::hex_digest(label);
}

} // namespace

TEST_CASE("TheoryHistModule URI+digest cache; has_specialized; digest mismatch",
          "[cuda][theory][module][dsl]") {
    TheoryHistModule::clear();
    const std::string uri = "parcae://theories/module_caesar@1";
    const std::string dig = proxy_digest("proxy-a");

    REQUIRE_FALSE(TheoryHistChi2Launch::has_specialized(uri));
    REQUIRE(TheoryHistModule::ensure_proxy(uri, dig).ok());
    REQUIRE(TheoryHistChi2Launch::has_specialized(uri));
    REQUIRE(TheoryHistModule::has(uri, dig));
    REQUIRE_FALSE(TheoryHistModule::has(uri, proxy_digest("other")));
    REQUIRE(TheoryHistModule::cache_size() == 1);

    std::vector<std::uint8_t> junk{0xDE, 0xAD, 0xBE, 0xEF};
    const std::string bad_dig = Sha256::hex_digest(junk.data(), junk.size());
    Status mismatch = TheoryHistModule::ensure_cubin(uri, proxy_digest("wrong"), junk);
    REQUIRE_FALSE(mismatch.ok());
    REQUIRE(mismatch.message().find("digest mismatch") != std::string::npos);

    Status invalid = TheoryHistModule::ensure_cubin(uri + "/bad", bad_dig, junk);
    REQUIRE_FALSE(invalid.ok());
    REQUIRE_FALSE(TheoryHistModule::has(uri + "/bad"));

    TheoryHistModule::clear();
    REQUIRE_FALSE(TheoryHistChi2Launch::has_specialized(uri));
    REQUIRE(TheoryHistModule::cache_size() == 0);
}

TEST_CASE("TheoryHistModule NVRTC load by source digest (proxy launch)",
          "[cuda][theory][module][nvrtc]") {
    TheoryHistModule::clear();
    const std::string uri = "parcae://theories/module_nvrtc@1";
    const char* src = R"CUDA(
extern "C" __global__ void parcae_theory_hist_module() {}
)CUDA";
    const std::string dig = Sha256::hex_digest(src);
    Status st = TheoryHistModule::ensure_nvrtc(uri, dig, src, TheoryHistModule::kDefaultKernelSymbol,
                                               /*proxy_launch=*/true);
    INFO(st.message());
    REQUIRE(st.ok());
    REQUIRE(TheoryHistModule::has(uri, dig));
    REQUIRE(TheoryHistChi2Launch::has_specialized(uri));
    TheoryHistModule::clear();
}

TEST_CASE("GpuCandidateExport prefers module; soft S0 on invalid/OOM",
          "[search][export][theory][module][cuda]") {
    TheoryHistModule::clear();

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "parcae_export_prefer_module";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const TheoryIr theory = make_caesar();
    REQUIRE(install_theory(root / "theories", theory).ok());

    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::string uri = "parcae://theories/module_caesar@1";
    const std::vector<Index29> cipher = make_cipher(64);
    std::vector<nlohmann::json> params_list;
    for (int shift = 0; shift < 8; ++shift) {
        params_list.push_back(nlohmann::json{{"shift", shift}});
    }

    TheoryExportCache cache;
    REQUIRE(cache.ensure(root / "theories", uri, TransformDirection::Decrypt).ok());

    // Prefer module (proxy → bytecode twin) over ShapeInline.
    REQUIRE(TheoryHistModule::ensure_proxy(uri, proxy_digest("export-prefer")).ok());
    StatusOr<std::vector<double>> mod_scores = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", uri, params_list, TransformDirection::Decrypt,
        {}, InterruptPolicy::none(), &cache);
    REQUIRE(mod_scores.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::ModuleLoaded);

    // Parity vs CPU χ² for a lane.
    StatusOr<std::vector<Index29>> plain =
        TheoryDispatch::apply(theory, cipher, params_list[0], TransformDirection::Decrypt);
    REQUIRE(plain.ok());
    StatusOr<double> cpu = Chi2EnglishGp::score(plain.value(), freqs.value());
    REQUIRE(cpu.ok());
    REQUIRE(mod_scores.value()[0] == Catch::Approx(cpu.value()).margin(1e-9));

    // Inject invalid module launch → soft S0 (still ok scores; note S0).
    TheoryHistModule::set_inject_invalid_launch(true);
    StatusOr<std::vector<double>> soft_invalid = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", uri, params_list, TransformDirection::Decrypt,
        {}, InterruptPolicy::none(), &cache);
    REQUIRE(soft_invalid.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE(soft_invalid.value()[0] == Catch::Approx(cpu.value()).margin(1e-9));
    TheoryHistModule::set_inject_invalid_launch(false);

    // Inject OOM → soft S0.
    TheoryHistModule::set_inject_oom(true);
    StatusOr<std::vector<double>> soft_oom = GpuCandidateExport::theory_scores_only(
        cipher, freqs.value(), root / "theories", uri, params_list, TransformDirection::Decrypt,
        {}, InterruptPolicy::none(), &cache);
    REQUIRE(soft_oom.ok());
    REQUIRE(cache.last_hist_launch() == TheoryHistChi2Emit::Strategy::S0Bytecode);
    REQUIRE(soft_oom.value()[0] == Catch::Approx(cpu.value()).margin(1e-9));
    TheoryHistModule::set_inject_oom(false);

    TheoryHistModule::clear();
    std::filesystem::remove_all(root, ec);
}

#else

TEST_CASE("TheoryHistModule skipped without CUDA", "[cuda][theory][module]") {
    SUCCEED("PARCAE_HAS_CUDA not defined");
}

#endif
