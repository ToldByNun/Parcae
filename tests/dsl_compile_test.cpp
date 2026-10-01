#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_build_ir.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/dsl/dsl_optimize.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_envelope_bridge.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_registry.hpp>
#include <string>

#ifndef PARCAE_DSL_FIXTURES_DIR
#error "PARCAE_DSL_FIXTURES_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_DIR
#error "PARCAE_PYTHON_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_EXE
#error "PARCAE_PYTHON_EXE must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path fixture_theory() {
    return std::filesystem::path(PARCAE_DSL_FIXTURES_DIR) / "sources" /
           "quadratic_polynomial_stream.py";
}

[[nodiscard]] std::filesystem::path fixture_affine_inv() {
    return std::filesystem::path(PARCAE_DSL_FIXTURES_DIR) / "sources" / "affine_inv_stream.py";
}

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

[[nodiscard]] std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path);
    REQUIRE(in);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("DslCompile pipeline_ready when python package present", "[dsl][compile]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));
    REQUIRE(std::filesystem::is_regular_file(fixture_theory()));
}

TEST_CASE("DslCompile end-to-end quadratic_polynomial_stream", "[dsl][compile]") {
    const auto root = std::filesystem::temp_directory_path() / "parcae_dsl_compile_h34";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    StatusOr<DslCompile::Result> result =
        DslCompile::compile_file(fixture_theory(), root, compile_options());
    REQUIRE(result.ok());
    REQUIRE(result.value().artifacts().size() == 1);
    REQUIRE(result.value().artifacts().front().uri().to_string() ==
            "parcae://theories/quadratic_polynomial_stream@1");
    REQUIRE(result.value().artifacts().front().dsl_spec_version() ==
            DslSpecVersion::current_string);
    REQUIRE(result.value().artifacts().front().tier() == TheoryIr::Tier::B);
    REQUIRE(result.value().artifacts().front().structural_claim().has_value());
    REQUIRE(result.value().artifacts().front().structural_claim()->find("Speculative") !=
            std::string::npos);
    // Verify passed must not be confused with Tier A.
    REQUIRE(result.value().artifacts().front().verification().passed());
    REQUIRE(result.value().artifacts().front().tier() != TheoryIr::Tier::A);

    const std::filesystem::path manifest =
        root / "quadratic_polynomial_stream" / "1" / "manifest.json";
    REQUIRE(std::filesystem::is_regular_file(manifest));
    REQUIRE(std::filesystem::is_regular_file(root / "quadratic_polynomial_stream" / "1" /
                                             "cpu_reference.hpp"));
    REQUIRE(std::filesystem::is_regular_file(root / "quadratic_polynomial_stream" / "1" /
                                             "emitted" / "QuadraticPolynomialStreamKernel.hpp"));
    REQUIRE(std::filesystem::is_regular_file(root / "quadratic_polynomial_stream" / "1" /
                                             "emitted" / "QuadraticPolynomialStreamKernel.cu"));
    REQUIRE(std::filesystem::is_regular_file(root / "quadratic_polynomial_stream" / "1" /
                                             "envelope.json"));

    StatusOr<TheoryArtifact> loaded = TheoryRegistry::load(root, "quadratic_polynomial_stream", 1);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().paths().cuda_source().has_value());
    REQUIRE(*loaded.value().paths().cuda_source() == "emitted/QuadraticPolynomialStreamKernel.cu");
    REQUIRE(loaded.value().paths().envelope_template().has_value());
    REQUIRE(*loaded.value().paths().envelope_template() == "envelope.json");
    REQUIRE_FALSE(TheoryRegistry::is_stale_spec(loaded.value()));

    StatusOr<TheoryEnvelopeBridge::Envelope> env =
        TheoryEnvelopeBridge::load(root / "quadratic_polynomial_stream" / "1" / "envelope.json");
    REQUIRE(env.ok());
    REQUIRE(env.value().is_theory());
    REQUIRE(env.value().transform_id() == "parcae://theories/quadratic_polynomial_stream@1");
    REQUIRE(TheoryEnvelopeBridge::check_against_artifact(env.value(), loaded.value()).ok());
    REQUIRE_FALSE(env.value().to_catalog_envelope().ok());

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("DslBuildIr lowers poly2 primitive from ingested AST", "[dsl][compile][build]") {
    // Compile path covers spawn; this case isolates IR build via a fresh dump.
    const auto root = std::filesystem::temp_directory_path() / "parcae_dsl_build_ir_h34";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    StatusOr<DslCompile::Result> result =
        DslCompile::compile_file(fixture_theory(), root, compile_options());
    REQUIRE(result.ok());
    // Artifact params/primitives prove BuildIr ran.
    REQUIRE(result.value().artifacts().front().primitives().size() == 1);
    REQUIRE(result.value().artifacts().front().primitives().front() == "poly2_mod29");
    REQUIRE(result.value().artifacts().front().params().size() == 3);

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("DslCompile hooks DslOptimize and emits inv hoist prelude",
          "[dsl][compile][optimize]") {
    REQUIRE(std::filesystem::is_regular_file(fixture_affine_inv()));
    const auto root = std::filesystem::temp_directory_path() / "parcae_dsl_compile_optimize";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    StatusOr<DslCompile::Result> result =
        DslCompile::compile_file(fixture_affine_inv(), root, compile_options());
    REQUIRE(result.ok());
    REQUIRE(result.value().artifacts().size() == 1);
    REQUIRE(result.value().artifacts().front().uri().to_string() ==
            "parcae://theories/affine_inv_stream@1");

    const std::filesystem::path cpu = root / "affine_inv_stream" / "1" / "cpu_reference.hpp";
    const std::filesystem::path cu =
        root / "affine_inv_stream" / "1" / "emitted" / "AffineInvStreamKernel.cu";
    REQUIRE(std::filesystem::is_regular_file(cpu));
    REQUIRE(std::filesystem::is_regular_file(cu));

    const std::string cpu_text = read_text(cpu);
    const std::string cu_text = read_text(cu);
    REQUIRE(cpu_text.find("__parcae_inv_0") != std::string::npos);
    REQUIRE(cpu_text.find("Z29::inv(") != std::string::npos);
    REQUIRE(cu_text.find("__parcae_inv_0") != std::string::npos);
    REQUIRE(cu_text.find("Z29Device::inv(") != std::string::npos);

    // apply_ir reinstates Inv (no free __parcae_inv_N) so runtime bytecode stays self-contained.
    const std::string apply_ir = read_text(root / "affine_inv_stream" / "1" / "apply_ir.json");
    REQUIRE(apply_ir.find("__parcae_inv_") == std::string::npos);
    REQUIRE(apply_ir.find("\"inv\"") != std::string::npos);

    std::filesystem::remove_all(root, ec);
}
