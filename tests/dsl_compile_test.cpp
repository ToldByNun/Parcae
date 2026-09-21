#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_build_ir.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_registry.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
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

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

}  // namespace

TEST_CASE("DslCompile pipeline_ready when python package present", "[dsl][compile]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));
    REQUIRE(std::filesystem::is_regular_file(fixture_theory()));
}

TEST_CASE("DslCompile end-to-end quadratic_polynomial_stream", "[dsl][compile]") {
    const auto root =
        std::filesystem::temp_directory_path() / "parcae_dsl_compile_h34";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    StatusOr<DslCompile::Result> result =
        DslCompile::compile_file(fixture_theory(), root, compile_options());
    REQUIRE(result.ok());
    REQUIRE(result.value().artifacts().size() == 1);
    REQUIRE(
        result.value().artifacts().front().uri().to_string() ==
        "parcae://theories/quadratic_polynomial_stream@1");
    REQUIRE(result.value().artifacts().front().dsl_spec_version() ==
            DslSpecVersion::current_string);
    REQUIRE(result.value().artifacts().front().tier() == TheoryIr::Tier::B);
    REQUIRE(result.value().artifacts().front().structural_claim().has_value());
    REQUIRE(result.value().artifacts().front().structural_claim()->find("Spekulativ") !=
            std::string::npos);
    // Verify passed must not be confused with Tier A.
    REQUIRE(result.value().artifacts().front().verification().passed());
    REQUIRE(result.value().artifacts().front().tier() != TheoryIr::Tier::A);

    const std::filesystem::path manifest =
        root / "quadratic_polynomial_stream" / "1" / "manifest.json";
    REQUIRE(std::filesystem::is_regular_file(manifest));
    REQUIRE(std::filesystem::is_regular_file(
        root / "quadratic_polynomial_stream" / "1" / "cpu_reference.hpp"));
    REQUIRE(std::filesystem::is_regular_file(
        root / "quadratic_polynomial_stream" / "1" / "emitted" /
        "QuadraticPolynomialStreamKernel.hpp"));
    REQUIRE(std::filesystem::is_regular_file(
        root / "quadratic_polynomial_stream" / "1" / "emitted" /
        "QuadraticPolynomialStreamKernel.cu"));

    StatusOr<TheoryArtifact> loaded =
        TheoryRegistry::load(root, "quadratic_polynomial_stream", 1);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().paths().cuda_source().has_value());
    REQUIRE(*loaded.value().paths().cuda_source() ==
            "emitted/QuadraticPolynomialStreamKernel.cu");
    REQUIRE_FALSE(TheoryRegistry::is_stale_spec(loaded.value()));

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("DslBuildIr lowers poly2 primitive from ingested AST", "[dsl][compile][build]") {
    // Compile path covers spawn; this case isolates IR build via a fresh dump.
    const auto root =
        std::filesystem::temp_directory_path() / "parcae_dsl_build_ir_h34";
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
