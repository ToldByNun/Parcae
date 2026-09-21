#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/theory_envelope_bridge.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_registry.hpp>
#include <parcae/dsl/theory_validate.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#ifndef PARCAE_EXAMPLES_DIR
#error "PARCAE_EXAMPLES_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_DIR
#error "PARCAE_PYTHON_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_EXE
#error "PARCAE_PYTHON_EXE must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path new_math_example() {
    return std::filesystem::path(PARCAE_EXAMPLES_DIR) / "new_math_example.py";
}

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

}  // namespace

TEST_CASE("examples/new_math_example.py is present", "[dsl][examples][i39]") {
    REQUIRE(std::filesystem::is_regular_file(new_math_example()));
}

TEST_CASE(
    "parcae-compile path: theories/examples/new_math_example.py",
    "[dsl][examples][i39][compile]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const auto root =
        std::filesystem::temp_directory_path() / "parcae_examples_new_math_i39";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    StatusOr<DslCompile::Result> result =
        DslCompile::compile_file(new_math_example(), root, compile_options());
    REQUIRE(result.ok());
    REQUIRE(result.value().artifacts().size() == 1);

    const TheoryArtifact& art = result.value().artifacts().front();
    REQUIRE(art.uri().to_string() == "parcae://theories/quadratic_polynomial_stream@1");
    REQUIRE(art.name() == "quadratic_polynomial_stream");
    REQUIRE(art.dsl_spec_version() == DslSpecVersion::current_string);
    REQUIRE(art.tier() == TheoryIr::Tier::B);
    REQUIRE(art.structural_claim().has_value());
    REQUIRE(art.structural_claim()->find("Speculative") != std::string::npos);
    REQUIRE(art.verification().passed());
    REQUIRE(art.primitives().size() == 1);
    REQUIRE(art.primitives().front() == "poly2_mod29");
    REQUIRE(art.params().size() == 3);
    REQUIRE(art.paths().envelope_template().has_value());
    REQUIRE(*art.paths().envelope_template() == "envelope.json");

    const std::filesystem::path dir = root / "quadratic_polynomial_stream" / "1";
    REQUIRE(std::filesystem::is_regular_file(dir / "manifest.json"));
    REQUIRE(std::filesystem::is_regular_file(dir / "cpu_reference.hpp"));
    REQUIRE(std::filesystem::is_regular_file(dir / "envelope.json"));
    REQUIRE(std::filesystem::is_regular_file(
        dir / "emitted" / "QuadraticPolynomialStreamKernel.hpp"));
    REQUIRE(std::filesystem::is_regular_file(
        dir / "emitted" / "QuadraticPolynomialStreamKernel.cu"));

    StatusOr<TheoryArtifact> loaded =
        TheoryRegistry::load(root, "quadratic_polynomial_stream", 1);
    REQUIRE(loaded.ok());
    REQUIRE_FALSE(TheoryRegistry::is_stale_spec(loaded.value()));

    StatusOr<TheoryEnvelopeBridge::Envelope> env =
        TheoryEnvelopeBridge::load(dir / "envelope.json");
    REQUIRE(env.ok());
    REQUIRE(env.value().is_theory());
    REQUIRE(
        env.value().transform_id() == "parcae://theories/quadratic_polynomial_stream@1");

    const TheoryValidate::Report report =
        TheoryValidate::validate(root, "quadratic_polynomial_stream", 1);
    REQUIRE(report.ok());

    std::filesystem::remove_all(root, ec);
}
