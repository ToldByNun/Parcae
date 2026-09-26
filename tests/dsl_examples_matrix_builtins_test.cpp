#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_registry.hpp>
#include <parcae/dsl/theory_validate.hpp>
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

[[nodiscard]] std::filesystem::path matrix_builtins_example() {
    return std::filesystem::path(PARCAE_EXAMPLES_DIR) / "matrix_builtins_example.py";
}

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

} // namespace

TEST_CASE("examples/matrix_builtins_example.py is present", "[dsl][examples][matrix]") {
    REQUIRE(std::filesystem::is_regular_file(matrix_builtins_example()));
}

TEST_CASE("parcae-compile path: theories/examples/matrix_builtins_example.py",
          "[dsl][examples][matrix][compile]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const auto root = std::filesystem::temp_directory_path() / "parcae_examples_matrix_builtins";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    StatusOr<DslCompile::Result> result =
        DslCompile::compile_file(matrix_builtins_example(), root, compile_options());
    REQUIRE(result.ok());
    REQUIRE(result.value().artifacts().size() == 2);

    bool saw_mix = false;
    bool saw_compose = false;
    for (const TheoryArtifact& art : result.value().artifacts()) {
        REQUIRE(art.dsl_spec_version() == DslSpecVersion::current_string);
        REQUIRE(art.verification().passed());
        if (art.name() == "matrix_mix_stream") {
            saw_mix = true;
            REQUIRE(art.uri().to_string() == "parcae://theories/matrix_mix_stream@1");
            REQUIRE(art.tier() == TheoryIr::Tier::A);
            REQUIRE(art.family() == TheoryIr::Family::Elementwise);
            REQUIRE(art.params().size() == 4);
            REQUIRE(art.primitives().size() == 2);
        } else if (art.name() == "atbash_then_matrix_mix") {
            saw_compose = true;
            REQUIRE(art.uri().to_string() == "parcae://theories/atbash_then_matrix_mix@1");
            REQUIRE(art.family() == TheoryIr::Family::Compose);
            REQUIRE(art.fusion() == TheoryArtifact::FusionStatus::Fused);
        }
    }
    REQUIRE(saw_mix);
    REQUIRE(saw_compose);

    const std::filesystem::path mix_dir = root / "matrix_mix_stream" / "1";
    REQUIRE(std::filesystem::is_regular_file(mix_dir / "manifest.json"));
    REQUIRE(std::filesystem::is_regular_file(mix_dir / "cpu_reference.hpp"));
    REQUIRE(std::filesystem::is_regular_file(mix_dir / "emitted" / "MatrixMixStreamKernel.hpp"));
    REQUIRE(std::filesystem::is_regular_file(mix_dir / "emitted" / "MatrixMixStreamKernel.cu"));

    const std::filesystem::path compose_dir = root / "atbash_then_matrix_mix" / "1";
    REQUIRE(std::filesystem::is_regular_file(compose_dir / "cpu_reference.hpp"));
    REQUIRE(std::filesystem::is_regular_file(compose_dir / "emitted" / "AtbashThenMatrixMixKernel.hpp"));

    const TheoryValidate::Report mix_ok = TheoryValidate::validate(root, "matrix_mix_stream", 1);
    REQUIRE(mix_ok.ok());
    const TheoryValidate::Report compose_ok =
        TheoryValidate::validate(root, "atbash_then_matrix_mix", 1);
    REQUIRE(compose_ok.ok());

    std::filesystem::remove_all(root, ec);
}
