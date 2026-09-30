#include <catch2/catch_test_macros.hpp>

#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/theory_dispatch.hpp>
#include <parcae/dsl/theory_uri.hpp>
#include <parcae/generate/theory_explicit_params_candidate_generator.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_EXAMPLES_DIR
#error "PARCAE_EXAMPLES_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_EXE
#error "PARCAE_PYTHON_EXE must be defined"
#endif
#ifndef PARCAE_PYTHON_DIR
#error "PARCAE_PYTHON_DIR must be defined"
#endif

namespace {

[[nodiscard]] DslCompile::Options compile_options() {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    return opt;
}

} // namespace

TEST_CASE("TheoryExplicitParamsCandidateGenerator loads once and matches applicator oracle",
          "[generate][theory][bytecode][compile]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));

    const auto root =
        std::filesystem::temp_directory_path() / "parcae_theory_explicit_params_harden";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "theories", ec);

    const auto src = std::filesystem::path(PARCAE_EXAMPLES_DIR) / "new_math_example.py";
    StatusOr<DslCompile::Result> compiled =
        DslCompile::compile_file(src, root / "theories", compile_options());
    REQUIRE(compiled.ok());

    const std::string uri = "parcae://theories/quadratic_polynomial_stream@1";
    StatusOr<TheoryUri> parsed = TheoryUri::parse(uri);
    REQUIRE(parsed.ok());
    StatusOr<TheoryIr> theory =
        TheoryDispatch::load_apply_ir(root / "theories", parsed.value());
    REQUIRE(theory.ok());

    const std::vector<Index29> cipher{Index29{3}, Index29{10}, Index29{28}, Index29{0},
                                      Index29{7}, Index29{15}, Index29{22}};
    const std::vector<nlohmann::json> params_list{
        nlohmann::json{{"c2", 1}, {"c1", 0}, {"c0", 0}},
        nlohmann::json{{"c2", 0}, {"c1", 1}, {"c0", 0}},
        nlohmann::json{{"c2", 2}, {"c1", 3}, {"c0", 5}},
        nlohmann::json{{"c2", 0}, {"c1", 0}, {"c0", 14}},
    };

    StatusOr<std::vector<TransformCandidate>> generated =
        TheoryExplicitParamsCandidateGenerator::generate(
            cipher, root / "theories", uri, params_list, TransformDirection::Decrypt);
    REQUIRE(generated.ok());
    REQUIRE(generated.value().size() == params_list.size());

    for (std::size_t i = 0; i < params_list.size(); ++i) {
        const StatusOr<std::vector<Index29>> oracle = DslIrApplicator::apply(
            theory.value(), cipher, params_list[i], TransformDirection::Decrypt);
        REQUIRE(oracle.ok());
        REQUIRE(generated.value()[i].output_indices() == oracle.value());
        REQUIRE(generated.value()[i].candidate_id() ==
                TheoryExplicitParamsCandidateGenerator::make_candidate_id(uri, i));
        REQUIRE(generated.value()[i].transform_id().str() == uri);
    }

    std::filesystem::remove_all(root, ec);
}
