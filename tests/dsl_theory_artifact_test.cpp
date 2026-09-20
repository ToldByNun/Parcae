#include <parcae/core/status_or.hpp>
#include <parcae/core/version.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_uri.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace {

constexpr const char* kSha =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

[[nodiscard]] TheoryArtifact::Verification ok_exhaustive() {
    return TheoryArtifact::Verification{
        DslVerifier::Mode::Exhaustive,
        true,
        std::nullopt,
        "2026-09-21T00:00:00Z",
    };
}

[[nodiscard]] StatusOr<TheoryArtifact> make_ready_a(std::string name = "poly2_mod29_demo") {
    return TheoryArtifact::make(
        std::move(name),
        1,
        TheoryIr::Tier::A,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {TheoryArtifact::Param{"c2", 0, 28},
         TheoryArtifact::Param{"c1", 0, 28},
         TheoryArtifact::Param{"c0", 0, 28}},
        {"poly2_mod29"},
        std::nullopt,
        std::string("theories/examples/new_math_example.py"));
}

}  // namespace

TEST_CASE("TheoryUri parse and format", "[dsl][artifact][uri]") {
    const StatusOr<TheoryUri> u = TheoryUri::parse("parcae://theories/my_affine@3");
    REQUIRE(u.ok());
    REQUIRE(u.value().name() == "my_affine");
    REQUIRE(u.value().version() == 3);
    REQUIRE(u.value().to_string() == "parcae://theories/my_affine@3");

    REQUIRE_FALSE(TheoryUri::parse("http://theories/x@1").ok());
    REQUIRE_FALSE(TheoryUri::parse("parcae://theories/Bad@1").ok());
    REQUIRE_FALSE(TheoryUri::parse("parcae://theories/ok@0").ok());
    REQUIRE_FALSE(TheoryUri::parse("parcae://theories/ok@01").ok());
    REQUIRE_FALSE(TheoryUri::make("Ok", 1).ok());
    REQUIRE_FALSE(TheoryUri::make("ok", 0).ok());
}

TEST_CASE("TheoryArtifact make embeds current dsl_spec_version", "[dsl][artifact]") {
    const StatusOr<TheoryArtifact> a = make_ready_a();
    REQUIRE(a.ok());
    REQUIRE(a.value().dsl_spec_version() == DslSpecVersion::current_string);
    REQUIRE(a.value().dsl_spec_version() == "1.0.0");
    REQUIRE(a.value().compiler_version() == PARCAE_VERSION_STRING);
    REQUIRE(a.value().uri().to_string() == "parcae://theories/poly2_mod29_demo@1");

    const nlohmann::json j = a.value().to_json();
    REQUIRE(j.at("schema") == "parcae.theory_artifact.v0");
    REQUIRE(j.at("dsl_spec_version") == "1.0.0");
    REQUIRE(j.at("compiler_version") == PARCAE_VERSION_STRING);
    REQUIRE(j.at("verification").at("passed") == true);
    REQUIRE(j.at("verification").at("seed").is_null());
    REQUIRE(j.at("fusion").at("status") == "n/a");
}

TEST_CASE("TheoryArtifact refuses failed verification", "[dsl][artifact]") {
    TheoryArtifact::Verification failed{
        DslVerifier::Mode::Exhaustive,
        false,
        std::nullopt,
        "2026-09-21T00:00:00Z",
    };
    const StatusOr<TheoryArtifact> a = TheoryArtifact::make(
        "poly2_mod29_demo",
        1,
        TheoryIr::Tier::A,
        TheoryIr::Family::KeyedStream,
        kSha,
        std::move(failed),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign);
    REQUIRE_FALSE(a.ok());
    REQUIRE(a.status().message().find("passed") != std::string::npos);
}

TEST_CASE("TheoryArtifact tier B requires structural_claim", "[dsl][artifact]") {
    const StatusOr<TheoryArtifact> missing = TheoryArtifact::make(
        "speculative_stream",
        1,
        TheoryIr::Tier::B,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::PolicyMethod);
    REQUIRE_FALSE(missing.ok());

    const StatusOr<TheoryArtifact> ok = TheoryArtifact::make(
        "speculative_stream",
        1,
        TheoryIr::Tier::B,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::PolicyMethod,
        {},
        {},
        std::string("Spekulativ. research anchor only."));
    REQUIRE(ok.ok());
    REQUIRE(ok.value().structural_claim().has_value());
}

TEST_CASE("TheoryArtifact paths reject escape", "[dsl][artifact]") {
    TheoryArtifact::Paths paths;
    paths.set_cpu_reference(std::string("../escape.hpp"));
    const StatusOr<TheoryArtifact> a = TheoryArtifact::make(
        "poly2_mod29_demo",
        1,
        TheoryIr::Tier::A,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {},
        {},
        std::nullopt,
        std::nullopt,
        std::move(paths));
    REQUIRE_FALSE(a.ok());
    REQUIRE(a.status().message().find("..") != std::string::npos);
}

TEST_CASE("TheoryArtifact store/load roundtrip embeds dsl_spec_version", "[dsl][artifact]") {
    const auto root =
        std::filesystem::temp_directory_path() / "parcae_theory_artifact_h31";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    TheoryArtifact::Paths paths;
    paths.set_cpu_reference(std::string("cpu_reference.hpp"));
    paths.set_verify_report(std::string("verify_report.json"));

    StatusOr<TheoryArtifact> with_paths = TheoryArtifact::make(
        "quadratic_polynomial_stream",
        1,
        TheoryIr::Tier::A,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {TheoryArtifact::Param{"c2", 0, 28}},
        {"poly2_mod29"},
        std::nullopt,
        std::string("theories/examples/new_math_example.py"),
        std::move(paths));
    REQUIRE(with_paths.ok());
    REQUIRE(with_paths.value().store(root).ok());

    const std::filesystem::path manifest =
        root / "quadratic_polynomial_stream" / "1" / "manifest.json";
    REQUIRE(std::filesystem::is_regular_file(manifest));

    {
        std::ifstream in(manifest);
        REQUIRE(in);
        nlohmann::json on_disk = nlohmann::json::parse(in);
        REQUIRE(on_disk.at("dsl_spec_version") == "1.0.0");
        REQUIRE(on_disk.at("compiler_version") == PARCAE_VERSION_STRING);
        REQUIRE(on_disk.at("uri") == "parcae://theories/quadratic_polynomial_stream@1");
    }

    StatusOr<TheoryArtifact> loaded =
        TheoryArtifact::load(root, "quadratic_polynomial_stream", 1);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().dsl_spec_version() == DslSpecVersion::current_string);
    REQUIRE(loaded.value().primitives().size() == 1);
    REQUIRE(loaded.value().paths().cpu_reference().has_value());
    REQUIRE(*loaded.value().paths().cpu_reference() == "cpu_reference.hpp");

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("TheoryArtifact from_json preserves foreign dsl_spec_version", "[dsl][artifact]") {
    StatusOr<TheoryArtifact> made = make_ready_a();
    REQUIRE(made.ok());
    nlohmann::json j = made.value().to_json();
    j["dsl_spec_version"] = "0.9.0";

    StatusOr<TheoryArtifact> parsed = TheoryArtifact::from_json(j);
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value().dsl_spec_version() == "0.9.0");
    // Writer path (make) always stamps current; load preserves for H32 reject.
    REQUIRE(parsed.value().dsl_spec_version() != DslSpecVersion::current_string);
}

TEST_CASE("TheoryArtifact fuzz seed rules", "[dsl][artifact]") {
    TheoryArtifact::Verification fuzz_ok{
        DslVerifier::Mode::Fuzz,
        true,
        DslVerifier::default_fuzz_seed,
        "2026-09-21T00:00:00Z",
    };
    REQUIRE(TheoryArtifact::make(
                "fuzz_theory",
                2,
                TheoryIr::Tier::A,
                TheoryIr::Family::Elementwise,
                kSha,
                fuzz_ok,
                TheoryArtifact::FusionStatus::NotApplicable,
                TheoryArtifact::InterruptMode::ElementwiseDefault)
                .ok());

    TheoryArtifact::Verification fuzz_no_seed{
        DslVerifier::Mode::Fuzz,
        true,
        std::nullopt,
        "2026-09-21T00:00:00Z",
    };
    REQUIRE_FALSE(TheoryArtifact::make(
                      "fuzz_theory",
                      2,
                      TheoryIr::Tier::A,
                      TheoryIr::Family::Elementwise,
                      kSha,
                      std::move(fuzz_no_seed),
                      TheoryArtifact::FusionStatus::NotApplicable,
                      TheoryArtifact::InterruptMode::ElementwiseDefault)
                      .ok());
}
