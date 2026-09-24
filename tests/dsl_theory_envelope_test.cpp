#include <parcae/core/status_or.hpp>
#include <parcae/core/version.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_envelope_bridge.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/tool/transform_envelope.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace {

constexpr const char* kSha =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

[[nodiscard]] StatusOr<TheoryArtifact> make_poly2_artifact() {
    TheoryArtifact::Paths paths;
    paths.set_envelope_template("envelope.json");
    return TheoryArtifact::make(
        "quadratic_polynomial_stream",
        1,
        TheoryIr::Tier::B,
        TheoryIr::Family::KeyedStream,
        kSha,
        TheoryArtifact::Verification{
            DslVerifier::Mode::Exhaustive,
            true,
            std::nullopt,
            "2026-09-21T00:00:00Z",
        },
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {TheoryArtifact::Param{"c2", 0, 28},
         TheoryArtifact::Param{"c1", 0, 28},
         TheoryArtifact::Param{"c0", 0, 28}},
        {"poly2_mod29"},
        std::string("Speculative. Test claim."),
        std::nullopt,
        std::move(paths));
}

}  // namespace

TEST_CASE("TheoryEnvelopeBridge parses catalog TransformEnvelope", "[dsl][envelope]") {
    const nlohmann::json root{
        {"transform_id", "caesar"},
        {"direction", "decrypt"},
        {"params", {{"shift", 3}}},
    };
    StatusOr<TheoryEnvelopeBridge::Envelope> env = TheoryEnvelopeBridge::from_json(root);
    REQUIRE(env.ok());
    REQUIRE(env.value().is_catalog());
    REQUIRE_FALSE(env.value().is_theory());
    REQUIRE(env.value().transform_id() == "caesar");

    StatusOr<TransformEnvelope> catalog = env.value().to_catalog_envelope();
    REQUIRE(catalog.ok());
    REQUIRE(catalog.value().transform_id() == TransformId::caesar());
    REQUIRE(catalog.value().params().at("shift") == 3);
}

TEST_CASE("TheoryEnvelopeBridge parses theory URI extension", "[dsl][envelope]") {
    const nlohmann::json root{
        {"transform_id", "parcae://theories/quadratic_polynomial_stream@1"},
        {"direction", "decrypt"},
        {"params", {{"c2", 0}, {"c1", 1}, {"c0", 2}}},
    };
    StatusOr<TheoryEnvelopeBridge::Envelope> env = TheoryEnvelopeBridge::from_json(root);
    REQUIRE(env.ok());
    REQUIRE(env.value().is_theory());
    REQUIRE(env.value().theory_uri().has_value());
    REQUIRE(
        env.value().theory_uri()->to_string() ==
        "parcae://theories/quadratic_polynomial_stream@1");

    StatusOr<TransformEnvelope> lowered = env.value().to_catalog_envelope();
    REQUIRE_FALSE(lowered.ok());
    REQUIRE(lowered.status().message().find("cannot be lowered") != std::string::npos);
    REQUIRE(lowered.status().message().find("TheoryDispatch") != std::string::npos);
}

TEST_CASE("TheoryEnvelopeBridge rejects unknown transform_id", "[dsl][envelope]") {
    const nlohmann::json root{
        {"transform_id", "not_a_real_transform"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    };
    StatusOr<TheoryEnvelopeBridge::Envelope> env = TheoryEnvelopeBridge::from_json(root);
    REQUIRE_FALSE(env.ok());
    REQUIRE(env.status().message().find("neither a catalog id") != std::string::npos);
}

TEST_CASE("TheoryEnvelopeBridge rejects malformed theory URI", "[dsl][envelope]") {
    const nlohmann::json root{
        {"transform_id", "parcae://theories/bad"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    };
    StatusOr<TheoryEnvelopeBridge::Envelope> env = TheoryEnvelopeBridge::from_json(root);
    REQUIRE_FALSE(env.ok());
}

TEST_CASE("TheoryEnvelopeBridge::template_for writes round-trip file", "[dsl][envelope]") {
    StatusOr<TheoryArtifact> art = make_poly2_artifact();
    REQUIRE(art.ok());

    StatusOr<TheoryEnvelopeBridge::Envelope> tmpl =
        TheoryEnvelopeBridge::template_for(art.value());
    REQUIRE(tmpl.ok());
    REQUIRE(tmpl.value().is_theory());
    REQUIRE(tmpl.value().params().at("c2") == 0);
    REQUIRE(tmpl.value().params().at("c1") == 0);
    REQUIRE(tmpl.value().params().at("c0") == 0);

    const auto dir =
        std::filesystem::temp_directory_path() / "parcae_theory_envelope_h38";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / "envelope.json";
    REQUIRE(TheoryEnvelopeBridge::write(path, tmpl.value()).ok());

    StatusOr<TheoryEnvelopeBridge::Envelope> loaded = TheoryEnvelopeBridge::load(path);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().is_theory());
    REQUIRE(loaded.value().to_json().at("transform_id") ==
            "parcae://theories/quadratic_polynomial_stream@1");
    REQUIRE(TheoryEnvelopeBridge::check_against_artifact(loaded.value(), art.value()).ok());

    std::filesystem::remove_all(dir, ec);
}

TEST_CASE(
    "TheoryEnvelopeBridge::check_against_artifact rejects URI mismatch",
    "[dsl][envelope]") {
    StatusOr<TheoryArtifact> art = make_poly2_artifact();
    REQUIRE(art.ok());

    const nlohmann::json root{
        {"transform_id", "parcae://theories/other_theory@1"},
        {"direction", "decrypt"},
        {"params", {{"c2", 0}, {"c1", 0}, {"c0", 0}}},
    };
    StatusOr<TheoryEnvelopeBridge::Envelope> env = TheoryEnvelopeBridge::from_json(root);
    REQUIRE(env.ok());
    Status check = TheoryEnvelopeBridge::check_against_artifact(env.value(), art.value());
    REQUIRE_FALSE(check.ok());
    REQUIRE(check.message().find("does not match") != std::string::npos);
}

TEST_CASE(
    "TheoryEnvelopeBridge::check_against_artifact rejects out-of-domain param",
    "[dsl][envelope]") {
    StatusOr<TheoryArtifact> art = make_poly2_artifact();
    REQUIRE(art.ok());

    const nlohmann::json root{
        {"transform_id", "parcae://theories/quadratic_polynomial_stream@1"},
        {"direction", "decrypt"},
        {"params", {{"c2", 99}, {"c1", 0}, {"c0", 0}}},
    };
    StatusOr<TheoryEnvelopeBridge::Envelope> env = TheoryEnvelopeBridge::from_json(root);
    REQUIRE(env.ok());
    Status check = TheoryEnvelopeBridge::check_against_artifact(env.value(), art.value());
    REQUIRE_FALSE(check.ok());
    REQUIRE(check.message().find("out of declared domain") != std::string::npos);
}

TEST_CASE("catalog TransformEnvelope still parses via tool API", "[dsl][envelope]") {
    // Bridge must not break the existing TransformEnvelope round-trip.
    const nlohmann::json root{
        {"transform_id", "affine"},
        {"direction", "encrypt"},
        {"params", {{"a", 3}, {"b", 5}}},
    };
    StatusOr<TransformEnvelope> direct =
        TransformEnvelope::from_json(root);
    REQUIRE(direct.ok());

    StatusOr<TheoryEnvelopeBridge::Envelope> via_bridge =
        TheoryEnvelopeBridge::from_json(root);
    REQUIRE(via_bridge.ok());
    StatusOr<TransformEnvelope> lowered =
        via_bridge.value().to_catalog_envelope();
    REQUIRE(lowered.ok());
    REQUIRE(lowered.value().to_json() == direct.value().to_json());
}
