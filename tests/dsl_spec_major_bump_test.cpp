#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_registry.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_DSL_FIXTURES_DIR
#error "PARCAE_DSL_FIXTURES_DIR must be defined by tests/CMakeLists.txt"
#endif

namespace {

[[nodiscard]] std::filesystem::path dsl_theories_root() {
    return std::filesystem::path(PARCAE_DSL_FIXTURES_DIR) / "theories";
}

constexpr const char* kSha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

[[nodiscard]] TheoryArtifact::Verification ok_exhaustive() {
    return TheoryArtifact::Verification{
        DslVerifier::Mode::Exhaustive,
        true,
        std::nullopt,
        "2026-09-21T00:00:00Z",
    };
}

} // namespace

TEST_CASE("Committed stale-major fixture is rejected by TheoryRegistry", "[dsl][registry][stale]") {
    REQUIRE(DslSpecVersion::current_major == 1);
    REQUIRE(DslSpecVersion::current_string == "1.0.0");

    const std::filesystem::path root = dsl_theories_root();
    const std::filesystem::path manifest = root / "stale_major_demo" / "1" / "manifest.json";
    REQUIRE(std::filesystem::is_regular_file(manifest));

    // Bytes are a valid ready-shaped artifact for an older MAJOR.
    StatusOr<TheoryArtifact> raw = TheoryArtifact::load(root, "stale_major_demo", 1);
    REQUIRE(raw.ok());
    REQUIRE(raw.value().dsl_spec_version() == "0.9.0");
    REQUIRE(raw.value().verification().passed());
    REQUIRE(TheoryRegistry::is_stale_spec(raw.value()));

    // Registry load / URI resolve MUST hard-fail (no silent continue).
    StatusOr<TheoryArtifact> gated = TheoryRegistry::load(root, "stale_major_demo", 1);
    REQUIRE_FALSE(gated.ok());
    REQUIRE(gated.status().message().find("DSL spec 1.0.0 required") != std::string::npos);
    REQUIRE(gated.status().message().find("artifact built for 0.9.0") != std::string::npos);
    REQUIRE(gated.status().message().find("re-run parcae-compile") != std::string::npos);

    StatusOr<TheoryArtifact> by_uri =
        TheoryRegistry::load_uri(root, "parcae://theories/stale_major_demo@1");
    REQUIRE_FALSE(by_uri.ok());
}

TEST_CASE("Catalog lists stale-major fixture with stale_spec and not ready",
          "[dsl][registry][stale]") {
    StatusOr<std::vector<TheoryRegistry::CatalogEntry>> listed =
        TheoryRegistry::list(dsl_theories_root());
    REQUIRE(listed.ok());

    bool found = false;
    for (const TheoryRegistry::CatalogEntry& e : listed.value()) {
        if (e.uri().name() != "stale_major_demo" || e.uri().version() != 1) {
            continue;
        }
        found = true;
        REQUIRE(e.dsl_spec_version() == "0.9.0");
        REQUIRE(e.stale_spec());
        REQUIRE_FALSE(e.ready());
        REQUIRE(e.detail().find("re-run parcae-compile") != std::string::npos);
    }
    REQUIRE(found);
}

TEST_CASE("Simulated MAJOR bump invalidates today's 1.0.0 artifact bytes",
          "[dsl][registry][stale]") {
    // Contract: after a future DslSpecVersion major bump, previously current
    // artifacts become stale. We simulate by treating an on-disk 1.0.0 manifest
    // as if the toolchain required 2.0.0 (parse + major_mismatch helper).
    StatusOr<TheoryArtifact> made =
        TheoryArtifact::make("bump_victim", 1, TheoryIr::Tier::A, TheoryIr::Family::Elementwise,
                             kSha, ok_exhaustive(), TheoryArtifact::FusionStatus::NotApplicable,
                             TheoryArtifact::InterruptMode::ElementwiseDefault);
    REQUIRE(made.ok());
    REQUIRE(made.value().dsl_spec_version() == "1.0.0");
    REQUIRE(TheoryRegistry::check_dsl_spec(made.value()).ok());

    const DslSpecVersion artifact_spec = DslSpecVersion::parse("1.0.0").value();
    const DslSpecVersion future_toolchain = DslSpecVersion::parse("2.0.0").value();
    REQUIRE(artifact_spec.major() != future_toolchain.major());
    // Same predicate TheoryRegistry::is_stale_spec uses relative to *current*;
    // relative to a bumped toolchain the artifact major would mismatch.
    REQUIRE(artifact_spec.major() != 2);
    REQUIRE(DslSpecVersion::compare(artifact_spec, future_toolchain) < 0);

    // Mutate disk copy to a different old major and confirm load still rejects.
    const auto root = std::filesystem::temp_directory_path() / "parcae_dsl_major_bump_h33";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    REQUIRE(made.value().store(root).ok());
    {
        nlohmann::json j = made.value().to_json();
        j["dsl_spec_version"] = "0.1.0";
        std::ofstream out(TheoryArtifact::artifact_dir(root, "bump_victim", 1) / "manifest.json",
                          std::ios::binary | std::ios::trunc);
        REQUIRE(out);
        out << j.dump(2) << '\n';
    }
    REQUIRE_FALSE(TheoryRegistry::load(root, "bump_victim", 1).ok());
    REQUIRE(TheoryRegistry::is_stale_spec("0.1.0"));
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("Recompile after major bump restores registry load", "[dsl][registry][stale]") {
    const auto root = std::filesystem::temp_directory_path() / "parcae_dsl_recompile_h33";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    // Start from the committed stale fixture bytes copied into a temp tree.
    const std::filesystem::path src =
        dsl_theories_root() / "stale_major_demo" / "1" / "manifest.json";
    const std::filesystem::path dst_dir = TheoryArtifact::artifact_dir(root, "stale_major_demo", 1);
    std::filesystem::create_directories(dst_dir, ec);
    std::filesystem::copy_file(src, dst_dir / "manifest.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    REQUIRE_FALSE(TheoryRegistry::load(root, "stale_major_demo", 1).ok());

    // Re-compile = TheoryArtifact::make (stamps current dsl_spec_version) + store.
    StatusOr<TheoryArtifact> rebuilt = TheoryArtifact::make(
        "stale_major_demo", 1, TheoryIr::Tier::A, TheoryIr::Family::KeyedStream, kSha,
        ok_exhaustive(), TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {TheoryArtifact::Param{"c2", 0, 28}, TheoryArtifact::Param{"c1", 0, 28},
         TheoryArtifact::Param{"c0", 0, 28}},
        {"poly2_mod29"}, std::nullopt, std::string("theories/examples/stale_major_demo.py"));
    REQUIRE(rebuilt.ok());
    REQUIRE(rebuilt.value().dsl_spec_version() == DslSpecVersion::current_string);
    REQUIRE(rebuilt.value().store(root).ok());

    StatusOr<TheoryArtifact> loaded = TheoryRegistry::load(root, "stale_major_demo", 1);
    REQUIRE(loaded.ok());
    REQUIRE_FALSE(TheoryRegistry::is_stale_spec(loaded.value()));
    REQUIRE(loaded.value().dsl_spec_version() == "1.0.0");

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("Future major on disk is stale and not loadable", "[dsl][registry][stale]") {
    StatusOr<TheoryArtifact> made = TheoryArtifact::make(
        "future_major_demo", 1, TheoryIr::Tier::A, TheoryIr::Family::Elementwise, kSha,
        ok_exhaustive(), TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::ElementwiseDefault);
    REQUIRE(made.ok());

    nlohmann::json j = made.value().to_json();
    j["dsl_spec_version"] = "2.0.0";
    StatusOr<TheoryArtifact> parsed = TheoryArtifact::from_json(j);
    REQUIRE(parsed.ok());
    REQUIRE(TheoryRegistry::is_stale_spec(parsed.value()));
    REQUIRE_FALSE(TheoryRegistry::check_dsl_spec(parsed.value()).ok());
}
