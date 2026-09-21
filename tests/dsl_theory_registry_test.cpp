#include <parcae/core/status_or.hpp>
#include <parcae/core/version.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_registry.hpp>
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

[[nodiscard]] StatusOr<TheoryArtifact> make_ready(std::string name, std::uint32_t version = 1) {
    return TheoryArtifact::make(
        std::move(name),
        version,
        TheoryIr::Tier::A,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {},
        {"poly2_mod29"});
}

[[nodiscard]] std::filesystem::path make_temp_root(std::string_view suffix) {
    const auto root =
        std::filesystem::temp_directory_path() / ("parcae_theory_registry_" + std::string(suffix));
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    return root;
}

}  // namespace

TEST_CASE("TheoryRegistry check_dsl_spec accepts current", "[dsl][registry]") {
    StatusOr<TheoryArtifact> a = make_ready("current_theory");
    REQUIRE(a.ok());
    REQUIRE(TheoryRegistry::check_dsl_spec(a.value()).ok());
    REQUIRE_FALSE(TheoryRegistry::is_stale_spec(a.value()));
}

TEST_CASE("TheoryRegistry rejects major-mismatched dsl_spec_version", "[dsl][registry]") {
    StatusOr<TheoryArtifact> a = make_ready("stale_theory");
    REQUIRE(a.ok());
    nlohmann::json j = a.value().to_json();
    j["dsl_spec_version"] = "0.9.0";

    StatusOr<TheoryArtifact> parsed = TheoryArtifact::from_json(j);
    REQUIRE(parsed.ok());
    REQUIRE(TheoryRegistry::is_stale_spec(parsed.value()));

    const Status check = TheoryRegistry::check_dsl_spec(parsed.value());
    REQUIRE_FALSE(check.ok());
    REQUIRE(check.message().find("DSL spec 1.0.0 required") != std::string::npos);
    REQUIRE(check.message().find("artifact built for 0.9.0") != std::string::npos);
    REQUIRE(check.message().find("re-run parcae-compile") != std::string::npos);
}

TEST_CASE("TheoryRegistry rejects forward-incompatible newer minor", "[dsl][registry]") {
    REQUIRE_FALSE(TheoryRegistry::check_dsl_spec_string("1.1.0").ok());
    REQUIRE_FALSE(TheoryRegistry::is_stale_spec("1.1.0"));  // same major → not catalog-stale
}

TEST_CASE("TheoryRegistry load enforces spec gate", "[dsl][registry]") {
    const std::filesystem::path root = make_temp_root("load");

    StatusOr<TheoryArtifact> current = make_ready("ok_theory");
    REQUIRE(current.ok());
    REQUIRE(current.value().store(root).ok());
    REQUIRE(TheoryRegistry::load(root, "ok_theory", 1).ok());
    REQUIRE(TheoryRegistry::load_uri(root, "parcae://theories/ok_theory@1").ok());

    StatusOr<TheoryArtifact> stale = make_ready("stale_theory");
    REQUIRE(stale.ok());
    REQUIRE(stale.value().store(root).ok());
    {
        nlohmann::json j = stale.value().to_json();
        j["dsl_spec_version"] = "0.9.0";
        const std::filesystem::path path =
            TheoryArtifact::artifact_dir(root, "stale_theory", 1) / "manifest.json";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        REQUIRE(out);
        out << j.dump(2) << '\n';
    }

    StatusOr<TheoryArtifact> loaded = TheoryRegistry::load(root, "stale_theory", 1);
    REQUIRE_FALSE(loaded.ok());
    REQUIRE(loaded.status().message().find("re-run parcae-compile") != std::string::npos);

    // Raw TheoryArtifact::load still reads the bytes (registry gate is the reject).
    REQUIRE(TheoryArtifact::load(root, "stale_theory", 1).ok());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("TheoryRegistry list marks stale_spec", "[dsl][registry]") {
    const std::filesystem::path root = make_temp_root("list");

    StatusOr<TheoryArtifact> ok = make_ready("ready_theory", 1);
    REQUIRE(ok.ok());
    REQUIRE(ok.value().store(root).ok());

    StatusOr<TheoryArtifact> stale = make_ready("old_theory", 2);
    REQUIRE(stale.ok());
    REQUIRE(stale.value().store(root).ok());
    {
        nlohmann::json j = stale.value().to_json();
        j["dsl_spec_version"] = "0.9.0";
        const std::filesystem::path path =
            TheoryArtifact::artifact_dir(root, "old_theory", 2) / "manifest.json";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        REQUIRE(out);
        out << j.dump(2) << '\n';
    }

    StatusOr<std::vector<TheoryRegistry::CatalogEntry>> listed = TheoryRegistry::list(root);
    REQUIRE(listed.ok());
    REQUIRE(listed.value().size() == 2);

    bool saw_ready = false;
    bool saw_stale = false;
    for (const TheoryRegistry::CatalogEntry& e : listed.value()) {
        if (e.uri().name() == "ready_theory") {
            REQUIRE_FALSE(e.stale_spec());
            REQUIRE(e.ready());
            REQUIRE(e.dsl_spec_version() == DslSpecVersion::current_string);
            saw_ready = true;
        }
        if (e.uri().name() == "old_theory") {
            REQUIRE(e.stale_spec());
            REQUIRE_FALSE(e.ready());
            REQUIRE(e.dsl_spec_version() == "0.9.0");
            REQUIRE(e.detail().find("re-run parcae-compile") != std::string::npos);
            saw_stale = true;
        }
    }
    REQUIRE(saw_ready);
    REQUIRE(saw_stale);

    for (const TheoryRegistry::CatalogEntry& e : listed.value()) {
        const nlohmann::json j = e.to_json();
        REQUIRE(j.at("uri").is_string());
        REQUIRE(j.at("dsl_spec_version").is_string());
        REQUIRE(j.at("stale_spec").is_boolean());
        REQUIRE(j.at("ready").is_boolean());
        REQUIRE(j.contains("detail"));
        if (e.uri().name() == "old_theory") {
            REQUIRE(j.at("stale_spec").get<bool>());
            REQUIRE_FALSE(j.at("ready").get<bool>());
            REQUIRE(j.at("detail").is_string());
        }
        if (e.uri().name() == "ready_theory") {
            REQUIRE_FALSE(j.at("stale_spec").get<bool>());
            REQUIRE(j.at("ready").get<bool>());
            REQUIRE(j.at("detail").is_null());
        }
    }

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("TheoryRegistry accepts older minor on same major", "[dsl][registry]") {
    // Current is 1.0.0; an older 1.0.0-equivalent patch/minor on same major is OK.
    // Simulate artifact stamped 1.0.0 (current) — already covered.
    // If toolchain were 1.1.0, 1.0.0 would be accepted; with current 1.0.0 we can
    // only assert the policy helper treats equal-major older as non-stale.
    const DslSpecVersion older = DslSpecVersion::parse("1.0.0").value();
    REQUIRE(older.check_compatible_with_current().ok());
    REQUIRE_FALSE(TheoryRegistry::is_stale_spec("1.0.0"));
    REQUIRE(TheoryRegistry::check_dsl_spec_string("1.0.0").ok());
}
