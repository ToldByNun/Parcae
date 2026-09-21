#include <parcae/core/status_or.hpp>
#include <parcae/core/version.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>
#include <parcae/dsl/dsl_verifier.hpp>
#include <parcae/dsl/theory_artifact.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/theory_validate.hpp>

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

[[nodiscard]] StatusOr<TheoryArtifact> make_ready(std::string name) {
    TheoryArtifact::Paths paths;
    paths.set_cpu_reference("cpu_reference.hpp");
    return TheoryArtifact::make(
        std::move(name),
        1,
        TheoryIr::Tier::A,
        TheoryIr::Family::KeyedStream,
        kSha,
        ok_exhaustive(),
        TheoryArtifact::FusionStatus::NotApplicable,
        TheoryArtifact::InterruptMode::NoneByDesign,
        {},
        {"poly2_mod29"},
        std::nullopt,
        std::nullopt,
        std::move(paths));
}

[[nodiscard]] std::filesystem::path make_temp_root(std::string_view suffix) {
    const auto root =
        std::filesystem::temp_directory_path() / ("parcae_theory_validate_" + std::string(suffix));
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);
    return root;
}

void write_cpu_stub(const std::filesystem::path& dir) {
    std::ofstream out(dir / "cpu_reference.hpp", std::ios::binary | std::ios::trunc);
    out << "// stub\n";
}

}  // namespace

TEST_CASE("TheoryValidate passes ready artifact with declared files", "[dsl][validate]") {
    const std::filesystem::path root = make_temp_root("ok");
    StatusOr<TheoryArtifact> a = make_ready("ok_theory");
    REQUIRE(a.ok());
    REQUIRE(a.value().store(root).ok());
    write_cpu_stub(a.value().artifact_dir(root));

    const TheoryValidate::Report r = TheoryValidate::validate(root, "ok_theory", 1);
    REQUIRE(r.ok());
    REQUIRE(r.uri() == "parcae://theories/ok_theory@1");
    REQUIRE(r.dsl_spec_version().has_value());
    REQUIRE(*r.dsl_spec_version() == DslSpecVersion::current_string);

    const TheoryValidate::Report by_uri =
        TheoryValidate::validate_uri(root, "parcae://theories/ok_theory@1");
    REQUIRE(by_uri.ok());
    const TheoryValidate::Report by_ref = TheoryValidate::validate_uri(root, "ok_theory@1");
    REQUIRE(by_ref.ok());
    const TheoryValidate::Report by_path =
        TheoryValidate::validate_target(root, (root / "ok_theory" / "1").string());
    REQUIRE(by_path.ok());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("TheoryValidate fails missing declared path file", "[dsl][validate]") {
    const std::filesystem::path root = make_temp_root("missing_path");
    StatusOr<TheoryArtifact> a = make_ready("missing_cpu");
    REQUIRE(a.ok());
    REQUIRE(a.value().store(root).ok());
    // Intentionally do not write cpu_reference.hpp

    const TheoryValidate::Report r = TheoryValidate::validate(root, "missing_cpu", 1);
    REQUIRE_FALSE(r.ok());
    bool saw_path = false;
    for (const TheoryValidate::Check& c : r.checks()) {
        if (c.name() == "paths.cpu_reference") {
            saw_path = true;
            REQUIRE_FALSE(c.ok());
            REQUIRE(c.message().find("missing") != std::string::npos);
        }
    }
    REQUIRE(saw_path);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("TheoryValidate fails stale dsl_spec_version", "[dsl][validate]") {
    const std::filesystem::path root = make_temp_root("stale");
    StatusOr<TheoryArtifact> a = make_ready("stale_theory");
    REQUIRE(a.ok());
    REQUIRE(a.value().store(root).ok());
    write_cpu_stub(a.value().artifact_dir(root));

    nlohmann::json j = a.value().to_json();
    j["dsl_spec_version"] = "0.9.0";
    {
        std::ofstream out(a.value().manifest_path(root), std::ios::binary | std::ios::trunc);
        out << j.dump(2) << '\n';
    }

    const TheoryValidate::Report r = TheoryValidate::validate(root, "stale_theory", 1);
    REQUIRE_FALSE(r.ok());
    bool saw_spec = false;
    for (const TheoryValidate::Check& c : r.checks()) {
        if (c.name() == "dsl_spec_version") {
            saw_spec = true;
            REQUIRE_FALSE(c.ok());
            REQUIRE(c.message().find("re-run parcae-compile") != std::string::npos);
        }
    }
    REQUIRE(saw_spec);

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

TEST_CASE("TheoryValidate::validate_all lists and checks", "[dsl][validate]") {
    const std::filesystem::path root = make_temp_root("all");
    StatusOr<TheoryArtifact> a = make_ready("listed");
    REQUIRE(a.ok());
    REQUIRE(a.value().store(root).ok());
    write_cpu_stub(a.value().artifact_dir(root));

    StatusOr<std::vector<TheoryValidate::Report>> all = TheoryValidate::validate_all(root);
    REQUIRE(all.ok());
    REQUIRE(all.value().size() == 1);
    REQUIRE(all.value().front().ok());

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}
