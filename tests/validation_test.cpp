#include <parcae/corpus/separator_grammar.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/validate/fixture_validator.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

GematriaProfile load_profile() {
    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/gematria/gematria-primus-v0.json");
    REQUIRE(profile.ok());
    return profile.value();
}

SeparatorGrammar load_grammar() {
    StatusOr<SeparatorGrammar> grammar = SeparatorGrammar::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/separators/rtkd-separator-grammar-v0.json");
    REQUIRE(grammar.ok());
    return grammar.value();
}

std::string fixture_dir(const std::string& id) {
    return std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/" + id;
}

void require_validation_ok(const ValidationReport& report) {
    INFO("fixture_id=" << report.fixture_id());
    if (report.diff_excerpt().has_value()) {
        INFO("diff=" << report.diff_excerpt().value());
    }
    for (const ValidationCheck& check : report.checks()) {
        INFO("check=" << check.name() << " ok=" << check.ok() << " msg=" << check.message());
        if (!check.ok()) {
            FAIL("validation check failed: " + check.name() + " — " + check.message());
        }
    }
    REQUIRE(report.ok());
}

}  // namespace

TEST_CASE("Validation runner reproduces a-warning (Atbash)", "[validation][a-warning]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const FixtureValidator validator(profile, grammar);

    const ValidationReport report = validator.validate_directory(fixture_dir("a-warning"));
    REQUIRE(report.fixture_id() == "a-warning");
    require_validation_ok(report);
}

TEST_CASE(
    "Validation runner reproduces identity pages (wisdom, loss, instruction, 57)",
    "[validation][identity]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const FixtureValidator validator(profile, grammar);

    for (const char* id :
         {"some-wisdom", "loss-of-divinity", "an-instruction", "lp2-57-identity"}) {
        SECTION(id) {
            const ValidationReport report = validator.validate_directory(fixture_dir(id));
            REQUIRE(report.fixture_id() == id);
            require_validation_ok(report);
        }
    }
}
