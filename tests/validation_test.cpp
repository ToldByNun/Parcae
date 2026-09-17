#include <parcae/corpus/fixture_loader.hpp>
#include <parcae/corpus/separator_grammar.hpp>
#include <parcae/corpus/tokenizer.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/gematria/latin_codec.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/apply_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <parcae/validate/fixture_validator.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

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
            std::string detail = "validation check failed: " + check.name() + " — " + check.message();
            if (report.diff_excerpt().has_value()) {
                detail += " | " + report.diff_excerpt().value();
            }
            FAIL(detail);
        }
    }
    REQUIRE(report.ok());
}

/// Negative-control policy: every consumable ciphertext F (index 0) is skipped.
[[nodiscard]] std::vector<std::size_t> skip_all_ciphertext_f(
    const std::vector<Index29>& cipher_indices) {
    std::vector<std::size_t> skips;
    for (std::size_t i = 0; i < cipher_indices.size(); ++i) {
        if (cipher_indices[i].value() == 0) {
            skips.push_back(i);
        }
    }
    return skips;
}

[[nodiscard]] std::string letters_only_upper(const std::string& text) {
    std::string letters;
    letters.reserve(text.size());
    for (unsigned char ch : text) {
        if (std::isalpha(ch) != 0) {
            letters.push_back(static_cast<char>(std::toupper(ch)));
        }
    }
    return letters;
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

TEST_CASE("Validation runner reproduces koan-1 (Atbash then Caesar +3)", "[validation][koan-1]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const FixtureValidator validator(profile, grammar);

    const ValidationReport report = validator.validate_directory(fixture_dir("koan-1"));
    REQUIRE(report.fixture_id() == "koan-1");
    require_validation_ok(report);
}

TEST_CASE(
    "Validation runner reproduces welcome (DIVINITY + skips)",
    "[validation][welcome]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const FixtureValidator validator(profile, grammar);

    const ValidationReport report = validator.validate_directory(fixture_dir("welcome"));
    REQUIRE(report.fixture_id() == "welcome");
    require_validation_ok(report);
}

TEST_CASE(
    "Welcome fails under ciphertext-F-all-skip negative control",
    "[validation][welcome][negative]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const LatinCodec codec(profile);
    const Tokenizer tokenizer(profile, grammar);

    StatusOr<Fixture> fixture = FixtureLoader::load_directory(fixture_dir("welcome"));
    REQUIRE(fixture.ok());

    StatusOr<TokenStream> stream = tokenizer.tokenize(fixture.value().ciphertext(), true);
    REQUIRE(stream.ok());
    const std::vector<Index29> cipher = stream.value().consumable_indices();

    const std::vector<std::size_t> bad_skips = skip_all_ciphertext_f(cipher);
    REQUIRE_FALSE(bad_skips.empty());
    // Wiki interrupters are a strict subset of ciphertext F positions.
    REQUIRE(bad_skips.size() > fixture.value().skip_indices().size());
    REQUIRE(bad_skips != fixture.value().skip_indices());

    StatusOr<InterruptPolicy> bad_interrupt = InterruptPolicy::from_skip_indices(bad_skips);
    REQUIRE(bad_interrupt.ok());

    StatusOr<std::vector<Index29>> decrypted = ApplyTransform::apply(
        TransformId::vigenere_key(),
        cipher,
        fixture.value().params(),
        TransformDirection::Decrypt,
        bad_interrupt.value());
    REQUIRE(decrypted.ok());

    const std::string actual = codec.latinize(decrypted.value());
    StatusOr<std::string> expected =
        codec.round_trip_preferred(letters_only_upper(fixture.value().plaintext()));
    REQUIRE(expected.ok());

    // Desync: skipping every ciphertext F destroys DIVINITY periodicity.
    REQUIRE(actual != expected.value());

    // Control: the explicit wiki skip list still recovers plaintext.
    StatusOr<InterruptPolicy> good_interrupt =
        InterruptPolicy::from_skip_indices(fixture.value().skip_indices());
    REQUIRE(good_interrupt.ok());
    StatusOr<std::vector<Index29>> good = ApplyTransform::apply(
        TransformId::vigenere_key(),
        cipher,
        fixture.value().params(),
        TransformDirection::Decrypt,
        good_interrupt.value());
    REQUIRE(good.ok());
    REQUIRE(codec.latinize(good.value()) == expected.value());
}

TEST_CASE(
    "Validation runner reproduces koan-2 (FIRFUMFERENFE + skips)",
    "[validation][koan-2]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const FixtureValidator validator(profile, grammar);

    const ValidationReport report = validator.validate_directory(fixture_dir("koan-2"));
    REQUIRE(report.fixture_id() == "koan-2");
    require_validation_ok(report);
}
