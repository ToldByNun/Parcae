#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <parcae/core/sha256.hpp>
#include <parcae/corpus/fixture_loader.hpp>
#include <parcae/corpus/separator_grammar.hpp>
#include <parcae/corpus/tokenizer.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/gematria/latin_codec.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/apply_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <parcae/validate/plaintext_normalizer.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

TEST_CASE("Sha256 empty and abc vectors", "[sha256]") {
    REQUIRE(Sha256::hex_digest("") ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    REQUIRE(Sha256::hex_digest("abc") ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_CASE("PlaintextNormalizer strips long hex and folds preferred Latin", "[sha256][normalizer]") {
    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/gematria/gematria-primus-v0.json");
    REQUIRE(profile.ok());
    LatinCodec codec(profile.value());
    PlaintextNormalizer normalizer(codec);

    StatusOr<std::string> an_endish = normalizer.normalize(
        "AN END 43824d7f455a598778a620513710418dfb95d2942c78b8ebb0c7ef237abe57eb");
    REQUIRE(an_endish.ok());
    REQUIRE(an_endish.value() == "ANEND");

    // Preferred labels: V→U (single-letter alias fold).
    StatusOr<std::string> folded = normalizer.normalize("DIVINITY");
    REQUIRE(folded.ok());
    REQUIRE(folded.value() == "DIUINITY");
}

TEST_CASE("Dump oracle fixture digests", "[.][hashdump]") {
    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/gematria/gematria-primus-v0.json");
    REQUIRE(profile.ok());
    StatusOr<SeparatorGrammar> grammar = SeparatorGrammar::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/separators/rtkd-separator-grammar-v0.json");
    REQUIRE(grammar.ok());

    LatinCodec codec(profile.value());
    PlaintextNormalizer normalizer(codec);
    Tokenizer tokenizer(profile.value(), grammar.value());

    const char* ids[] = {
        "a-warning", "some-wisdom", "loss-of-divinity", "an-instruction",  "koan-1",
        "welcome",   "koan-2",      "an-end",           "lp2-57-identity",
    };

    for (const char* id : ids) {
        const std::string dir = std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/" + id;
        StatusOr<Fixture> fixture = FixtureLoader::load_directory(dir);
        REQUIRE(fixture.ok());

        StatusOr<TokenStream> stream = tokenizer.tokenize(fixture.value().ciphertext(), true);
        REQUIRE(stream.ok());
        StatusOr<InterruptPolicy> interrupt =
            InterruptPolicy::from_skip_indices(fixture.value().skip_indices());
        REQUIRE(interrupt.ok());
        StatusOr<TransformId> tid = TransformId::from_string(fixture.value().transform_id());
        REQUIRE(tid.ok());
        StatusOr<TransformDirection> dirn =
            TransformDirectionUtil::from_string(fixture.value().direction());
        REQUIRE(dirn.ok());
        StatusOr<std::vector<Index29>> plain =
            ApplyTransform::apply(tid.value(), stream.value().consumable_indices(),
                                  fixture.value().params(), dirn.value(), interrupt.value());
        REQUIRE(plain.ok());
        const std::string normalized = normalizer.from_indices(plain.value());
        StatusOr<std::string> expected = normalizer.normalize(fixture.value().plaintext());
        REQUIRE(expected.ok());
        REQUIRE(normalized == expected.value());

        std::cout << id << '\n'
                  << "  ciphertext_sha256: " << Sha256::hex_digest(fixture.value().ciphertext())
                  << '\n'
                  << "  plaintext_sha256: " << Sha256::hex_digest(fixture.value().plaintext())
                  << '\n'
                  << "  normalized_plaintext_sha256: " << Sha256::hex_digest(normalized) << '\n';
    }
}
