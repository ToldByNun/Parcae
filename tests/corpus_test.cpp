#include <parcae/corpus/fixture_loader.hpp>
#include <parcae/corpus/separator_grammar.hpp>
#include <parcae/corpus/tokenizer.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/gematria/latin_labels.hpp>
#include <parcae/gematria/rune_codec.hpp>

#include <catch2/catch_test_macros.hpp>

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

}  // namespace

TEST_CASE("RuneCodec encodes and decodes profile runes", "[rune_codec]") {
    const GematriaProfile profile = load_profile();
    const RuneCodec codec(profile);

    StatusOr<std::string> encoded = codec.encode(Index29{0});
    REQUIRE(encoded.ok());
    REQUIRE(encoded.value() == "ᚠ");

    StatusOr<RuneCodec::DecodeResult> decoded = codec.decode_at(encoded.value(), 0, true);
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value().index == Index29{0});

    StatusOr<std::vector<Index29>> all = codec.decode_all("ᚠᛞᛠ", true);
    REQUIRE(all.ok());
    REQUIRE(all.value().size() == 3);
    REQUIRE(all.value()[0] == Index29{0});
    REQUIRE(all.value()[1] == Index29{23});
    REQUIRE(all.value()[2] == Index29{28});
}

TEST_CASE("RuneCodec rejects unknown runes in strict mode", "[rune_codec]") {
    const GematriaProfile profile = load_profile();
    const RuneCodec codec(profile);
    StatusOr<std::vector<Index29>> decoded = codec.decode_all("A", true);
    REQUIRE_FALSE(decoded.ok());
}

TEST_CASE("LatinLabels preferred fold and aliases", "[latin_labels]") {
    const GematriaProfile profile = load_profile();
    const LatinLabels latin(profile);

    REQUIRE(latin.preferred(Index29{1}) == "U");
    StatusOr<Index29> from_alias = latin.index_for_label("V");
    REQUIRE(from_alias.ok());
    REQUIRE(from_alias.value() == Index29{1});

    StatusOr<std::string> folded = latin.fold_to_preferred("V");
    REQUIRE(folded.ok());
    REQUIRE(folded.value() == "U");

    StatusOr<std::string> canonical = latin.canonicalize("THNG");
    REQUIRE(canonical.ok());
    REQUIRE(canonical.value() == "THING");

    const std::vector<Index29> indices{Index29{2}, Index29{21}};
    REQUIRE(latin.to_preferred_string(indices) == "THING");
}

TEST_CASE("Tokenizer golden ASCII separators and runes", "[tokenizer]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    StatusOr<TokenStream> stream = tokenizer.tokenize("ᚱ-ᚠ.ᛖ/", true);
    REQUIRE(stream.ok());
    REQUIRE(stream.value().size() == 6);
    REQUIRE(stream.value().consumable_count() == 3);
    REQUIRE(stream.value().at(0).is_rune());
    REQUIRE(stream.value().at(1).kind() == TokenKind::WordSep);
    REQUIRE(stream.value().at(2).is_rune());
    REQUIRE(stream.value().at(3).kind() == TokenKind::ClauseSep);
    REQUIRE(stream.value().at(4).is_rune());
    REQUIRE(stream.value().at(5).kind() == TokenKind::LineSep);
    REQUIRE(stream.value().text() == "ᚱ-ᚠ.ᛖ/");
}

TEST_CASE("Tokenizer strict mode rejects unknown symbols", "[tokenizer]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    StatusOr<TokenStream> stream = tokenizer.tokenize("ᚠ?", true);
    REQUIRE_FALSE(stream.ok());
}

TEST_CASE("Tokenizer keeps numbers as non-consumable", "[tokenizer]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    StatusOr<TokenStream> stream = tokenizer.tokenize("ᚠ-272/", true);
    REQUIRE(stream.ok());
    REQUIRE(stream.value().consumable_count() == 1);
    REQUIRE(stream.value().at(2).kind() == TokenKind::Number);
    REQUIRE(stream.value().at(2).text() == "272");
}

TEST_CASE("FixtureLoader loads draft synth-identity", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/synth-identity");
    REQUIRE(fixture.ok());
    REQUIRE(fixture.value().id() == "synth-identity");
    REQUIRE(fixture.value().transform_id() == "identity");
    REQUIRE(fixture.value().verification_status() == "draft");
    REQUIRE(fixture.value().ciphertext().find("ᚱ") != std::string::npos);
    REQUIRE(fixture.value().plaintext().find('R') != std::string::npos);
}
