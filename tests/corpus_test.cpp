#include <parcae/corpus/fixture_loader.hpp>
#include <parcae/corpus/separator_grammar.hpp>
#include <parcae/corpus/tokenizer.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/gematria/latin_codec.hpp>
#include <parcae/gematria/latin_labels.hpp>
#include <parcae/gematria/rune_codec.hpp>

#include <catch2/catch_test_macros.hpp>

#include <fstream>
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

std::string read_data_file(const std::string& relative_path) {
    const std::string path = std::string(PARCAE_TEST_DATA_DIR) + "/" + relative_path;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
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

TEST_CASE("LatinCodec latinize/delatinize round-trip preferred labels", "[latin_codec]") {
    const GematriaProfile profile = load_profile();
    const LatinCodec codec(profile);

    const std::vector<Index29> welcome{
        Index29{7},   // W
        Index29{25},  // AE -> preferred "AE"
        Index29{20},  // L
        Index29{5},   // C
        Index29{22},  // OE
        Index29{19},  // M
        Index29{25},  // AE
    };

    const std::string preferred = codec.latinize(welcome);
    REQUIRE(preferred == "WAELCOEMAE");

    StatusOr<std::vector<Index29>> decoded = codec.delatinize(preferred);
    REQUIRE(decoded.ok());
    REQUIRE(decoded.value() == welcome);

    // Alias forms collapse to preferred multi-letter labels.
    StatusOr<std::string> round_trip = codec.round_trip_preferred("THING");
    REQUIRE(round_trip.ok());
    REQUIRE(round_trip.value() == "THING");

    StatusOr<std::string> alias_trip = codec.round_trip_preferred("THNG");
    REQUIRE(alias_trip.ok());
    REQUIRE(alias_trip.value() == "THING");

    StatusOr<std::string> v_to_u = codec.round_trip_preferred("V");
    REQUIRE(v_to_u.ok());
    REQUIRE(v_to_u.value() == "U");

    StatusOr<std::vector<Index29>> bad = codec.delatinize("QX");
    REQUIRE_FALSE(bad.ok());
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

TEST_CASE("Tokenizer golden fixture covers - . / & % with runes", "[tokenizer]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    const std::string golden = read_data_file("fixtures/tokenizer/golden-ascii-separators.txt");
    REQUIRE(golden == "ᚱ-ᛝᚱ.ᚪ&ᛗᚹ/%");

    StatusOr<TokenStream> stream = tokenizer.tokenize(golden, true);
    REQUIRE(stream.ok());
    REQUIRE(stream.value().size() == 11);
    REQUIRE(stream.value().consumable_count() == 6);
    REQUIRE(stream.value().text() == golden);

    const TokenStream& tokens = stream.value();
    REQUIRE(tokens.at(0).is_rune());
    REQUIRE(tokens.at(0).index29() == Index29{4});   // ᚱ
    REQUIRE(tokens.at(1).kind() == TokenKind::WordSep);
    REQUIRE(tokens.at(1).text() == "-");
    REQUIRE(tokens.at(2).is_rune());
    REQUIRE(tokens.at(2).index29() == Index29{21});  // ᛝ
    REQUIRE(tokens.at(3).is_rune());
    REQUIRE(tokens.at(3).index29() == Index29{4});   // ᚱ
    REQUIRE(tokens.at(4).kind() == TokenKind::ClauseSep);
    REQUIRE(tokens.at(4).text() == ".");
    REQUIRE(tokens.at(5).is_rune());
    REQUIRE(tokens.at(5).index29() == Index29{24});  // ᚪ
    REQUIRE(tokens.at(6).kind() == TokenKind::ParaSep);
    REQUIRE(tokens.at(6).text() == "&");
    REQUIRE(tokens.at(7).is_rune());
    REQUIRE(tokens.at(7).index29() == Index29{19});  // ᛗ
    REQUIRE(tokens.at(8).is_rune());
    REQUIRE(tokens.at(8).index29() == Index29{7});   // ᚹ
    REQUIRE(tokens.at(9).kind() == TokenKind::LineSep);
    REQUIRE(tokens.at(9).text() == "/");
    REQUIRE(tokens.at(10).kind() == TokenKind::PageMark);
    REQUIRE(tokens.at(10).text() == "%");

    REQUIRE(tokens.at(0).consumable_index() == 0);
    REQUIRE(tokens.at(2).consumable_index() == 1);
    REQUIRE(tokens.at(3).consumable_index() == 2);
    REQUIRE(tokens.at(5).consumable_index() == 3);
    REQUIRE(tokens.at(7).consumable_index() == 4);
    REQUIRE(tokens.at(8).consumable_index() == 5);
}

TEST_CASE("Tokenizer preserves byte ranges and consumable rune indices", "[tokenizer]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    // ᚱ = 3 UTF-8 bytes, then ASCII '-', then ᚠ = 3 bytes.
    const std::string text = "ᚱ-ᚠ";
    StatusOr<TokenStream> stream = tokenizer.tokenize(text, true);
    REQUIRE(stream.ok());
    REQUIRE(stream.value().size() == 3);

    const Token& first = stream.value().at(0);
    const Token& sep = stream.value().at(1);
    const Token& second = stream.value().at(2);

    REQUIRE(first.is_rune());
    REQUIRE(first.byte_begin() == 0);
    REQUIRE(first.byte_end() == 3);
    REQUIRE(first.consumable_index() == 0);
    REQUIRE(first.index29() == Index29{4});  // ᚱ

    REQUIRE(sep.kind() == TokenKind::WordSep);
    REQUIRE(sep.byte_begin() == 3);
    REQUIRE(sep.byte_end() == 4);
    REQUIRE_FALSE(sep.consumable_index().has_value());

    REQUIRE(second.is_rune());
    REQUIRE(second.byte_begin() == 4);
    REQUIRE(second.byte_end() == 7);
    REQUIRE(second.consumable_index() == 1);
    REQUIRE(second.index29() == Index29{0});  // ᚠ

    REQUIRE(stream.value().text() == text);
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
    REQUIRE_FALSE(fixture.value().hashes().ciphertext_sha256().has_value());
    REQUIRE_FALSE(fixture.value().hashes().plaintext_sha256().has_value());
}

TEST_CASE("FixtureLoader loads skips, key params, and expected hashes", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/synth-vigenere-draft");
    REQUIRE(fixture.ok());

    const Fixture& loaded = fixture.value();
    REQUIRE(loaded.id() == "synth-vigenere-draft");
    REQUIRE(loaded.transform_id() == "vigenere_key");
    REQUIRE(loaded.direction() == "decrypt");
    REQUIRE(loaded.verification_status() == "draft");
    REQUIRE_FALSE(loaded.recomputed_ok());

    REQUIRE(loaded.skip_indices().size() == 2);
    REQUIRE(loaded.skip_indices()[0] == 1);
    REQUIRE(loaded.skip_indices()[1] == 3);

    REQUIRE(loaded.key_latin().has_value());
    REQUIRE(loaded.key_latin().value() == "DIVINITY");
    REQUIRE(loaded.key_indices().has_value());
    REQUIRE(loaded.key_indices().value() ==
            std::vector<int>{23, 10, 1, 10, 9, 10, 16, 26});

    REQUIRE(loaded.hashes().ciphertext_sha256().has_value());
    REQUIRE(
        loaded.hashes().ciphertext_sha256().value() ==
        "6ab2044c028557bda0508b98567b1b93f5c921893039fb6e98de9b57d43a3bef");
    REQUIRE(loaded.hashes().plaintext_sha256().has_value());
    REQUIRE(
        loaded.hashes().plaintext_sha256().value() ==
        "d8fe1d54f8f9a165ed0067085ad964db3cf7a3cec1cd7d4dd5976cf699815551");
    REQUIRE_FALSE(loaded.hashes().normalized_plaintext_sha256().has_value());

    REQUIRE(loaded.ciphertext() == "ᚢᛠᚠᚱ\n");
    REQUIRE(loaded.plaintext() == "WELC\n");
}

TEST_CASE("FixtureLoader rejects locked fixtures missing hashes", "[fixture]") {
    Fixture incomplete{
        "locked-missing-hashes",
        "ct",
        "pt",
        "identity",
        "decrypt",
        "locked",
        true,
        {},
        {},
        std::nullopt,
        std::nullopt,
        FixtureHashes{},
    };
    Status status = incomplete.validate_lock_rules();
    REQUIRE_FALSE(status.ok());
}
