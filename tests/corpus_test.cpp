#include <parcae/corpus/fixture_loader.hpp>
#include <parcae/corpus/separator_grammar.hpp>
#include <parcae/corpus/tokenizer.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <parcae/gematria/latin_codec.hpp>
#include <parcae/gematria/latin_labels.hpp>
#include <parcae/gematria/rune_codec.hpp>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
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

TEST_CASE("Tokenizer keeps hex literal runs as non-consumable Hex", "[tokenizer]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    StatusOr<TokenStream> stream =
        tokenizer.tokenize("ᚠ-36367763ab73783c7af284446c/", true);
    REQUIRE(stream.ok());
    REQUIRE(stream.value().consumable_count() == 1);
    REQUIRE(stream.value().at(2).kind() == TokenKind::Hex);
    REQUIRE(stream.value().at(2).text() == "36367763ab73783c7af284446c");
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

TEST_CASE("FixtureLoader loads draft a-warning", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/a-warning");
    REQUIRE(fixture.ok());
    REQUIRE(fixture.value().id() == "a-warning");
    REQUIRE(fixture.value().transform_id() == "atbash");
    REQUIRE(fixture.value().verification_status() == "draft");
    REQUIRE(fixture.value().skip_indices().empty());
    REQUIRE(fixture.value().ciphertext().find("ᚱ-ᛝᚱᚪᛗᚹ") != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("A WARNNG") != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("FOR ALL IS SACRED") != std::string::npos);
}

TEST_CASE("FixtureLoader loads some-wisdom with know-this decimal grid", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/some-wisdom");
    REQUIRE(fixture.ok());

    const Fixture& loaded = fixture.value();
    REQUIRE(loaded.id() == "some-wisdom");
    REQUIRE(loaded.transform_id() == "identity");
    REQUIRE(loaded.verification_status() == "draft");
    REQUIRE(loaded.ciphertext().find("ᛋᚩᛗᛖ-ᚹᛁᛋᛞᚩᛗ") != std::string::npos);
    REQUIRE(loaded.ciphertext().find("272-138-") != std::string::npos);
    REQUIRE(loaded.plaintext().find("CNOW THIS") != std::string::npos);

    REQUIRE(loaded.literal_regions().size() == 1);
    REQUIRE(loaded.literal_regions()[0].kind() == "decimal_grid");
    REQUIRE(loaded.literal_regions()[0].role() == "ciphertext_embedded");
    REQUIRE(loaded.literal_regions()[0].value_file() == "literals/know-this-grid.txt");
    REQUIRE(loaded.literal_regions()[0].compare() == "ignore_whitespace");
}

TEST_CASE("Tokenizer treats some-wisdom know-this numbers as non-runes", "[tokenizer][fixture]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/some-wisdom");
    REQUIRE(fixture.ok());

    StatusOr<TokenStream> stream = tokenizer.tokenize(fixture.value().ciphertext(), true);
    REQUIRE(stream.ok());

    std::vector<std::string> numbers;
    for (std::size_t i = 0; i < stream.value().size(); ++i) {
        const Token& token = stream.value().at(i);
        if (token.kind() == TokenKind::Number) {
            numbers.push_back(token.text());
            REQUIRE_FALSE(token.consumable_index().has_value());
        }
    }

    REQUIRE(numbers == std::vector<std::string>{
        "272",
        "138",
        "131",
        "151",
        "18",
        "226",
        "245",
        "18",
        "151",
        "131",
        "138",
        "272",
    });
    REQUIRE(stream.value().consumable_count() > 0);
}

TEST_CASE("FixtureLoader loads draft loss-of-divinity", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/loss-of-divinity");
    REQUIRE(fixture.ok());
    REQUIRE(fixture.value().id() == "loss-of-divinity");
    REQUIRE(fixture.value().transform_id() == "identity");
    REQUIRE(fixture.value().verification_status() == "draft");
    REQUIRE(fixture.value().ciphertext().find("ᚦᛖ-ᛚᚩᛋᛋ-ᚩᚠ-ᛞᛁᚢᛁᚾᛁᛏᚣ") != std::string::npos);
    REQUIRE(fixture.value().ciphertext().find('%') != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("THE LOSS OF DIVINITY") != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("PROGRAM REALITY") != std::string::npos);
    REQUIRE(fixture.value().literal_regions().empty());
}

TEST_CASE("FixtureLoader loads an-instruction with know-this decimal grid", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/an-instruction");
    REQUIRE(fixture.ok());

    const Fixture& loaded = fixture.value();
    REQUIRE(loaded.id() == "an-instruction");
    REQUIRE(loaded.transform_id() == "identity");
    REQUIRE(loaded.ciphertext().find("ᚪᚾ-ᛁᚾᛋᛏᚱᚢᚳᛏᛡᚾ") != std::string::npos);
    REQUIRE(loaded.ciphertext().find("434-1311-312-278-966") != std::string::npos);
    REQUIRE(loaded.plaintext().find("CWESTIAN ALL THNGS") != std::string::npos);

    REQUIRE(loaded.literal_regions().size() == 1);
    REQUIRE(loaded.literal_regions()[0].kind() == "decimal_grid");
    REQUIRE(loaded.literal_regions()[0].value_file() == "literals/know-this-grid.txt");
}

TEST_CASE("Tokenizer treats an-instruction grid and loss-of-divinity enums as numbers", "[tokenizer][fixture]") {
    const GematriaProfile profile = load_profile();
    const SeparatorGrammar grammar = load_grammar();
    const Tokenizer tokenizer(profile, grammar);

    StatusOr<Fixture> instruction = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/an-instruction");
    REQUIRE(instruction.ok());
    StatusOr<TokenStream> instruction_stream =
        tokenizer.tokenize(instruction.value().ciphertext(), true);
    REQUIRE(instruction_stream.ok());

    std::vector<std::string> grid_numbers;
    for (std::size_t i = 0; i < instruction_stream.value().size(); ++i) {
        const Token& token = instruction_stream.value().at(i);
        if (token.kind() == TokenKind::Number) {
            grid_numbers.push_back(token.text());
            REQUIRE_FALSE(token.consumable_index().has_value());
        }
    }
    REQUIRE(grid_numbers == std::vector<std::string>{
        "434",
        "1311",
        "312",
        "278",
        "966",
        "204",
        "812",
        "934",
        "280",
        "1071",
        "626",
        "620",
        "809",
        "620",
        "626",
        "1071",
        "280",
        "934",
        "812",
        "204",
        "966",
        "278",
        "312",
        "1311",
        "434",
    });

    StatusOr<Fixture> loss = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/loss-of-divinity");
    REQUIRE(loss.ok());
    StatusOr<TokenStream> loss_stream = tokenizer.tokenize(loss.value().ciphertext(), true);
    REQUIRE(loss_stream.ok());

    std::vector<std::string> enum_numbers;
    for (std::size_t i = 0; i < loss_stream.value().size(); ++i) {
        const Token& token = loss_stream.value().at(i);
        if (token.kind() == TokenKind::Number) {
            enum_numbers.push_back(token.text());
        }
        if (token.kind() == TokenKind::PageMark) {
            REQUIRE(token.text() == "%");
        }
    }
    REQUIRE(enum_numbers == std::vector<std::string>{"1", "2"});
}

TEST_CASE("FixtureLoader loads draft koan-1 compose envelope", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/koan-1");
    REQUIRE(fixture.ok());
    REQUIRE(fixture.value().id() == "koan-1");
    REQUIRE(fixture.value().transform_id() == "compose");
    REQUIRE(fixture.value().direction() == "decrypt");
    REQUIRE(fixture.value().skip_indices().empty());
    REQUIRE(fixture.value().ciphertext().find("ᚹ-ᚣᛠᚹᛟ") != std::string::npos);
    REQUIRE(fixture.value().ciphertext().find('%') != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("A COAN") != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("DO FOUR UNREASONABLE THNGS EACH DAY") !=
            std::string::npos);
}

TEST_CASE("FixtureLoader loads welcome with DIVINITY skip indices", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/welcome");
    REQUIRE(fixture.ok());

    const Fixture& loaded = fixture.value();
    REQUIRE(loaded.id() == "welcome");
    REQUIRE(loaded.transform_id() == "vigenere_key");
    REQUIRE(loaded.direction() == "decrypt");
    REQUIRE(loaded.verification_status() == "draft");
    REQUIRE_FALSE(loaded.recomputed_ok());

    REQUIRE(loaded.key_latin().has_value());
    REQUIRE(loaded.key_latin().value() == "DIVINITY");
    REQUIRE(loaded.key_indices().has_value());
    REQUIRE(loaded.key_indices().value() ==
            std::vector<int>{23, 10, 1, 10, 9, 10, 16, 26});

    REQUIRE(loaded.skip_indices() == std::vector<std::size_t>{
        48,
        74,
        84,
        132,
        159,
        160,
        250,
        421,
        443,
        465,
        514,
    });

    REQUIRE(loaded.ciphertext().find("ᚢᛠᛝᛋᛇᚠᚳ") != std::string::npos);
    REQUIRE(loaded.plaintext().find("WELCOME PILGRIM") != std::string::npos);
    REQUIRE(loaded.plaintext().find("AN INSTRUCTIAN COMMAND YOUR OWN SELF") !=
            std::string::npos);
}

TEST_CASE("FixtureLoader loads koan-2 with FIRFUMFERENFE skip indices", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/koan-2");
    REQUIRE(fixture.ok());

    const Fixture& loaded = fixture.value();
    REQUIRE(loaded.id() == "koan-2");
    REQUIRE(loaded.transform_id() == "vigenere_key");
    REQUIRE(loaded.key_latin().has_value());
    REQUIRE(loaded.key_latin().value() == "FIRFUMFERENFE");
    REQUIRE(loaded.key_indices().has_value());
    REQUIRE(loaded.key_indices().value() ==
            std::vector<int>{0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18});
    REQUIRE(loaded.skip_indices() == std::vector<std::size_t>{49, 58});
    REQUIRE(loaded.ciphertext().find("ᚪ-ᛋᚹᚪᛁ") != std::string::npos);
    REQUIRE(loaded.plaintext().find("THE I IS THE VOICE OF THE CIRCUMFERENCE") !=
            std::string::npos);
}

TEST_CASE("FixtureLoader loads an-end totient page with hex literal and skip", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/an-end");
    REQUIRE(fixture.ok());

    const Fixture& loaded = fixture.value();
    REQUIRE(loaded.id() == "an-end");
    REQUIRE(loaded.transform_id() == "totient_prime_stream");
    REQUIRE(loaded.direction() == "decrypt");
    REQUIRE(loaded.verification_status() == "draft");
    REQUIRE_FALSE(loaded.recomputed_ok());
    REQUIRE(loaded.skip_indices() == std::vector<std::size_t>{56});

    REQUIRE(loaded.ciphertext().find("ᚫᛄ-ᛟᛋᚱ") != std::string::npos);
    REQUIRE(loaded.ciphertext().find("36367763ab73783c7af284446c") != std::string::npos);
    REQUIRE(loaded.plaintext().find("HASHES TO") != std::string::npos);
    REQUIRE(loaded.plaintext().find(
                "36367763ab73783c7af284446c59466b4cd653239a311cb7116d4618dee09a8425893dc7500b464fdaf1672d7bef5e891c6e2274568926a49fb4f45132c2a8b4") !=
            std::string::npos);

    REQUIRE(loaded.literal_regions().size() == 1);
    REQUIRE(loaded.literal_regions()[0].kind() == "hex_string");
    REQUIRE(loaded.literal_regions()[0].role() == "plaintext_embedded");
    REQUIRE(loaded.literal_regions()[0].value_file() == "literals/deep-web-hash.txt");
    REQUIRE(loaded.literal_regions()[0].compare() == "exact");
}

TEST_CASE("FixtureLoader loads draft lp2-57-identity", "[fixture]") {
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(
        std::string(PARCAE_TEST_DATA_DIR) + "/fixtures/solved/lp2-57-identity");
    REQUIRE(fixture.ok());
    REQUIRE(fixture.value().id() == "lp2-57-identity");
    REQUIRE(fixture.value().transform_id() == "identity");
    REQUIRE(fixture.value().verification_status() == "draft");
    REQUIRE(fixture.value().ciphertext().find("ᛈᚪᚱᚪᛒᛚᛖ") != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("PARABLE") != std::string::npos);
    REQUIRE(fixture.value().plaintext().find("FIND THE DIVINITY WITHIN AND EMERGE") !=
            std::string::npos);
    REQUIRE(fixture.value().skip_indices().empty());
}

TEST_CASE("All solved fixture manifests parse with hash fields present", "[fixture]") {
    const std::filesystem::path solved_root =
        std::filesystem::path(PARCAE_TEST_DATA_DIR) / "fixtures" / "solved";
    REQUIRE(std::filesystem::is_directory(solved_root));

    std::size_t fixture_count = 0;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(solved_root)) {
        if (!entry.is_directory()) {
            continue;
        }

        const std::filesystem::path manifest_path = entry.path() / "manifest.json";
        if (!std::filesystem::exists(manifest_path)) {
            continue;
        }

        ++fixture_count;
        INFO(entry.path().filename().string());

        std::ifstream manifest_input(manifest_path, std::ios::binary);
        REQUIRE(manifest_input);
        nlohmann::json root;
        REQUIRE_NOTHROW(manifest_input >> root);

        REQUIRE(root.contains("schema"));
        REQUIRE(root.at("schema") == "parcae.fixture_manifest.v0");
        REQUIRE(root.contains("id"));
        REQUIRE(root.at("id").get<std::string>() == entry.path().filename().string());
        REQUIRE(root.contains("method"));
        REQUIRE(root.at("method").contains("transform_id"));
        REQUIRE(root.contains("files"));
        REQUIRE(root.at("files").contains("ciphertext"));
        REQUIRE(root.at("files").contains("plaintext"));
        REQUIRE(root.contains("verification"));
        REQUIRE(root.at("verification").contains("status"));

        REQUIRE(root.contains("hashes"));
        REQUIRE(root.at("hashes").is_object());
        REQUIRE(root.at("hashes").contains("ciphertext_sha256"));
        REQUIRE(root.at("hashes").contains("plaintext_sha256"));
        REQUIRE(root.at("hashes").contains("normalized_plaintext_sha256"));

        StatusOr<Fixture> fixture = FixtureLoader::load_directory(entry.path().string());
        REQUIRE(fixture.ok());
        REQUIRE(fixture.value().id() == entry.path().filename().string());
        REQUIRE_FALSE(fixture.value().ciphertext().empty());
        REQUIRE_FALSE(fixture.value().plaintext().empty());
        REQUIRE_FALSE(fixture.value().transform_id().empty());

        // Draft fixtures may leave digests null; locked ones must have all three.
        if (fixture.value().verification_status() == "locked") {
            REQUIRE(fixture.value().hashes().ciphertext_sha256().has_value());
            REQUIRE(fixture.value().hashes().plaintext_sha256().has_value());
            REQUIRE(fixture.value().hashes().normalized_plaintext_sha256().has_value());
        }
    }

    REQUIRE(fixture_count >= 10);
}

