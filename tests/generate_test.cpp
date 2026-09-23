#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/generate/affine_candidate_generator.hpp>
#include <parcae/generate/atbash_candidate_generator.hpp>
#include <parcae/generate/atbash_caesar_candidate_generator.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/generate/generator_registry.hpp>
#include <parcae/generate/vigenere_explicit_key_candidate_generator.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/exact_match.hpp>
#include <parcae/transform/affine_transform.hpp>
#include <parcae/transform/atbash_transform.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <parcae/transform/vigenere_key_transform.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

/// Among candidates, find unique exact-match=1.0 against `plain`; return its index.
[[nodiscard]] std::size_t require_unique_exact_match_rank1(
    const std::vector<TransformCandidate>& candidates,
    const std::vector<Index29>& plain) {
    std::size_t hits = 0;
    std::size_t hit_index = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        StatusOr<double> score = ExactMatch::score(candidates[i].output_indices(), plain);
        REQUIRE(score.ok());
        if (score.value() == 1.0) {
            ++hits;
            hit_index = i;
        } else {
            REQUIRE(score.value() == 0.0);
        }
    }
    REQUIRE(hits == 1);
    return hit_index;
}

}  // namespace

TEST_CASE("CaesarCandidateGenerator emits all 29 shifts in order", "[generate][caesar]") {
    const std::vector<Index29> cipher = {I(0), I(5), I(10), I(28)};

    StatusOr<std::vector<TransformCandidate>> candidates =
        CaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 29);
    REQUIRE(CaesarCandidateGenerator::candidate_count == 29);
    REQUIRE(CaesarCandidateGenerator::generator_id == "gen_caesar");

    const CaesarTransform caesar;
    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        const TransformCandidate& c = candidates.value()[shift];
        REQUIRE(c.candidate_id() == CaesarCandidateGenerator::make_candidate_id(shift));
        REQUIRE(c.transform_id() == TransformId::caesar());
        REQUIRE(c.direction() == TransformDirection::Decrypt);
        REQUIRE(c.params().at("shift").get<int>() == static_cast<int>(shift));
        REQUIRE(c.output_indices().size() == cipher.size());

        StatusOr<std::vector<Index29>> expected = caesar.apply(
            cipher,
            nlohmann::json{{"shift", static_cast<int>(shift)}},
            TransformDirection::Decrypt);
        REQUIRE(expected.ok());
        REQUIRE(c.output_indices() == expected.value());

        const nlohmann::json env = c.envelope();
        REQUIRE(env.at("transform_id") == "caesar");
        REQUIRE(env.at("direction") == "decrypt");
        REQUIRE(env.at("params").at("shift") == static_cast<int>(shift));
    }

    // shift=0 decrypt is identity.
    REQUIRE(candidates.value()[0].output_indices() == cipher);
}

TEST_CASE("CaesarCandidateGenerator encrypt path and round-trip", "[generate][caesar]") {
    const std::vector<Index29> plain = {I(1), I(2), I(3), I(4), I(5)};

    StatusOr<std::vector<TransformCandidate>> encrypted =
        CaesarCandidateGenerator::generate(plain, TransformDirection::Encrypt);
    REQUIRE(encrypted.ok());
    REQUIRE(encrypted.value().size() == 29);

    constexpr std::uint8_t kShift = 7;
    const std::vector<Index29>& cipher = encrypted.value()[kShift].output_indices();
    REQUIRE(encrypted.value()[kShift].candidate_id() == "caesar:shift=7");
    REQUIRE(encrypted.value()[kShift].envelope().at("direction") == "encrypt");

    StatusOr<std::vector<TransformCandidate>> decrypted =
        CaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(decrypted.ok());
    REQUIRE(decrypted.value()[kShift].output_indices() == plain);

    StatusOr<double> exact =
        ExactMatch::score(decrypted.value()[kShift].output_indices(), plain);
    REQUIRE(exact.ok());
    REQUIRE(exact.value() == 1.0);
}

TEST_CASE("CaesarCandidateGenerator JSON record shape", "[generate][caesar]") {
    const std::vector<Index29> cipher = {I(3), I(4)};
    StatusOr<std::vector<TransformCandidate>> candidates =
        CaesarCandidateGenerator::generate(cipher);
    REQUIRE(candidates.ok());

    const nlohmann::json record = candidates.value()[3].to_json();
    REQUIRE(record.at("candidate_id") == "caesar:shift=3");
    REQUIRE(record.contains("envelope"));
    REQUIRE(record.contains("output_indices"));
    REQUIRE(record.at("output_indices").is_array());
    REQUIRE(record.at("output_indices").size() == 2);
    REQUIRE(record.at("output_indices")[0].get<int>() ==
            static_cast<int>(Z29::sub(I(3), I(3)).value()));
    REQUIRE(record.at("output_indices")[1].get<int>() ==
            static_cast<int>(Z29::sub(I(4), I(3)).value()));
}

TEST_CASE("AtbashCandidateGenerator emits a single candidate", "[generate][atbash]") {
    const std::vector<Index29> cipher = {I(0), I(4), I(14), I(28)};

    StatusOr<std::vector<TransformCandidate>> candidates =
        AtbashCandidateGenerator::generate(cipher);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 1);
    REQUIRE(AtbashCandidateGenerator::candidate_count == 1);
    REQUIRE(AtbashCandidateGenerator::generator_id == "gen_atbash");

    const TransformCandidate& c = candidates.value()[0];
    REQUIRE(c.candidate_id() == "atbash");
    REQUIRE(c.transform_id() == TransformId::atbash());
    REQUIRE(c.params().empty());
    REQUIRE(
        c.output_indices() ==
        std::vector<Index29>{I(28), I(24), I(14), I(0)});

    // Involution: applying again recovers ciphertext.
    StatusOr<std::vector<TransformCandidate>> again =
        AtbashCandidateGenerator::generate(c.output_indices());
    REQUIRE(again.ok());
    REQUIRE(again.value()[0].output_indices() == cipher);
}

TEST_CASE("AtbashCaesarCandidateGenerator emits 29 Koan-1-family shifts", "[generate][atbash_caesar]") {
    const std::vector<Index29> cipher = {I(0), I(5), I(28)};

    StatusOr<std::vector<TransformCandidate>> candidates =
        AtbashCaesarCandidateGenerator::generate(cipher);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 29);
    REQUIRE(AtbashCaesarCandidateGenerator::candidate_count == 29);
    REQUIRE(AtbashCaesarCandidateGenerator::generator_id == "gen_atbash_caesar");

    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        const TransformCandidate& c = candidates.value()[shift];
        REQUIRE(c.candidate_id() == AtbashCaesarCandidateGenerator::make_candidate_id(shift));
        REQUIRE(c.transform_id() == TransformId::compose());
        REQUIRE(c.params() == ComposeTransform::atbash_then_caesar_params(shift));

        StatusOr<std::vector<Index29>> expected =
            ComposeTransform::apply_atbash_then_caesar(cipher, shift, TransformDirection::Decrypt);
        REQUIRE(expected.ok());
        REQUIRE(c.output_indices() == expected.value());
    }

    // shift=3 matches the hand vector from transform tests.
    REQUIRE(
        candidates.value()[3].output_indices() ==
        std::vector<Index29>{I(2), I(26), I(3)});
}

TEST_CASE("AffineCandidateGenerator enumerates 28x29=812 with documented cost", "[generate][affine]") {
    REQUIRE(AffineCandidateGenerator::a_count == 28);
    REQUIRE(AffineCandidateGenerator::b_count == 29);
    REQUIRE(AffineCandidateGenerator::candidate_count == 812);
    REQUIRE(AffineCandidateGenerator::generator_id == "gen_affine");

    const std::vector<Index29> cipher = {I(0), I(1), I(10), I(14)};
    StatusOr<std::vector<TransformCandidate>> candidates =
        AffineCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 812);

    // Order: a ascending outer, b ascending inner.
    REQUIRE(candidates.value()[0].candidate_id() == "affine:a=1,b=0");
    REQUIRE(candidates.value()[28].candidate_id() == "affine:a=1,b=28");
    REQUIRE(candidates.value()[29].candidate_id() == "affine:a=2,b=0");
    REQUIRE(candidates.value().back().candidate_id() == "affine:a=28,b=28");

    // Spot-check a=2,b=5 against AffineTransform (index = (2-1)*29 + 5 = 34).
    const TransformCandidate& a2b5 = candidates.value()[34];
    REQUIRE(a2b5.candidate_id() == "affine:a=2,b=5");
    REQUIRE(a2b5.params().at("a").get<int>() == 2);
    REQUIRE(a2b5.params().at("b").get<int>() == 5);

    const AffineTransform affine;
    StatusOr<std::vector<Index29>> expected = affine.apply(
        cipher,
        nlohmann::json{{"a", 2}, {"b", 5}},
        TransformDirection::Decrypt);
    REQUIRE(expected.ok());
    REQUIRE(a2b5.output_indices() == expected.value());

    // Encrypt→decrypt round-trip for one pair via the generator.
    StatusOr<std::vector<TransformCandidate>> encrypted =
        AffineCandidateGenerator::generate(cipher, TransformDirection::Encrypt);
    REQUIRE(encrypted.ok());
    const std::vector<Index29>& mid = encrypted.value()[34].output_indices();
    StatusOr<std::vector<TransformCandidate>> recovered =
        AffineCandidateGenerator::generate(mid, TransformDirection::Decrypt);
    REQUIRE(recovered.ok());
    REQUIRE(ExactMatch::score(recovered.value()[34].output_indices(), cipher).value() == 1.0);
}

TEST_CASE("VigenereExplicitKeyCandidateGenerator applies caller keys only", "[generate][vigenere]") {
    REQUIRE(VigenereExplicitKeyCandidateGenerator::generator_id == "gen_vigenere_explicit_keys");

    const std::vector<Index29> cipher = {I(1), I(3), I(3), I(5)};
    SECTION("rejects empty key list") {
        const std::vector<std::vector<Index29>> empty;
        REQUIRE_FALSE(VigenereExplicitKeyCandidateGenerator::generate(cipher, empty).ok());
    }

    SECTION("enumerates provided keys in order") {
        const std::vector<std::vector<Index29>> keys = {
            {I(9), I(9)},
            {I(1), I(2)},
            {I(3), I(4), I(5)},
        };
        StatusOr<std::vector<TransformCandidate>> candidates =
            VigenereExplicitKeyCandidateGenerator::generate(cipher, keys);
        REQUIRE(candidates.ok());
        REQUIRE(candidates.value().size() == 3);
        REQUIRE(candidates.value()[0].candidate_id() == "vigenere_key:i=0:key_indices=9,9");
        REQUIRE(candidates.value()[1].candidate_id() == "vigenere_key:i=1:key_indices=1,2");
        REQUIRE(candidates.value()[2].params().at("key_indices") == nlohmann::json{3, 4, 5});
        REQUIRE_FALSE(candidates.value()[0].interrupt().has_value());
    }

    SECTION("optional key_latin metadata and interrupt in envelope") {
        const std::vector<ExplicitVigenereKey> keys = {
            ExplicitVigenereKey{{I(1), I(2)}, std::string{"AB"}},
        };
        StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1});
        REQUIRE(interrupt.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            VigenereExplicitKeyCandidateGenerator::generate(
                cipher,
                keys,
                TransformDirection::Decrypt,
                interrupt.value());
        REQUIRE(candidates.ok());
        REQUIRE(candidates.value().size() == 1);
        REQUIRE(candidates.value()[0].params().at("key_latin") == "AB");
        REQUIRE(candidates.value()[0].interrupt().has_value());
        REQUIRE(candidates.value()[0].envelope().contains("interrupt"));
        REQUIRE(
            candidates.value()[0].envelope().at("interrupt").at("skip_indices") ==
            nlohmann::json{1});
    }
}

TEST_CASE(
    "Generators include params that recover known synthetic ciphertexts at rank 1 under exact-match",
    "[generate][rank1]") {
    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};

    SECTION("gen_caesar") {
        constexpr int kShift = 7;
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain,
            nlohmann::json{{"shift", kShift}},
            TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            CaesarCandidateGenerator::generate(cipher.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 =
            require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == static_cast<std::size_t>(kShift));
        REQUIRE(candidates.value()[rank1].params().at("shift") == kShift);
    }

    SECTION("gen_atbash") {
        StatusOr<std::vector<Index29>> cipher = AtbashTransform{}.apply(
            plain,
            nlohmann::json::object(),
            TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            AtbashCandidateGenerator::generate(cipher.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 =
            require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == 0);
        REQUIRE(candidates.value()[rank1].candidate_id() == "atbash");
    }

    SECTION("gen_atbash_caesar") {
        constexpr std::uint8_t kShift = 3;
        StatusOr<std::vector<Index29>> cipher =
            ComposeTransform::apply_atbash_then_caesar(plain, kShift, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            AtbashCaesarCandidateGenerator::generate(cipher.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 =
            require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == kShift);
        REQUIRE(candidates.value()[rank1].params() ==
                ComposeTransform::atbash_then_caesar_params(kShift));
    }

    SECTION("gen_affine") {
        constexpr int kA = 2;
        constexpr int kB = 5;
        StatusOr<std::vector<Index29>> cipher = AffineTransform{}.apply(
            plain,
            nlohmann::json{{"a", kA}, {"b", kB}},
            TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            AffineCandidateGenerator::generate(cipher.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 =
            require_unique_exact_match_rank1(candidates.value(), plain);
        const std::size_t expected_index =
            static_cast<std::size_t>(kA - 1) * 29u + static_cast<std::size_t>(kB);
        REQUIRE(rank1 == expected_index);
        REQUIRE(candidates.value()[rank1].params().at("a") == kA);
        REQUIRE(candidates.value()[rank1].params().at("b") == kB);
    }

    SECTION("gen_vigenere_explicit_keys") {
        const std::vector<Index29> correct_key = {I(1), I(2), I(5)};
        const std::vector<std::vector<Index29>> key_list = {
            {I(0), I(0), I(0)},
            correct_key,
            {I(4), I(4)},
            {I(1), I(2), I(6)},
        };

        StatusOr<std::vector<Index29>> cipher = VigenereKeyTransform{}.apply(
            plain,
            nlohmann::json{{"key_indices", {1, 2, 5}}},
            TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            VigenereExplicitKeyCandidateGenerator::generate(cipher.value(), key_list);
        REQUIRE(candidates.ok());
        REQUIRE(candidates.value().size() == key_list.size());
        const std::size_t rank1 =
            require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == 1);
        REQUIRE(candidates.value()[rank1].params().at("key_indices") == nlohmann::json{1, 2, 5});
    }

    SECTION("gen_vigenere_explicit_keys with interrupts") {
        const std::vector<Index29> correct_key = {I(1), I(2)};
        StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1, 3});
        REQUIRE(interrupt.ok());

        StatusOr<std::vector<Index29>> cipher = VigenereKeyTransform{}.apply(
            plain,
            nlohmann::json{{"key_indices", {1, 2}}},
            TransformDirection::Encrypt,
            interrupt.value());
        REQUIRE(cipher.ok());

        const std::vector<std::vector<Index29>> key_list = {
            {I(7), I(8)},
            correct_key,
            {I(1), I(3)},
        };
        StatusOr<std::vector<TransformCandidate>> candidates =
            VigenereExplicitKeyCandidateGenerator::generate(
                cipher.value(),
                key_list,
                TransformDirection::Decrypt,
                interrupt.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 =
            require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == 1);
    }
}

TEST_CASE("GeneratorRegistry lists gen_* ids and dispatches", "[generate][registry]") {
    const std::vector<std::string> ids = GeneratorRegistry::list_generator_ids();
    REQUIRE(ids.size() == 8);
    REQUIRE(ids[0] == "gen_atbash");
    REQUIRE(ids[1] == "gen_caesar");
    REQUIRE(ids[2] == "gen_atbash_caesar");
    REQUIRE(ids[3] == "gen_affine");
    REQUIRE(ids[4] == "gen_vigenere_explicit_keys");
    REQUIRE(ids[5] == "gen_beaufort_explicit_keys");
    REQUIRE(ids[6] == "gen_totient_offsets");
    REQUIRE(ids[7] == "gen_compose_recipes");
    REQUIRE(GeneratorRegistry::is_known("gen_caesar"));
    REQUIRE(GeneratorRegistry::is_known("gen_beaufort_explicit_keys"));
    REQUIRE(GeneratorRegistry::is_known("gen_totient_offsets"));
    REQUIRE(GeneratorRegistry::is_known("gen_compose_recipes"));
    REQUIRE_FALSE(GeneratorRegistry::is_known("gen_nope"));

    const std::vector<GeneratorCatalogEntry> entries = GeneratorRegistry::catalog();
    REQUIRE(entries.size() == 8);
    REQUIRE(entries[1].bounded_count() == 29);
    REQUIRE_FALSE(entries[1].requires_params());
    REQUIRE(entries[4].requires_params());
    REQUIRE(entries[4].bounded_count() == 0);
    REQUIRE(entries[5].requires_params());
    REQUIRE(entries[6].requires_params());
    REQUIRE(entries[7].requires_params());
    REQUIRE(entries[1].to_json().at("generator_id").get<std::string>() == "gen_caesar");

    const std::vector<Index29> cipher = {I(0), I(5), I(10)};
    StatusOr<std::vector<TransformCandidate>> via_registry =
        GeneratorRegistry::generate("gen_caesar", cipher);
    StatusOr<std::vector<TransformCandidate>> direct =
        CaesarCandidateGenerator::generate(cipher);
    REQUIRE(via_registry.ok());
    REQUIRE(direct.ok());
    REQUIRE(via_registry.value().size() == direct.value().size());
    REQUIRE(via_registry.value()[3].params() == direct.value()[3].params());
    REQUIRE(via_registry.value()[3].output_indices() == direct.value()[3].output_indices());

    StatusOr<std::vector<TransformCandidate>> atbash =
        GeneratorRegistry::generate("gen_atbash", cipher);
    REQUIRE(atbash.ok());
    REQUIRE(atbash.value().size() == 1);

    REQUIRE_FALSE(GeneratorRegistry::generate("gen_vigenere_explicit_keys", cipher).ok());
    REQUIRE_FALSE(GeneratorRegistry::generate("gen_beaufort_explicit_keys", cipher).ok());
    REQUIRE_FALSE(GeneratorRegistry::generate("gen_totient_offsets", cipher).ok());

    const nlohmann::json vig_params = {
        {"key_indices_list", {{1, 2, 3}, {4, 5}}},
    };
    StatusOr<std::vector<TransformCandidate>> vig =
        GeneratorRegistry::generate("gen_vigenere_explicit_keys", cipher, TransformDirection::Decrypt, vig_params);
    REQUIRE(vig.ok());
    REQUIRE(vig.value().size() == 2);

    StatusOr<std::vector<TransformCandidate>> beaufort =
        GeneratorRegistry::generate(
            "gen_beaufort_explicit_keys", cipher, TransformDirection::Decrypt, vig_params);
    REQUIRE(beaufort.ok());
    REQUIRE(beaufort.value().size() == 2);
    REQUIRE(beaufort.value()[0].transform_id() == TransformId::beaufort_key());

    const nlohmann::json totient_params = {{"prime_start_indices", {0, 1, 2}}};
    StatusOr<std::vector<TransformCandidate>> totient = GeneratorRegistry::generate(
        "gen_totient_offsets", cipher, TransformDirection::Decrypt, totient_params);
    REQUIRE(totient.ok());
    REQUIRE(totient.value().size() == 3);
    REQUIRE(totient.value()[0].transform_id() == TransformId::totient_prime_stream());

    StatusOr<std::vector<TransformCandidate>> compose_default =
        GeneratorRegistry::generate("gen_compose_recipes", cipher);
    REQUIRE(compose_default.ok());
    REQUIRE(compose_default.value().size() == 29);
    REQUIRE(compose_default.value()[0].transform_id() == TransformId::compose());

    const nlohmann::json one_recipe = {
        {"stages",
         nlohmann::json::array(
             {nlohmann::json{{"transform_id", "atbash"}, {"params", nlohmann::json::object()}},
              nlohmann::json{
                  {"transform_id", "caesar"},
                  {"direction", "encrypt"},
                  {"params", {{"shift", 3}}},
              }})}};
    StatusOr<std::vector<TransformCandidate>> compose_one = GeneratorRegistry::generate(
        "gen_compose_recipes", cipher, TransformDirection::Decrypt, one_recipe);
    REQUIRE(compose_one.ok());
    REQUIRE(compose_one.value().size() == 1);
    REQUIRE(compose_one.value()[0].candidate_id() == "atbash_caesar:shift=3");

    const nlohmann::json vig_keys = {
        {"keys",
         {{{"key_indices", {1, 2}}, {"key_latin", "BC"}},
          {{"key_indices", {3, 4, 5}}}}},
    };
    StatusOr<std::vector<TransformCandidate>> vig2 =
        GeneratorRegistry::generate(
            "gen_vigenere_explicit_keys", cipher, TransformDirection::Decrypt, vig_keys);
    REQUIRE(vig2.ok());
    REQUIRE(vig2.value().size() == 2);
    REQUIRE(vig2.value()[0].params().at("key_latin").get<std::string>() == "BC");

    REQUIRE_FALSE(GeneratorRegistry::generate("gen_unknown", cipher).ok());
}
