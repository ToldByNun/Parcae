#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/generate/affine_candidate_generator.hpp>
#include <parcae/generate/atbash_caesar_candidate_generator.hpp>
#include <parcae/generate/atbash_candidate_generator.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/generate/ciphertext_autokey_explicit_primer_candidate_generator.hpp>
#include <parcae/generate/generator_registry.hpp>
#include <parcae/generate/hill2_candidate_generator.hpp>
#include <parcae/generate/hill3_candidate_generator.hpp>
#include <parcae/generate/plaintext_autokey_explicit_primer_candidate_generator.hpp>
#include <parcae/generate/vigenere_explicit_key_candidate_generator.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/score/exact_match.hpp>
#include <parcae/transform/affine_transform.hpp>
#include <parcae/transform/atbash_transform.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/ciphertext_autokey_transform.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/hill2_transform.hpp>
#include <parcae/transform/hill3_transform.hpp>
#include <parcae/transform/plaintext_autokey_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <parcae/transform/vigenere_key_transform.hpp>
#include <string>
#include <vector>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

/// Among candidates, find unique exact-match=1.0 against `plain`; return its index.
[[nodiscard]] std::size_t
require_unique_exact_match_rank1(const std::vector<TransformCandidate>& candidates,
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

} // namespace

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

        StatusOr<std::vector<Index29>> expected =
            caesar.apply(cipher, nlohmann::json{{"shift", static_cast<int>(shift)}},
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

    StatusOr<double> exact = ExactMatch::score(decrypted.value()[kShift].output_indices(), plain);
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
    REQUIRE(c.output_indices() == std::vector<Index29>{I(28), I(24), I(14), I(0)});

    // Involution: applying again recovers ciphertext.
    StatusOr<std::vector<TransformCandidate>> again =
        AtbashCandidateGenerator::generate(c.output_indices());
    REQUIRE(again.ok());
    REQUIRE(again.value()[0].output_indices() == cipher);
}

TEST_CASE("AtbashCaesarCandidateGenerator emits 29 Koan-1-family shifts",
          "[generate][atbash_caesar]") {
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
    REQUIRE(candidates.value()[3].output_indices() == std::vector<Index29>{I(2), I(26), I(3)});
}

TEST_CASE("AffineCandidateGenerator enumerates 28x29=812 with documented cost",
          "[generate][affine]") {
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
    StatusOr<std::vector<Index29>> expected =
        affine.apply(cipher, nlohmann::json{{"a", 2}, {"b", 5}}, TransformDirection::Decrypt);
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

TEST_CASE("Hill2CandidateGenerator explicit matrices and seed sample", "[generate][hill2]") {
    REQUIRE(Hill2CandidateGenerator::generator_id == "gen_hill_2");
    REQUIRE(Hill2CandidateGenerator::default_max_candidates == 256);

    const std::vector<Index29> cipher = {I(4), I(6), I(1), I(0)};
    const nlohmann::json explicit_params{{"matrices", {{2, 3, 5, 7}, {1, 0, 0, 1}}}};

    StatusOr<std::vector<TransformCandidate>> candidates =
        Hill2CandidateGenerator::generate(cipher, TransformDirection::Decrypt, explicit_params);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 2);
    REQUIRE(candidates.value()[0].transform_id() == TransformId::hill_2());
    REQUIRE(candidates.value()[0].candidate_id() == "hill_2:i=0:matrix=2,3,5,7");
    REQUIRE(candidates.value()[1].params().at("matrix") == nlohmann::json{1, 0, 0, 1});

    const Hill2Transform hill;
    StatusOr<std::vector<Index29>> expected =
        hill.apply(cipher, nlohmann::json{{"matrix", {2, 3, 5, 7}}}, TransformDirection::Decrypt);
    REQUIRE(expected.ok());
    REQUIRE(candidates.value()[0].output_indices() == expected.value());

    // Encrypt with known matrix then recover via explicit decrypt list.
    StatusOr<std::vector<TransformCandidate>> encrypted = Hill2CandidateGenerator::generate(
        cipher, TransformDirection::Encrypt, nlohmann::json{{"matrices", {{2, 3, 5, 7}}}});
    REQUIRE(encrypted.ok());
    StatusOr<std::vector<TransformCandidate>> recovered = Hill2CandidateGenerator::generate(
        encrypted.value()[0].output_indices(), TransformDirection::Decrypt,
        nlohmann::json{{"matrices", {{2, 3, 5, 7}}}});
    REQUIRE(recovered.ok());
    REQUIRE(ExactMatch::score(recovered.value()[0].output_indices(), cipher).value() == 1.0);

    SECTION("rejects singular and odd length") {
        REQUIRE_FALSE(Hill2CandidateGenerator::generate(
                          cipher, TransformDirection::Decrypt,
                          nlohmann::json{{"matrices", {{1, 2, 2, 4}}}})
                          .ok());
        const std::vector<Index29> odd = {I(1), I(2), I(3)};
        REQUIRE_FALSE(Hill2CandidateGenerator::generate(odd, TransformDirection::Decrypt,
                                                        explicit_params)
                          .ok());
    }

    SECTION("seed sample is bounded deterministic and invertible") {
        const nlohmann::json sample_params{{"max_candidates", 16}, {"seed", 7}};
        StatusOr<std::vector<TransformCandidate>> a =
            Hill2CandidateGenerator::generate(cipher, TransformDirection::Decrypt, sample_params);
        StatusOr<std::vector<TransformCandidate>> b =
            Hill2CandidateGenerator::generate(cipher, TransformDirection::Decrypt, sample_params);
        REQUIRE(a.ok());
        REQUIRE(b.ok());
        REQUIRE(a.value().size() == 16);
        REQUIRE(b.value().size() == 16);
        for (std::size_t i = 0; i < a.value().size(); ++i) {
            REQUIRE(a.value()[i].candidate_id() == b.value()[i].candidate_id());
            REQUIRE(a.value()[i].params() == b.value()[i].params());
        }
        StatusOr<std::vector<TransformCandidate>> defaults =
            Hill2CandidateGenerator::generate(cipher);
        REQUIRE(defaults.ok());
        REQUIRE(defaults.value().size() == Hill2CandidateGenerator::default_max_candidates);
    }
}

TEST_CASE("Hill3CandidateGenerator explicit matrices and seed sample", "[generate][hill3]") {
    REQUIRE(Hill3CandidateGenerator::generator_id == "gen_hill_3");
    REQUIRE(Hill3CandidateGenerator::default_max_candidates == 128);

    const std::vector<Index29> cipher = {I(2), I(3), I(4), I(5), I(6), I(7)};
    const nlohmann::json explicit_params{
        {"matrices", {{1, 2, 3, 0, 1, 4, 5, 6, 0}, {1, 0, 0, 0, 1, 0, 0, 0, 1}}}};

    StatusOr<std::vector<TransformCandidate>> candidates =
        Hill3CandidateGenerator::generate(cipher, TransformDirection::Decrypt, explicit_params);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 2);
    REQUIRE(candidates.value()[0].transform_id() == TransformId::hill_3());

    const Hill3Transform hill;
    StatusOr<std::vector<Index29>> expected = hill.apply(
        cipher, nlohmann::json{{"matrix", {1, 2, 3, 0, 1, 4, 5, 6, 0}}}, TransformDirection::Decrypt);
    REQUIRE(expected.ok());
    REQUIRE(candidates.value()[0].output_indices() == expected.value());

    SECTION("rejects bad length and singular") {
        const std::vector<Index29> bad_len = {I(1), I(2), I(3), I(4)};
        REQUIRE_FALSE(Hill3CandidateGenerator::generate(bad_len, TransformDirection::Decrypt,
                                                        explicit_params)
                          .ok());
        REQUIRE_FALSE(Hill3CandidateGenerator::generate(
                          cipher, TransformDirection::Decrypt,
                          nlohmann::json{{"matrices", {{1, 2, 3, 2, 4, 6, 0, 1, 0}}}})
                          .ok());
    }

    SECTION("seed sample capped") {
        StatusOr<std::vector<TransformCandidate>> sampled = Hill3CandidateGenerator::generate(
            cipher, TransformDirection::Decrypt,
            nlohmann::json{{"max_candidates", 8}, {"seed", 3}});
        REQUIRE(sampled.ok());
        REQUIRE(sampled.value().size() == 8);
    }
}

TEST_CASE("VigenereExplicitKeyCandidateGenerator applies caller keys only",
          "[generate][vigenere]") {
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
                cipher, keys, TransformDirection::Decrypt, interrupt.value());
        REQUIRE(candidates.ok());
        REQUIRE(candidates.value().size() == 1);
        REQUIRE(candidates.value()[0].params().at("key_latin") == "AB");
        REQUIRE(candidates.value()[0].interrupt().has_value());
        REQUIRE(candidates.value()[0].envelope().contains("interrupt"));
        REQUIRE(candidates.value()[0].envelope().at("interrupt").at("skip_indices") ==
                nlohmann::json{1});
    }
}

TEST_CASE("Generators include params that recover known synthetic ciphertexts at rank 1 under "
          "exact-match",
          "[generate][rank1]") {
    const std::vector<Index29> plain = {I(0), I(1), I(2), I(3), I(10), I(14), I(28)};

    SECTION("gen_caesar") {
        constexpr int kShift = 7;
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain, nlohmann::json{{"shift", kShift}}, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            CaesarCandidateGenerator::generate(cipher.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 = require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == static_cast<std::size_t>(kShift));
        REQUIRE(candidates.value()[rank1].params().at("shift") == kShift);
    }

    SECTION("gen_atbash") {
        StatusOr<std::vector<Index29>> cipher =
            AtbashTransform{}.apply(plain, nlohmann::json::object(), TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            AtbashCandidateGenerator::generate(cipher.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 = require_unique_exact_match_rank1(candidates.value(), plain);
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
        const std::size_t rank1 = require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == kShift);
        REQUIRE(candidates.value()[rank1].params() ==
                ComposeTransform::atbash_then_caesar_params(kShift));
    }

    SECTION("gen_affine") {
        constexpr int kA = 2;
        constexpr int kB = 5;
        StatusOr<std::vector<Index29>> cipher = AffineTransform{}.apply(
            plain, nlohmann::json{{"a", kA}, {"b", kB}}, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            AffineCandidateGenerator::generate(cipher.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 = require_unique_exact_match_rank1(candidates.value(), plain);
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
            plain, nlohmann::json{{"key_indices", {1, 2, 5}}}, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());

        StatusOr<std::vector<TransformCandidate>> candidates =
            VigenereExplicitKeyCandidateGenerator::generate(cipher.value(), key_list);
        REQUIRE(candidates.ok());
        REQUIRE(candidates.value().size() == key_list.size());
        const std::size_t rank1 = require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == 1);
        REQUIRE(candidates.value()[rank1].params().at("key_indices") == nlohmann::json{1, 2, 5});
    }

    SECTION("gen_vigenere_explicit_keys with interrupts") {
        const std::vector<Index29> correct_key = {I(1), I(2)};
        StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1, 3});
        REQUIRE(interrupt.ok());

        StatusOr<std::vector<Index29>> cipher =
            VigenereKeyTransform{}.apply(plain, nlohmann::json{{"key_indices", {1, 2}}},
                                         TransformDirection::Encrypt, interrupt.value());
        REQUIRE(cipher.ok());

        const std::vector<std::vector<Index29>> key_list = {
            {I(7), I(8)},
            correct_key,
            {I(1), I(3)},
        };
        StatusOr<std::vector<TransformCandidate>> candidates =
            VigenereExplicitKeyCandidateGenerator::generate(
                cipher.value(), key_list, TransformDirection::Decrypt, interrupt.value());
        REQUIRE(candidates.ok());
        const std::size_t rank1 = require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == 1);
    }
}

TEST_CASE("CiphertextAutokeyExplicitPrimerCandidateGenerator applies primers only",
          "[generate][autokey][ctak]") {
    REQUIRE(CiphertextAutokeyExplicitPrimerCandidateGenerator::generator_id ==
            "gen_ciphertext_autokey_explicit_primers");

    const std::vector<Index29> plain = {I(1), I(2), I(4), I(7), I(8)};
    const nlohmann::json primer_params{{"key_indices", {3, 5}}};
    StatusOr<std::vector<Index29>> cipher = CiphertextAutokeyTransform{}.apply(
        plain, primer_params, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    SECTION("rejects empty primer list") {
        const std::vector<std::vector<Index29>> empty;
        REQUIRE_FALSE(
            CiphertextAutokeyExplicitPrimerCandidateGenerator::generate(cipher.value(), empty)
                .ok());
    }

    SECTION("enumerates primers and recovers plaintext") {
        const std::vector<std::vector<Index29>> primers = {
            {I(1), I(1)},
            {I(3), I(5)},
            {I(9), I(9)},
        };
        StatusOr<std::vector<TransformCandidate>> candidates =
            CiphertextAutokeyExplicitPrimerCandidateGenerator::generate(cipher.value(), primers);
        REQUIRE(candidates.ok());
        REQUIRE(candidates.value().size() == 3);
        REQUIRE(candidates.value()[1].candidate_id() ==
                "ciphertext_autokey:i=1:key_indices=3,5");
        REQUIRE(candidates.value()[1].transform_id() == TransformId::ciphertext_autokey());
        const std::size_t rank1 =
            require_unique_exact_match_rank1(candidates.value(), plain);
        REQUIRE(rank1 == 1);
    }

    SECTION("interrupt roundtrip via generator") {
        StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1});
        REQUIRE(interrupt.ok());
        StatusOr<std::vector<Index29>> cipher_i = CiphertextAutokeyTransform{}.apply(
            plain, primer_params, TransformDirection::Encrypt, interrupt.value());
        REQUIRE(cipher_i.ok());
        const std::vector<std::vector<Index29>> primers = {{I(3), I(5)}};
        StatusOr<std::vector<TransformCandidate>> candidates =
            CiphertextAutokeyExplicitPrimerCandidateGenerator::generate(
                cipher_i.value(), primers, TransformDirection::Decrypt, interrupt.value());
        REQUIRE(candidates.ok());
        REQUIRE(candidates.value()[0].output_indices() == plain);
        REQUIRE(candidates.value()[0].interrupt().has_value());
    }
}

TEST_CASE("PlaintextAutokeyExplicitPrimerCandidateGenerator applies primers only",
          "[generate][autokey][ptak]") {
    REQUIRE(PlaintextAutokeyExplicitPrimerCandidateGenerator::generator_id ==
            "gen_plaintext_autokey_explicit_primers");

    const std::vector<Index29> plain = {I(1), I(2), I(4), I(7), I(8)};
    const nlohmann::json primer_params{{"key_indices", {3, 5}}};
    StatusOr<std::vector<Index29>> cipher =
        PlaintextAutokeyTransform{}.apply(plain, primer_params, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    const std::vector<ExplicitVigenereKey> primers = {
        ExplicitVigenereKey{{I(0), I(1)}, std::nullopt},
        ExplicitVigenereKey{{I(3), I(5)}, std::string{"CF"}},
    };
    StatusOr<std::vector<TransformCandidate>> candidates =
        PlaintextAutokeyExplicitPrimerCandidateGenerator::generate(cipher.value(), primers);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 2);
    REQUIRE(candidates.value()[1].params().at("key_latin").get<std::string>() == "CF");
    REQUIRE(ExactMatch::score(candidates.value()[1].output_indices(), plain).value() == 1.0);
    REQUIRE(ExactMatch::score(candidates.value()[0].output_indices(), plain).value() == 0.0);
}

TEST_CASE("GeneratorRegistry lists gen_* ids and dispatches", "[generate][registry]") {
    const std::vector<std::string> ids = GeneratorRegistry::list_generator_ids();
    REQUIRE(ids.size() == 12);
    REQUIRE(ids[0] == "gen_atbash");
    REQUIRE(ids[1] == "gen_caesar");
    REQUIRE(ids[2] == "gen_atbash_caesar");
    REQUIRE(ids[3] == "gen_affine");
    REQUIRE(ids[4] == "gen_hill_2");
    REQUIRE(ids[5] == "gen_hill_3");
    REQUIRE(ids[6] == "gen_vigenere_explicit_keys");
    REQUIRE(ids[7] == "gen_beaufort_explicit_keys");
    REQUIRE(ids[8] == "gen_ciphertext_autokey_explicit_primers");
    REQUIRE(ids[9] == "gen_plaintext_autokey_explicit_primers");
    REQUIRE(ids[10] == "gen_totient_offsets");
    REQUIRE(ids[11] == "gen_compose_recipes");
    REQUIRE(GeneratorRegistry::is_known("gen_caesar"));
    REQUIRE(GeneratorRegistry::is_known("gen_hill_2"));
    REQUIRE(GeneratorRegistry::is_known("gen_hill_3"));
    REQUIRE(GeneratorRegistry::is_known("gen_beaufort_explicit_keys"));
    REQUIRE(GeneratorRegistry::is_known("gen_ciphertext_autokey_explicit_primers"));
    REQUIRE(GeneratorRegistry::is_known("gen_plaintext_autokey_explicit_primers"));
    REQUIRE(GeneratorRegistry::is_known("gen_totient_offsets"));
    REQUIRE(GeneratorRegistry::is_known("gen_compose_recipes"));
    REQUIRE_FALSE(GeneratorRegistry::is_known("gen_nope"));

    const std::vector<GeneratorCatalogEntry> entries = GeneratorRegistry::catalog();
    REQUIRE(entries.size() == 12);
    REQUIRE(entries[1].bounded_count() == 29);
    REQUIRE_FALSE(entries[1].requires_params());
    REQUIRE(entries[4].requires_params());
    REQUIRE(entries[4].bounded_count() == 0);
    REQUIRE(entries[5].requires_params());
    REQUIRE(entries[6].requires_params());
    REQUIRE(entries[6].bounded_count() == 0);
    REQUIRE(entries[7].requires_params());
    REQUIRE(entries[8].requires_params());
    REQUIRE(entries[9].requires_params());
    REQUIRE(entries[10].requires_params());
    REQUIRE(entries[11].requires_params());
    REQUIRE(entries[1].to_json().at("generator_id").get<std::string>() == "gen_caesar");

    const std::vector<Index29> cipher = {I(0), I(5), I(10)};
    StatusOr<std::vector<TransformCandidate>> via_registry =
        GeneratorRegistry::generate("gen_caesar", cipher);
    StatusOr<std::vector<TransformCandidate>> direct = CaesarCandidateGenerator::generate(cipher);
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
    REQUIRE_FALSE(
        GeneratorRegistry::generate("gen_ciphertext_autokey_explicit_primers", cipher).ok());
    REQUIRE_FALSE(
        GeneratorRegistry::generate("gen_plaintext_autokey_explicit_primers", cipher).ok());
    REQUIRE_FALSE(GeneratorRegistry::generate("gen_totient_offsets", cipher).ok());

    const std::vector<Index29> hill_cipher = {I(0), I(5), I(10), I(15)};
    StatusOr<std::vector<TransformCandidate>> hill2 = GeneratorRegistry::generate(
        "gen_hill_2", hill_cipher, TransformDirection::Decrypt,
        nlohmann::json{{"matrices", {{2, 3, 5, 7}}}});
    REQUIRE(hill2.ok());
    REQUIRE(hill2.value().size() == 1);
    REQUIRE(hill2.value()[0].transform_id() == TransformId::hill_2());

    const std::vector<Index29> hill3_cipher = {I(0), I(5), I(10), I(15), I(20), I(25)};
    StatusOr<std::vector<TransformCandidate>> hill3 = GeneratorRegistry::generate(
        "gen_hill_3", hill3_cipher, TransformDirection::Decrypt,
        nlohmann::json{{"max_candidates", 4}, {"seed", 2}});
    REQUIRE(hill3.ok());
    REQUIRE(hill3.value().size() == 4);
    REQUIRE(hill3.value()[0].transform_id() == TransformId::hill_3());

    const nlohmann::json vig_params = {
        {"key_indices_list", {{1, 2, 3}, {4, 5}}},
    };
    StatusOr<std::vector<TransformCandidate>> vig = GeneratorRegistry::generate(
        "gen_vigenere_explicit_keys", cipher, TransformDirection::Decrypt, vig_params);
    REQUIRE(vig.ok());
    REQUIRE(vig.value().size() == 2);

    StatusOr<std::vector<TransformCandidate>> beaufort = GeneratorRegistry::generate(
        "gen_beaufort_explicit_keys", cipher, TransformDirection::Decrypt, vig_params);
    REQUIRE(beaufort.ok());
    REQUIRE(beaufort.value().size() == 2);
    REQUIRE(beaufort.value()[0].transform_id() == TransformId::beaufort_key());

    StatusOr<std::vector<TransformCandidate>> ctak = GeneratorRegistry::generate(
        "gen_ciphertext_autokey_explicit_primers", cipher, TransformDirection::Decrypt, vig_params);
    REQUIRE(ctak.ok());
    REQUIRE(ctak.value().size() == 2);
    REQUIRE(ctak.value()[0].transform_id() == TransformId::ciphertext_autokey());

    StatusOr<std::vector<TransformCandidate>> ptak = GeneratorRegistry::generate(
        "gen_plaintext_autokey_explicit_primers", cipher, TransformDirection::Decrypt, vig_params);
    REQUIRE(ptak.ok());
    REQUIRE(ptak.value().size() == 2);
    REQUIRE(ptak.value()[0].transform_id() == TransformId::plaintext_autokey());

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
        {"stages", nlohmann::json::array({nlohmann::json{{"transform_id", "atbash"},
                                                         {"params", nlohmann::json::object()}},
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
        {"keys", {{{"key_indices", {1, 2}}, {"key_latin", "BC"}}, {{"key_indices", {3, 4, 5}}}}},
    };
    StatusOr<std::vector<TransformCandidate>> vig2 = GeneratorRegistry::generate(
        "gen_vigenere_explicit_keys", cipher, TransformDirection::Decrypt, vig_keys);
    REQUIRE(vig2.ok());
    REQUIRE(vig2.value().size() == 2);
    REQUIRE(vig2.value()[0].params().at("key_latin").get<std::string>() == "BC");

    REQUIRE_FALSE(GeneratorRegistry::generate("gen_unknown", cipher).ok());
}
