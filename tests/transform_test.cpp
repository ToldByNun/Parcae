#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/affine_transform.hpp>
#include <parcae/transform/atbash_transform.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/identity_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

TEST_CASE("TransformId serializes known catalog ids", "[transform]") {
    REQUIRE(TransformId::identity().str() == "identity");
    REQUIRE(TransformId::atbash().str() == "atbash");
    REQUIRE(TransformId::vigenere_key().str() == "vigenere_key");
    REQUIRE(TransformId::totient_prime_stream().str() == "totient_prime_stream");

    StatusOr<TransformId> parsed = TransformId::from_string("compose");
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value() == TransformId::compose());

    StatusOr<TransformId> unknown = TransformId::from_string("not-a-transform");
    REQUIRE_FALSE(unknown.ok());
}

TEST_CASE("TransformDirection round-trips", "[transform]") {
    REQUIRE(TransformDirectionUtil::to_string(TransformDirection::Decrypt) == "decrypt");
    REQUIRE(TransformDirectionUtil::to_string(TransformDirection::Encrypt) == "encrypt");

    StatusOr<TransformDirection> decrypt = TransformDirectionUtil::from_string("decrypt");
    REQUIRE(decrypt.ok());
    REQUIRE(decrypt.value() == TransformDirection::Decrypt);

    StatusOr<TransformDirection> bad = TransformDirectionUtil::from_string("sideways");
    REQUIRE_FALSE(bad.ok());
}

TEST_CASE("InterruptPolicy parses skip indices and queries skips", "[transform][interrupt]") {
    StatusOr<InterruptPolicy> policy = InterruptPolicy::from_skip_indices({56, 49, 56});
    REQUIRE_FALSE(policy.ok());

    StatusOr<InterruptPolicy> ok = InterruptPolicy::from_skip_indices({56, 49});
    REQUIRE(ok.ok());
    REQUIRE(ok.value().skip_indices() == std::vector<std::size_t>{49, 56});
    REQUIRE(ok.value().should_skip(49));
    REQUIRE(ok.value().should_skip(56));
    REQUIRE_FALSE(ok.value().should_skip(50));

    nlohmann::json encoded = ok.value().to_json();
    StatusOr<InterruptPolicy> round_trip = InterruptPolicy::from_json(encoded);
    REQUIRE(round_trip.ok());
    REQUIRE(round_trip.value().skip_indices() == ok.value().skip_indices());
}

TEST_CASE("IdentityTransform copies Index29 span", "[transform]") {
    const IdentityTransform transform;
    REQUIRE(transform.id() == TransformId::identity());

    const std::vector<Index29> input{Index29{0}, Index29{7}, Index29{28}, Index29{14}};
    StatusOr<std::vector<Index29>> out = transform.apply(
        input,
        nlohmann::json::object(),
        TransformDirection::Decrypt);
    REQUIRE(out.ok());
    REQUIRE(out.value() == input);

    StatusOr<std::vector<Index29>> encrypt = transform.apply(
        input,
        nlohmann::json::object(),
        TransformDirection::Encrypt,
        InterruptPolicy::none());
    REQUIRE(encrypt.ok());
    REQUIRE(encrypt.value() == input);
}

TEST_CASE("IdentityTransform rejects non-empty params", "[transform]") {
    const IdentityTransform transform;
    const std::vector<Index29> input{Index29{1}};
    StatusOr<std::vector<Index29>> bad = transform.apply(
        input,
        nlohmann::json{{"shift", 1}},
        TransformDirection::Decrypt);
    REQUIRE_FALSE(bad.ok());
}

TEST_CASE("IdentityTransform via Transform interface", "[transform]") {
    const IdentityTransform concrete;
    const Transform& transform = concrete;

    const std::vector<Index29> input{Index29{3}, Index29{5}};
    StatusOr<std::vector<Index29>> out =
        transform.apply(input, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(out.ok());
    REQUIRE(out.value().size() == input.size());
    REQUIRE(out.value()[0] == Index29{3});
    REQUIRE(out.value()[1] == Index29{5});
}

TEST_CASE("AtbashTransform is 28 - x and an involution", "[transform]") {
    const AtbashTransform transform;
    REQUIRE(transform.id() == TransformId::atbash());

    const std::vector<Index29> input{Index29{0}, Index29{1}, Index29{14}, Index29{28}};
    StatusOr<std::vector<Index29>> once = transform.apply(
        input,
        nlohmann::json::object(),
        TransformDirection::Decrypt);
    REQUIRE(once.ok());
    REQUIRE(once.value() == std::vector<Index29>{Index29{28}, Index29{27}, Index29{14}, Index29{0}});

    StatusOr<std::vector<Index29>> twice = transform.apply(
        once.value(),
        nlohmann::json::object(),
        TransformDirection::Encrypt);
    REQUIRE(twice.ok());
    REQUIRE(twice.value() == input);
}

TEST_CASE("AtbashTransform rejects non-empty params", "[transform]") {
    const AtbashTransform transform;
    StatusOr<std::vector<Index29>> bad = transform.apply(
        std::vector<Index29>{Index29{2}},
        nlohmann::json{{"shift", 1}},
        TransformDirection::Decrypt);
    REQUIRE_FALSE(bad.ok());
}

TEST_CASE("CaesarTransform encrypt adds and decrypt subtracts shift", "[transform]") {
    const CaesarTransform transform;
    REQUIRE(transform.id() == TransformId::caesar());

    const std::vector<Index29> plain{Index29{0}, Index29{26}, Index29{28}};
    const nlohmann::json params{{"shift", 3}};

    StatusOr<std::vector<Index29>> cipher =
        transform.apply(plain, params, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value() == std::vector<Index29>{Index29{3}, Index29{0}, Index29{2}});

    StatusOr<std::vector<Index29>> recovered =
        transform.apply(cipher.value(), params, TransformDirection::Decrypt);
    REQUIRE(recovered.ok());
    REQUIRE(recovered.value() == plain);
}

TEST_CASE("CaesarTransform validates shift range and params shape", "[transform]") {
    const CaesarTransform transform;
    const std::vector<Index29> input{Index29{1}};

    REQUIRE_FALSE(
        transform.apply(input, nlohmann::json::object(), TransformDirection::Decrypt).ok());
    REQUIRE_FALSE(
        transform.apply(input, nlohmann::json{{"shift", 29}}, TransformDirection::Decrypt).ok());
    REQUIRE_FALSE(
        transform.apply(input, nlohmann::json{{"shift", -1}}, TransformDirection::Decrypt).ok());
    REQUIRE_FALSE(transform
                      .apply(
                          input,
                          nlohmann::json{{"shift", 3}, {"extra", true}},
                          TransformDirection::Decrypt)
                      .ok());

    StatusOr<std::vector<Index29>> zero =
        transform.apply(input, nlohmann::json{{"shift", 0}}, TransformDirection::Encrypt);
    REQUIRE(zero.ok());
    REQUIRE(zero.value() == input);
}

TEST_CASE("AffineTransform encrypt/decrypt uses modular inverse", "[transform]") {
    const AffineTransform transform;
    REQUIRE(transform.id() == TransformId::affine());

    // inv(2) = 15 because 2 * 15 = 30 ≡ 1 (mod 29).
    REQUIRE(Z29::mul(Index29{2}, Z29::inv(Index29{2})).value() == 1);

    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{10}, Index29{14}};
    const nlohmann::json params{{"a", 2}, {"b", 5}};

    StatusOr<std::vector<Index29>> cipher =
        transform.apply(plain, params, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value() == std::vector<Index29>{Index29{5}, Index29{7}, Index29{25}, Index29{4}});

    StatusOr<std::vector<Index29>> recovered =
        transform.apply(cipher.value(), params, TransformDirection::Decrypt);
    REQUIRE(recovered.ok());
    REQUIRE(recovered.value() == plain);
}

TEST_CASE("AffineTransform rejects invalid a/b", "[transform]") {
    const AffineTransform transform;
    const std::vector<Index29> input{Index29{3}};
    REQUIRE_FALSE(
        transform.apply(input, nlohmann::json{{"a", 0}, {"b", 1}}, TransformDirection::Encrypt)
            .ok());
    REQUIRE_FALSE(
        transform.apply(input, nlohmann::json{{"a", 2}, {"b", 29}}, TransformDirection::Encrypt)
            .ok());
}

TEST_CASE("ComposeTransform applies stages in order and reverses on encrypt", "[transform]") {
    const ComposeTransform transform;
    REQUIRE(transform.id() == TransformId::compose());
    REQUIRE(ComposeTransform::max_depth >= 4);

    const std::vector<Index29> plain{Index29{0}, Index29{10}};
    const nlohmann::json params{
        {"stages",
         nlohmann::json::array(
             {nlohmann::json{{"transform_id", "identity"}, {"params", nlohmann::json::object()}},
              nlohmann::json{{"transform_id", "caesar"}, {"params", {{"shift", 3}}}}})},
    };

    StatusOr<std::vector<Index29>> cipher =
        transform.apply(plain, params, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value() == std::vector<Index29>{Index29{3}, Index29{13}});

    StatusOr<std::vector<Index29>> recovered =
        transform.apply(cipher.value(), params, TransformDirection::Decrypt);
    REQUIRE(recovered.ok());
    REQUIRE(recovered.value() == plain);
}

TEST_CASE("Compose atbash_then_caesar helper matches Koan 1 path", "[transform]") {
    // Decrypt: atbash then +3. Encrypt: -3 then atbash.
    const std::vector<Index29> cipher{Index29{0}, Index29{5}, Index29{28}};
    StatusOr<std::vector<Index29>> plain = ComposeTransform::apply_atbash_then_caesar(
        cipher,
        3,
        TransformDirection::Decrypt);
    REQUIRE(plain.ok());
    REQUIRE(plain.value() == std::vector<Index29>{Index29{2}, Index29{26}, Index29{3}});

    for (std::size_t i = 0; i < cipher.size(); ++i) {
        const auto expected = static_cast<std::uint8_t>(
            (28 - cipher[i].value() + 3) % Index29::modulus);
        REQUIRE(plain.value()[i].value() == expected);
    }

    StatusOr<std::vector<Index29>> round_trip = ComposeTransform::apply_atbash_then_caesar(
        plain.value(),
        3,
        TransformDirection::Encrypt);
    REQUIRE(round_trip.ok());
    REQUIRE(round_trip.value() == cipher);

    REQUIRE(
        ComposeTransform::atbash_then_caesar_params(3).at("stages").size() == 2);
}

TEST_CASE("Synthetic primitives match hand vectors", "[transform][synth]") {
    SECTION("synth-atbash-01") {
        const AtbashTransform transform;
        const std::vector<Index29> input{Index29{0}, Index29{4}, Index29{14}, Index29{28}};
        StatusOr<std::vector<Index29>> out = transform.apply(
            input,
            nlohmann::json::object(),
            TransformDirection::Decrypt);
        REQUIRE(out.ok());
        REQUIRE(
            out.value() ==
            std::vector<Index29>{Index29{28}, Index29{24}, Index29{14}, Index29{0}});
    }

    SECTION("synth-caesar-b3") {
        const CaesarTransform transform;
        const std::vector<Index29> plain{Index29{0}, Index29{10}, Index29{28}};
        const nlohmann::json params{{"shift", 3}};
        StatusOr<std::vector<Index29>> cipher =
            transform.apply(plain, params, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());
        REQUIRE(cipher.value() == std::vector<Index29>{Index29{3}, Index29{13}, Index29{2}});
        StatusOr<std::vector<Index29>> recovered =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt);
        REQUIRE(recovered.ok());
        REQUIRE(recovered.value() == plain);
    }

    SECTION("synth-affine-a2b5") {
        const AffineTransform transform;
        const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{10}, Index29{14}};
        const nlohmann::json params{{"a", 2}, {"b", 5}};
        StatusOr<std::vector<Index29>> cipher =
            transform.apply(plain, params, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());
        REQUIRE(
            cipher.value() ==
            std::vector<Index29>{Index29{5}, Index29{7}, Index29{25}, Index29{4}});
        StatusOr<std::vector<Index29>> recovered =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt);
        REQUIRE(recovered.ok());
        REQUIRE(recovered.value() == plain);
    }

    SECTION("synth-koan1-atbash-then-caesar-plus3") {
        const std::vector<Index29> cipher{Index29{4}, Index29{11}, Index29{20}};
        StatusOr<std::vector<Index29>> plain = ComposeTransform::apply_atbash_then_caesar(
            cipher,
            3,
            TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        REQUIRE(plain.value() == std::vector<Index29>{Index29{27}, Index29{20}, Index29{11}});
        StatusOr<std::vector<Index29>> back = ComposeTransform::apply_atbash_then_caesar(
            plain.value(),
            3,
            TransformDirection::Encrypt);
        REQUIRE(back.ok());
        REQUIRE(back.value() == cipher);
    }
}
