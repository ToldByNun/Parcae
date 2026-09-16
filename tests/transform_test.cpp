#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/atbash_transform.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/identity_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>

#include <catch2/catch_test_macros.hpp>

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
