#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/affine_transform.hpp>
#include <parcae/transform/atbash_transform.hpp>
#include <parcae/transform/beaufort_key_transform.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/identity_transform.hpp>
#include <parcae/transform/totient_prime_stream_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/transform_id.hpp>
#include <parcae/transform/vigenere_key_transform.hpp>
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
    StatusOr<std::vector<Index29>> out =
        transform.apply(input, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(out.ok());
    REQUIRE(out.value() == input);

    StatusOr<std::vector<Index29>> encrypt = transform.apply(
        input, nlohmann::json::object(), TransformDirection::Encrypt, InterruptPolicy::none());
    REQUIRE(encrypt.ok());
    REQUIRE(encrypt.value() == input);
}

TEST_CASE("IdentityTransform rejects non-empty params", "[transform]") {
    const IdentityTransform transform;
    const std::vector<Index29> input{Index29{1}};
    StatusOr<std::vector<Index29>> bad =
        transform.apply(input, nlohmann::json{{"shift", 1}}, TransformDirection::Decrypt);
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
    StatusOr<std::vector<Index29>> once =
        transform.apply(input, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(once.ok());
    REQUIRE(once.value() ==
            std::vector<Index29>{Index29{28}, Index29{27}, Index29{14}, Index29{0}});

    StatusOr<std::vector<Index29>> twice =
        transform.apply(once.value(), nlohmann::json::object(), TransformDirection::Encrypt);
    REQUIRE(twice.ok());
    REQUIRE(twice.value() == input);
}

TEST_CASE("AtbashTransform rejects non-empty params", "[transform]") {
    const AtbashTransform transform;
    StatusOr<std::vector<Index29>> bad =
        transform.apply(std::vector<Index29>{Index29{2}}, nlohmann::json{{"shift", 1}},
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
                      .apply(input, nlohmann::json{{"shift", 3}, {"extra", true}},
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
    REQUIRE(cipher.value() ==
            std::vector<Index29>{Index29{5}, Index29{7}, Index29{25}, Index29{4}});

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
    StatusOr<std::vector<Index29>> plain =
        ComposeTransform::apply_atbash_then_caesar(cipher, 3, TransformDirection::Decrypt);
    REQUIRE(plain.ok());
    REQUIRE(plain.value() == std::vector<Index29>{Index29{2}, Index29{26}, Index29{3}});

    for (std::size_t i = 0; i < cipher.size(); ++i) {
        const auto expected =
            static_cast<std::uint8_t>((28 - cipher[i].value() + 3) % Index29::modulus);
        REQUIRE(plain.value()[i].value() == expected);
    }

    StatusOr<std::vector<Index29>> round_trip =
        ComposeTransform::apply_atbash_then_caesar(plain.value(), 3, TransformDirection::Encrypt);
    REQUIRE(round_trip.ok());
    REQUIRE(round_trip.value() == cipher);

    const ComposeTransform compose;
    StatusOr<std::vector<Index29>> via_params = compose.apply(
        cipher, ComposeTransform::atbash_then_caesar_params(3), TransformDirection::Decrypt);
    REQUIRE(via_params.ok());
    REQUIRE(via_params.value() == plain.value());
}

TEST_CASE("Synthetic primitives match hand vectors", "[transform][synth]") {
    SECTION("synth-atbash-01") {
        const AtbashTransform transform;
        const std::vector<Index29> input{Index29{0}, Index29{4}, Index29{14}, Index29{28}};
        StatusOr<std::vector<Index29>> out =
            transform.apply(input, nlohmann::json::object(), TransformDirection::Decrypt);
        REQUIRE(out.ok());
        REQUIRE(out.value() ==
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
        REQUIRE(cipher.value() ==
                std::vector<Index29>{Index29{5}, Index29{7}, Index29{25}, Index29{4}});
        StatusOr<std::vector<Index29>> recovered =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt);
        REQUIRE(recovered.ok());
        REQUIRE(recovered.value() == plain);
    }

    SECTION("synth-koan1-atbash-then-caesar-plus3") {
        const std::vector<Index29> cipher{Index29{4}, Index29{11}, Index29{20}};
        StatusOr<std::vector<Index29>> plain =
            ComposeTransform::apply_atbash_then_caesar(cipher, 3, TransformDirection::Decrypt);
        REQUIRE(plain.ok());
        REQUIRE(plain.value() == std::vector<Index29>{Index29{27}, Index29{20}, Index29{11}});
        StatusOr<std::vector<Index29>> back = ComposeTransform::apply_atbash_then_caesar(
            plain.value(), 3, TransformDirection::Encrypt);
        REQUIRE(back.ok());
        REQUIRE(back.value() == cipher);
    }
}

TEST_CASE("VigenereKeyTransform id and params", "[transform]") {
    const VigenereKeyTransform transform;
    REQUIRE(transform.id() == TransformId::vigenere_key());

    const std::vector<Index29> plain{Index29{0}};
    REQUIRE_FALSE(
        transform.apply(plain, nlohmann::json::object(), TransformDirection::Encrypt).ok());
    REQUIRE_FALSE(transform
                      .apply(plain, nlohmann::json{{"key_indices", nlohmann::json::array()}},
                             TransformDirection::Encrypt)
                      .ok());
    REQUIRE_FALSE(
        transform.apply(plain, nlohmann::json{{"key_indices", {29}}}, TransformDirection::Encrypt)
            .ok());
}

TEST_CASE("Synthetic Vigenere hand vectors with and without interrupts", "[transform]") {
    const VigenereKeyTransform transform;
    const nlohmann::json params{{"key_indices", {1, 2}}};
    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{2}, Index29{3}};

    SECTION("synth-vigenere-no-interrupt") {
        StatusOr<std::vector<Index29>> cipher =
            transform.apply(plain, params, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());
        // (0+1), (1+2), (2+1), (3+2)
        REQUIRE(cipher.value() ==
                std::vector<Index29>{Index29{1}, Index29{3}, Index29{3}, Index29{5}});

        StatusOr<std::vector<Index29>> recovered =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt);
        REQUIRE(recovered.ok());
        REQUIRE(recovered.value() == plain);
    }

    SECTION("synth-vigenere-with-interrupts") {
        StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1, 3});
        REQUIRE(interrupt.ok());

        StatusOr<std::vector<Index29>> cipher =
            transform.apply(plain, params, TransformDirection::Encrypt, interrupt.value());
        REQUIRE(cipher.ok());
        // i=0 consume key[0]=1 → 1; i=1 skip → 1; i=2 consume key[1]=2 → 4; i=3 skip → 3
        REQUIRE(cipher.value() ==
                std::vector<Index29>{Index29{1}, Index29{1}, Index29{4}, Index29{3}});

        StatusOr<std::vector<Index29>> recovered =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt, interrupt.value());
        REQUIRE(recovered.ok());
        REQUIRE(recovered.value() == plain);

        // Empty skips would desync: key advances on former interrupt positions.
        StatusOr<std::vector<Index29>> desynced =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt);
        REQUIRE(desynced.ok());
        REQUIRE(desynced.value() != plain);
    }
}

TEST_CASE("BeaufortKeyTransform involution, interrupts, Atbash identity", "[transform]") {
    const BeaufortKeyTransform transform;
    REQUIRE(transform.id() == TransformId::beaufort_key());

    SECTION("params reject empty key") {
        const std::vector<Index29> plain{Index29{0}};
        REQUIRE_FALSE(transform
                          .apply(plain, nlohmann::json{{"key_indices", nlohmann::json::array()}},
                                 TransformDirection::Decrypt)
                          .ok());
    }

    SECTION("synth-beaufort-involution") {
        const nlohmann::json params{{"key_indices", {5, 7, 11}}};
        const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{10}, Index29{28},
                                         Index29{14}};

        StatusOr<std::vector<Index29>> once =
            transform.apply(plain, params, TransformDirection::Decrypt);
        REQUIRE(once.ok());
        // (5-0), (7-1), (11-10), (5-28), (7-14) → 5, 6, 1, 6, 22
        REQUIRE(once.value() ==
                std::vector<Index29>{Index29{5}, Index29{6}, Index29{1}, Index29{6}, Index29{22}});

        StatusOr<std::vector<Index29>> twice =
            transform.apply(once.value(), params, TransformDirection::Encrypt);
        REQUIRE(twice.ok());
        REQUIRE(twice.value() == plain);
    }

    SECTION("synth-beaufort-with-interrupts") {
        const nlohmann::json params{{"key_indices", {1, 2}}};
        const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
        StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1, 3});
        REQUIRE(interrupt.ok());

        StatusOr<std::vector<Index29>> once =
            transform.apply(plain, params, TransformDirection::Decrypt, interrupt.value());
        REQUIRE(once.ok());
        // i=0: 1-0=1; i=1 skip→1; i=2: 2-2=0; i=3 skip→3
        REQUIRE(once.value() ==
                std::vector<Index29>{Index29{1}, Index29{1}, Index29{0}, Index29{3}});

        StatusOr<std::vector<Index29>> twice =
            transform.apply(once.value(), params, TransformDirection::Decrypt, interrupt.value());
        REQUIRE(twice.ok());
        REQUIRE(twice.value() == plain);
    }

    SECTION("Atbash equals Beaufort with constant key 28") {
        // Wiki pitfall: Vigenère is (c-k); Beaufort is (k-c). Constant k=28 ⇒ 28-c.
        const AtbashTransform atbash;
        const nlohmann::json beaufort_params{{"key_indices", {28}}};
        const std::vector<Index29> input{Index29{0}, Index29{1}, Index29{14}, Index29{27},
                                         Index29{28}};

        StatusOr<std::vector<Index29>> via_atbash =
            atbash.apply(input, nlohmann::json::object(), TransformDirection::Decrypt);
        StatusOr<std::vector<Index29>> via_beaufort =
            transform.apply(input, beaufort_params, TransformDirection::Decrypt);
        REQUIRE(via_atbash.ok());
        REQUIRE(via_beaufort.ok());
        REQUIRE(via_atbash.value() == via_beaufort.value());
        REQUIRE(via_atbash.value() == std::vector<Index29>{Index29{28}, Index29{27}, Index29{14},
                                                           Index29{1}, Index29{0}});
    }
}

TEST_CASE("TotientPrimeStreamTransform with InterruptPolicy", "[transform]") {
    const TotientPrimeStreamTransform transform;
    REQUIRE(transform.id() == TransformId::totient_prime_stream());

    const nlohmann::json params{
        {"prime_start_index", 0},
        {"shift_mode", "prime_minus_one_mod_29"},
    };

    SECTION("params") {
        const std::vector<Index29> plain{Index29{0}};
        REQUIRE(transform.apply(plain, nlohmann::json::object(), TransformDirection::Decrypt).ok());
        REQUIRE_FALSE(
            transform
                .apply(plain, nlohmann::json{{"shift_mode", "wrong"}}, TransformDirection::Decrypt)
                .ok());
        REQUIRE_FALSE(transform
                          .apply(plain, nlohmann::json{{"prime_start_index", -1}},
                                 TransformDirection::Decrypt)
                          .ok());
    }

    SECTION("synth-totient-round-trip-no-interrupt") {
        // First shifts: 1,2,4,6 — encrypt adds, decrypt subtracts.
        const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
        StatusOr<std::vector<Index29>> cipher =
            transform.apply(plain, params, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());
        REQUIRE(cipher.value() ==
                std::vector<Index29>{Index29{1}, Index29{3}, Index29{6}, Index29{9}});

        StatusOr<std::vector<Index29>> recovered =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt);
        REQUIRE(recovered.ok());
        REQUIRE(recovered.value() == plain);
    }

    SECTION("synth-totient-with-interrupts") {
        const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
        StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_skip_indices({1});
        REQUIRE(interrupt.ok());

        StatusOr<std::vector<Index29>> cipher =
            transform.apply(plain, params, TransformDirection::Encrypt, interrupt.value());
        REQUIRE(cipher.ok());
        // i=0: +1 → 1; i=1 skip → 1; i=2: +2 → 4; i=3: +4 → 7
        REQUIRE(cipher.value() ==
                std::vector<Index29>{Index29{1}, Index29{1}, Index29{4}, Index29{7}});

        StatusOr<std::vector<Index29>> recovered =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt, interrupt.value());
        REQUIRE(recovered.ok());
        REQUIRE(recovered.value() == plain);

        StatusOr<std::vector<Index29>> desynced =
            transform.apply(cipher.value(), params, TransformDirection::Decrypt);
        REQUIRE(desynced.ok());
        REQUIRE(desynced.value() != plain);
    }

    SECTION("prime_start_index offsets the stream") {
        const std::vector<Index29> plain{Index29{10}};
        const nlohmann::json start_at_three{{"prime_start_index", 3}}; // p3=7 → shift 6
        StatusOr<std::vector<Index29>> cipher =
            transform.apply(plain, start_at_three, TransformDirection::Encrypt);
        REQUIRE(cipher.ok());
        REQUIRE(cipher.value() == std::vector<Index29>{Index29{16}});
    }
}

TEST_CASE("apply_into / kernel are span-based and match apply", "[transform][span]") {
    const std::vector<Index29> input{Index29{0}, Index29{1}, Index29{2}, Index29{28}};

    SECTION("atbash kernel in-place") {
        std::vector<Index29> buf = input;
        REQUIRE(AtbashTransform::kernel(buf, buf).ok());
        REQUIRE(buf == std::vector<Index29>{Index29{28}, Index29{27}, Index29{26}, Index29{0}});
        REQUIRE(AtbashTransform::kernel(buf, buf).ok());
        REQUIRE(buf == input);
    }

    SECTION("caesar apply_into matches apply") {
        CaesarTransform transform;
        const nlohmann::json params{{"shift", 3}};
        StatusOr<std::vector<Index29>> via_apply =
            transform.apply(input, params, TransformDirection::Encrypt);

        std::vector<Index29> via_into(input.size());
        REQUIRE(transform.apply_into(input, via_into, params, TransformDirection::Encrypt).ok());
        REQUIRE(via_apply.ok());
        REQUIRE(via_into == via_apply.value());
    }

    SECTION("length mismatch is an error") {
        std::vector<Index29> short_out(2);
        REQUIRE_FALSE(AtbashTransform::kernel(input, short_out).ok());
    }

    SECTION("vigenere kernel with skips") {
        const std::vector<Index29> key{Index29{1}, Index29{2}};
        const std::vector<std::size_t> skips{1};
        std::vector<Index29> out(input.size());
        REQUIRE(
            VigenereKeyTransform::kernel(input, out, key, skips, TransformDirection::Encrypt).ok());
        // i0: +1 → 1; i1 skip → 1; i2: +2 → 4; i3: +1 → 0
        REQUIRE(out == std::vector<Index29>{Index29{1}, Index29{1}, Index29{4}, Index29{0}});
    }

    SECTION("compose apply_into ping-pong matches apply") {
        ComposeTransform compose;
        const nlohmann::json params = ComposeTransform::atbash_then_caesar_params(3);
        StatusOr<std::vector<Index29>> via_apply =
            compose.apply(input, params, TransformDirection::Decrypt);
        std::vector<Index29> via_into(input.size());
        REQUIRE(compose.apply_into(input, via_into, params, TransformDirection::Decrypt).ok());
        REQUIRE(via_apply.ok());
        REQUIRE(via_into == via_apply.value());
    }
}
