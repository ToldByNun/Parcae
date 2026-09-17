#include <parcae/core/index29.hpp>
#include <parcae/interrupt/policy.hpp>
#include <parcae/transform/affine_transform.hpp>
#include <parcae/transform/atbash_transform.hpp>
#include <parcae/transform/beaufort_key_transform.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/identity_transform.hpp>
#include <parcae/transform/totient_prime_stream_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <parcae/transform/vigenere_key_transform.hpp>

#include <catch2/catch_test_macros.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

[[nodiscard]] std::vector<Index29> random_indices(std::mt19937& rng, std::size_t length) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(Index29::modulus) - 1);
    std::vector<Index29> out;
    out.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        out.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }
    return out;
}

[[nodiscard]] InterruptPolicy random_skips(std::mt19937& rng, std::size_t length) {
    if (length == 0) {
        return InterruptPolicy::none();
    }
    std::bernoulli_distribution coin(0.15);
    std::vector<std::size_t> skips;
    for (std::size_t i = 0; i < length; ++i) {
        if (coin(rng)) {
            skips.push_back(i);
        }
    }
    StatusOr<InterruptPolicy> policy = InterruptPolicy::from_skip_indices(std::move(skips));
    REQUIRE(policy.ok());
    return policy.value();
}

[[nodiscard]] nlohmann::json random_key_params(std::mt19937& rng, std::size_t key_len) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(Index29::modulus) - 1);
    nlohmann::json key = nlohmann::json::array();
    for (std::size_t i = 0; i < key_len; ++i) {
        key.push_back(dist(rng));
    }
    return nlohmann::json{{"key_indices", std::move(key)}};
}

template <typename TransformT>
void require_encrypt_decrypt_round_trip(
    const TransformT& transform,
    const std::vector<Index29>& plain,
    const nlohmann::json& params,
    const InterruptPolicy& interrupt) {
    StatusOr<std::vector<Index29>> cipher =
        transform.apply(plain, params, TransformDirection::Encrypt, interrupt);
    REQUIRE(cipher.ok());
    StatusOr<std::vector<Index29>> recovered =
        transform.apply(cipher.value(), params, TransformDirection::Decrypt, interrupt);
    REQUIRE(recovered.ok());
    REQUIRE(recovered.value() == plain);
}

}  // namespace

TEST_CASE("Property: invertible families round-trip random Index29 streams", "[transform][property]") {
    std::mt19937 rng(0xC1CADAu);
    constexpr int trials = 48;

    for (int trial = 0; trial < trials; ++trial) {
        std::uniform_int_distribution<std::size_t> len_dist(1, 64);
        const std::size_t length = len_dist(rng);
        const std::vector<Index29> plain = random_indices(rng, length);
        const InterruptPolicy interrupt = random_skips(rng, length);

        require_encrypt_decrypt_round_trip(
            IdentityTransform{}, plain, nlohmann::json::object(), interrupt);

        {
            AtbashTransform transform;
            StatusOr<std::vector<Index29>> once =
                transform.apply(plain, nlohmann::json::object(), TransformDirection::Decrypt);
            REQUIRE(once.ok());
            StatusOr<std::vector<Index29>> twice = transform.apply(
                once.value(), nlohmann::json::object(), TransformDirection::Decrypt);
            REQUIRE(twice.ok());
            REQUIRE(twice.value() == plain);
        }

        {
            std::uniform_int_distribution<int> shift_dist(0, 28);
            const nlohmann::json params{{"shift", shift_dist(rng)}};
            require_encrypt_decrypt_round_trip(CaesarTransform{}, plain, params, interrupt);
        }

        {
            std::uniform_int_distribution<int> a_dist(1, 28);
            std::uniform_int_distribution<int> b_dist(0, 28);
            const nlohmann::json params{{"a", a_dist(rng)}, {"b", b_dist(rng)}};
            require_encrypt_decrypt_round_trip(AffineTransform{}, plain, params, interrupt);
        }

        {
            std::uniform_int_distribution<std::size_t> key_len_dist(1, 12);
            const nlohmann::json params = random_key_params(rng, key_len_dist(rng));
            require_encrypt_decrypt_round_trip(
                VigenereKeyTransform{}, plain, params, interrupt);
        }

        {
            std::uniform_int_distribution<std::size_t> key_len_dist(1, 12);
            const nlohmann::json params = random_key_params(rng, key_len_dist(rng));
            BeaufortKeyTransform transform;
            StatusOr<std::vector<Index29>> once =
                transform.apply(plain, params, TransformDirection::Encrypt, interrupt);
            REQUIRE(once.ok());
            StatusOr<std::vector<Index29>> twice =
                transform.apply(once.value(), params, TransformDirection::Encrypt, interrupt);
            REQUIRE(twice.ok());
            REQUIRE(twice.value() == plain);
        }

        {
            std::uniform_int_distribution<int> start_dist(0, 20);
            const nlohmann::json params{{"prime_start_index", start_dist(rng)}};
            require_encrypt_decrypt_round_trip(
                TotientPrimeStreamTransform{}, plain, params, interrupt);
        }

        {
            std::uniform_int_distribution<int> shift_dist(0, 28);
            const std::uint8_t shift = static_cast<std::uint8_t>(shift_dist(rng));
            const nlohmann::json params = ComposeTransform::atbash_then_caesar_params(shift);
            require_encrypt_decrypt_round_trip(
                ComposeTransform{}, plain, params, InterruptPolicy::none());
        }
    }
}
