#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/apply_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/identity_transform.hpp"
#include "parcae/transform/totient_prime_stream_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include "backend.hpp"
#include "parcae_cuda.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#if !defined(PARCAE_HAS_CUDA)

TEST_CASE("CUDA property round-trips (skipped without CUDA)", "[cuda][property]") {
    SUCCEED("PARCAE_HAS_CUDA unset — device property suite not linked");
}

#else

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

void require_cuda_matches_cpu(const TransformId& id, const std::vector<Index29>& input,
                              const nlohmann::json& params, TransformDirection direction,
                              const InterruptPolicy& interrupt) {
    StatusOr<std::vector<Index29>> cpu =
        ApplyTransform::apply(id, input, params, direction, interrupt);
    REQUIRE(cpu.ok());

    std::vector<Index29> cuda_out(input.size());
    REQUIRE(CudaBackend::apply_into(id, input, cuda_out, params, direction, interrupt).ok());
    REQUIRE(cuda_out == cpu.value());
}

void require_cuda_encrypt_decrypt_round_trip(const TransformId& id,
                                             const std::vector<Index29>& plain,
                                             const nlohmann::json& params,
                                             const InterruptPolicy& interrupt) {
    require_cuda_matches_cpu(id, plain, params, TransformDirection::Encrypt, interrupt);

    std::vector<Index29> cipher(plain.size());
    REQUIRE(
        CudaBackend::apply_into(id, plain, cipher, params, TransformDirection::Encrypt, interrupt)
            .ok());

    require_cuda_matches_cpu(id, cipher, params, TransformDirection::Decrypt, interrupt);

    std::vector<Index29> recovered(plain.size());
    REQUIRE(CudaBackend::apply_into(id, cipher, recovered, params, TransformDirection::Decrypt,
                                    interrupt)
                .ok());
    REQUIRE(recovered == plain);
}

void require_cuda_involution_twice(const TransformId& id, const std::vector<Index29>& plain,
                                   const nlohmann::json& params, const InterruptPolicy& interrupt,
                                   TransformDirection direction) {
    require_cuda_matches_cpu(id, plain, params, direction, interrupt);

    std::vector<Index29> once(plain.size());
    REQUIRE(CudaBackend::apply_into(id, plain, once, params, direction, interrupt).ok());

    require_cuda_matches_cpu(id, once, params, direction, interrupt);

    std::vector<Index29> twice(plain.size());
    REQUIRE(CudaBackend::apply_into(id, once, twice, params, direction, interrupt).ok());
    REQUIRE(twice == plain);
}

} // namespace

TEST_CASE("Property: CUDA invertible families round-trip random Index29 streams",
          "[cuda][property]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(CudaBackend::available());

    std::mt19937 rng(0xC0DA40u);
    constexpr int trials = 48;

    for (int trial = 0; trial < trials; ++trial) {
        INFO("trial=" << trial);
        std::uniform_int_distribution<std::size_t> len_dist(1, 64);
        const std::size_t length = len_dist(rng);
        const std::vector<Index29> plain = random_indices(rng, length);
        const InterruptPolicy interrupt = random_skips(rng, length);

        require_cuda_encrypt_decrypt_round_trip(TransformId::identity(), plain,
                                                nlohmann::json::object(), interrupt);

        require_cuda_involution_twice(TransformId::atbash(), plain, nlohmann::json::object(),
                                      InterruptPolicy::none(), TransformDirection::Decrypt);

        {
            std::uniform_int_distribution<int> shift_dist(0, 28);
            const nlohmann::json params{{"shift", shift_dist(rng)}};
            require_cuda_encrypt_decrypt_round_trip(TransformId::caesar(), plain, params,
                                                    interrupt);
        }

        {
            std::uniform_int_distribution<int> a_dist(1, 28);
            std::uniform_int_distribution<int> b_dist(0, 28);
            const nlohmann::json params{{"a", a_dist(rng)}, {"b", b_dist(rng)}};
            require_cuda_encrypt_decrypt_round_trip(TransformId::affine(), plain, params,
                                                    interrupt);
        }

        {
            std::uniform_int_distribution<std::size_t> key_len_dist(1, 12);
            const nlohmann::json params = random_key_params(rng, key_len_dist(rng));
            require_cuda_encrypt_decrypt_round_trip(TransformId::vigenere_key(), plain, params,
                                                    interrupt);
        }

        {
            std::uniform_int_distribution<std::size_t> key_len_dist(1, 12);
            const nlohmann::json params = random_key_params(rng, key_len_dist(rng));
            require_cuda_involution_twice(TransformId::beaufort_key(), plain, params, interrupt,
                                          TransformDirection::Encrypt);
        }

        {
            std::uniform_int_distribution<int> start_dist(0, 20);
            const nlohmann::json params{{"prime_start_index", start_dist(rng)}};
            require_cuda_encrypt_decrypt_round_trip(TransformId::totient_prime_stream(), plain,
                                                    params, interrupt);
        }

        {
            std::uniform_int_distribution<int> shift_dist(0, 28);
            const std::uint8_t shift = static_cast<std::uint8_t>(shift_dist(rng));
            const nlohmann::json params = ComposeTransform::atbash_then_caesar_params(shift);
            // Compose interrupt support matches CPU property suite (none).
            require_cuda_encrypt_decrypt_round_trip(TransformId::compose(), plain, params,
                                                    InterruptPolicy::none());
        }
    }
}

TEST_CASE("Property: CUDA long-stream caesar/affine/vigenere round-trips", "[cuda][property]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xC0DA41u);
    constexpr std::size_t length = 4096;
    const std::vector<Index29> plain = random_indices(rng, length);
    const InterruptPolicy interrupt = random_skips(rng, length);

    {
        const nlohmann::json params{{"shift", 17}};
        require_cuda_encrypt_decrypt_round_trip(TransformId::caesar(), plain, params, interrupt);
    }
    {
        const nlohmann::json params{{"a", 11}, {"b", 5}};
        require_cuda_encrypt_decrypt_round_trip(TransformId::affine(), plain, params, interrupt);
    }
    {
        const nlohmann::json params = random_key_params(rng, 8);
        require_cuda_encrypt_decrypt_round_trip(TransformId::vigenere_key(), plain, params,
                                                interrupt);
    }
}

#endif // PARCAE_HAS_CUDA
