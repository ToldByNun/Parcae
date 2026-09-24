#include "parcae/transform/compose_transform.hpp"

#include "params.hpp"
#include "params_json.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

TEST_CASE("CUDA POD params caesar / affine from JSON", "[cuda][params]") {
    SECTION("caesar") {
        StatusOr<CaesarParams> p = CudaParamsJson::caesar_from_json(nlohmann::json{{"shift", 3}});
        REQUIRE(p.ok());
        REQUIRE(p.value().shift == 3);
        REQUIRE(CudaParamsJson::caesar_to_json(p.value()) == nlohmann::json{{"shift", 3}});
        REQUIRE_FALSE(CudaParamsJson::caesar_from_json(nlohmann::json{{"shift", 29}}).ok());
    }

    SECTION("affine") {
        StatusOr<AffineParams> p =
            CudaParamsJson::affine_from_json(nlohmann::json{{"a", 2}, {"b", 5}});
        REQUIRE(p.ok());
        REQUIRE(p.value().a == 2);
        REQUIRE(p.value().b == 5);
        REQUIRE(CudaParamsJson::affine_to_json(p.value()) == nlohmann::json{{"a", 2}, {"b", 5}});
        REQUIRE_FALSE(CudaParamsJson::affine_from_json(nlohmann::json{{"a", 0}, {"b", 1}}).ok());
    }
}

TEST_CASE("CUDA POD params vigenere key list", "[cuda][params]") {
    const nlohmann::json json{
        {"key_indices", {23, 10, 1, 10, 9, 10, 16, 26}},
        {"key_latin", "DIVINITY"},
    };
    StatusOr<KeyParamsHost> key = CudaParamsJson::key_from_json(json);
    REQUIRE(key.ok());
    REQUIRE(key.value().key.size() == 8);
    REQUIRE(key.value().view.key_len == 8);
    REQUIRE(key.value().view.key_ptr == key.value().key.data());
    REQUIRE(key.value().key[0] == 23);
    REQUIRE(key.value().key[7] == 26);

    const nlohmann::json back = CudaParamsJson::key_to_json(key.value());
    REQUIRE(back.at("key_indices") == json.at("key_indices"));
}

TEST_CASE("CUDA POD params totient start index", "[cuda][params]") {
    StatusOr<TotientParams> p =
        CudaParamsJson::totient_from_json(nlohmann::json{{"prime_start_index", 3}});
    REQUIRE(p.ok());
    REQUIRE(p.value().prime_start_index == 3);
}

TEST_CASE("CUDA POD params compose atbash_then_caesar", "[cuda][params]") {
    const nlohmann::json json = ComposeTransform::atbash_then_caesar_params(3);
    StatusOr<ComposeParamsHost> compose = CudaParamsJson::compose_from_json(json);
    REQUIRE(compose.ok());
    REQUIRE(compose.value().stages.size() == 2);
    REQUIRE(compose.value().stages[0].family == CudaFamilyId::Atbash);
    REQUIRE(compose.value().stages[0].direction == CudaDir::Decrypt);
    REQUIRE(compose.value().stages[1].family == CudaFamilyId::Caesar);
    REQUIRE(compose.value().stages[1].direction == CudaDir::Encrypt);
    REQUIRE(compose.value().stages[1].caesar.shift == 3);
    REQUIRE(compose.value().key_arena.empty());
}

TEST_CASE("CUDA Dir matches TransformDirection", "[cuda][params]") {
    REQUIRE(CudaDirUtil::from_transform_direction(TransformDirection::Decrypt) == CudaDir::Decrypt);
    REQUIRE(CudaDirUtil::from_transform_direction(TransformDirection::Encrypt) == CudaDir::Encrypt);
    REQUIRE(CudaDirUtil::to_transform_direction(CudaDir::Encrypt) == TransformDirection::Encrypt);
}

TEST_CASE("CUDA FamilyId from TransformId", "[cuda][params]") {
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::caesar()).value() ==
            CudaFamilyId::Caesar);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::vigenere_key()).value() ==
            CudaFamilyId::VigenereKey);
}
