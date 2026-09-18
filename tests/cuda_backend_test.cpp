#include "backend.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

[[nodiscard]] bool message_contains(const Status& status, const char* fragment) {
    return status.message().find(fragment) != std::string::npos;
}

}  // namespace

TEST_CASE("CudaBackend apply_into size mismatch", "[cuda][backend]") {
    std::vector<Index29> in{Index29{1}, Index29{2}};
    std::vector<Index29> out(1);
    Status status = CudaBackend::apply_into(
        TransformId::identity(),
        in,
        out,
        nlohmann::json::object(),
        TransformDirection::Decrypt);
    REQUIRE_FALSE(status.ok());
    REQUIRE(message_contains(status, "size mismatch"));
}

TEST_CASE("CudaBackend rejects bad caesar params before dispatch", "[cuda][backend]") {
    std::vector<Index29> in{Index29{3}};
    std::vector<Index29> out(1);
    Status status = CudaBackend::apply_into(
        TransformId::caesar(),
        in,
        out,
        nlohmann::json{{"shift", 29}},
        TransformDirection::Decrypt);
    REQUIRE_FALSE(status.ok());
    REQUIRE_FALSE(message_contains(status, "not implemented"));
    REQUIRE_FALSE(message_contains(status, "not available"));
}

TEST_CASE("CudaBackend rejects out-of-range interrupt", "[cuda][backend]") {
    StatusOr<InterruptPolicy> policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{5});
    REQUIRE(policy.ok());

    std::vector<Index29> in{Index29{0}, Index29{1}, Index29{2}};
    std::vector<Index29> out(3);
    Status status = CudaBackend::apply_into(
        TransformId::identity(),
        in,
        out,
        nlohmann::json::object(),
        TransformDirection::Decrypt,
        policy.value());
    REQUIRE_FALSE(status.ok());
    REQUIRE(message_contains(status, "skip index"));
}

TEST_CASE("CudaBackend catalog ids reach availability or stub gate", "[cuda][backend]") {
    const std::vector<Index29> in{Index29{1}, Index29{2}, Index29{3}};
    std::vector<Index29> out(in.size());

    const auto expect_gate = [&](const TransformId& id, const nlohmann::json& params) {
        Status status = CudaBackend::apply_into(
            id, in, out, params, TransformDirection::Decrypt);
        REQUIRE_FALSE(status.ok());
        if (CudaBackend::available()) {
            REQUIRE(message_contains(status, "not implemented"));
        } else {
            REQUIRE(message_contains(status, "not available"));
        }
    };

    expect_gate(TransformId::identity(), nlohmann::json::object());
    expect_gate(TransformId::atbash(), nlohmann::json::object());
    expect_gate(TransformId::caesar(), nlohmann::json{{"shift", 3}});
    expect_gate(TransformId::affine(), nlohmann::json{{"a", 2}, {"b", 5}});
    expect_gate(
        TransformId::vigenere_key(),
        nlohmann::json{{"key_indices", {1, 2, 3}}});
    expect_gate(
        TransformId::beaufort_key(),
        nlohmann::json{{"key_indices", {4, 5}}});
    expect_gate(TransformId::totient_prime_stream(), nlohmann::json::object());
    expect_gate(TransformId::compose(), ComposeTransform::atbash_then_caesar_params(3));
}

TEST_CASE("CudaBackend apply mirrors apply_into Status", "[cuda][backend]") {
    const std::vector<Index29> in{Index29{7}};
    StatusOr<std::vector<Index29>> result = CudaBackend::apply(
        TransformId::caesar(),
        in,
        nlohmann::json{{"shift", 1}},
        TransformDirection::Encrypt);
    REQUIRE_FALSE(result.ok());
    if (CudaBackend::available()) {
        REQUIRE(message_contains(result.status(), "not implemented"));
        REQUIRE(message_contains(result.status(), "caesar"));
    } else {
        REQUIRE(message_contains(result.status(), "not available"));
    }
}
