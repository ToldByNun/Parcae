#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/parity/parity_record.hpp"
#include "parcae/transform/apply_transform.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include "backend.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

[[nodiscard]] bool message_contains(const Status& status, const char* fragment) {
    return status.message().find(fragment) != std::string::npos;
}

[[nodiscard]] bool digests_match_except_backend(const ParityRecord& cpu, const ParityRecord& cuda) {
    return cpu.transform_id() == cuda.transform_id() &&
           cpu.params_hash_sha256() == cuda.params_hash_sha256() &&
           cpu.input_sha256() == cuda.input_sha256() &&
           cpu.output_sha256() == cuda.output_sha256() &&
           cpu.interrupt_sha256() == cuda.interrupt_sha256() && cpu.backend() == "cpu" &&
           cuda.backend() == "cuda";
}

} // namespace

TEST_CASE("CudaBackend apply_into size mismatch", "[cuda][backend]") {
    std::vector<Index29> in{Index29{1}, Index29{2}};
    std::vector<Index29> out(1);
    Status status = CudaBackend::apply_into(TransformId::identity(), in, out,
                                            nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE_FALSE(status.ok());
    REQUIRE(message_contains(status, "size mismatch"));
}

TEST_CASE("CudaBackend rejects bad caesar params before dispatch", "[cuda][backend]") {
    std::vector<Index29> in{Index29{3}};
    std::vector<Index29> out(1);
    Status status = CudaBackend::apply_into(
        TransformId::caesar(), in, out, nlohmann::json{{"shift", 29}}, TransformDirection::Decrypt);
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
    Status status =
        CudaBackend::apply_into(TransformId::identity(), in, out, nlohmann::json::object(),
                                TransformDirection::Decrypt, policy.value());
    REQUIRE_FALSE(status.ok());
    REQUIRE(message_contains(status, "skip index"));
}

TEST_CASE("CudaBackend full catalog matches CPU or reports unavailable", "[cuda][backend]") {
    const std::vector<Index29> in{Index29{0}, Index29{1}, Index29{2}, Index29{3}};

    const auto check = [&](const TransformId& id, const nlohmann::json& params,
                           TransformDirection direction,
                           const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        StatusOr<std::vector<Index29>> cpu =
            ApplyTransform::apply(id, in, params, direction, interrupt);
        REQUIRE(cpu.ok());

        StatusOr<std::vector<Index29>> cuda =
            CudaBackend::apply(id, in, params, direction, interrupt);
        if (!CudaBackend::available()) {
            REQUIRE_FALSE(cuda.ok());
            REQUIRE(message_contains(cuda.status(), "not available"));
            return;
        }
        REQUIRE(cuda.ok());
        REQUIRE(cuda.value() == cpu.value());
    };

    check(TransformId::identity(), nlohmann::json::object(), TransformDirection::Decrypt);
    check(TransformId::atbash(), nlohmann::json::object(), TransformDirection::Decrypt);
    check(TransformId::caesar(), nlohmann::json{{"shift", 3}}, TransformDirection::Encrypt);
    check(TransformId::affine(), nlohmann::json{{"a", 2}, {"b", 5}}, TransformDirection::Decrypt);
    check(TransformId::vigenere_key(), nlohmann::json{{"key_indices", {1, 2}}},
          TransformDirection::Encrypt);
    check(TransformId::beaufort_key(), nlohmann::json{{"key_indices", {5, 7}}},
          TransformDirection::Decrypt);

    StatusOr<InterruptPolicy> skips =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{1});
    REQUIRE(skips.ok());
    check(TransformId::totient_prime_stream(), nlohmann::json{{"prime_start_index", 0}},
          TransformDirection::Encrypt, skips.value());
    check(TransformId::compose(), ComposeTransform::atbash_then_caesar_params(3),
          TransformDirection::Decrypt);
}

TEST_CASE("CudaBackend apply_and_capture matches CPU digests", "[cuda][backend][parity]") {
    const std::vector<Index29> in{Index29{4}, Index29{5}, Index29{6}};
    const nlohmann::json params{{"shift", 2}};

    StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cpu = ParityRecord::apply_and_capture(
        TransformId::caesar(), in, params, TransformDirection::Decrypt);
    REQUIRE(cpu.ok());

    StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cuda = CudaBackend::apply_and_capture(
        TransformId::caesar(), in, params, TransformDirection::Decrypt);

    if (!CudaBackend::available()) {
        REQUIRE_FALSE(cuda.ok());
        REQUIRE(message_contains(cuda.status(), "not available"));
        return;
    }

    REQUIRE(cuda.ok());
    REQUIRE(cuda.value().first == cpu.value().first);
    REQUIRE(digests_match_except_backend(cpu.value().second, cuda.value().second));
}
