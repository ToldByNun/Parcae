#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/transform/identity_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include "backend.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <random>
#include <vector>

TEST_CASE("CUDA identity via CudaBackend matches CPU", "[cuda][parity][identity]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(CudaBackend::available());

    const std::vector<Index29> input{
        Index29{0}, Index29{1}, Index29{14}, Index29{27}, Index29{28},
    };

    std::vector<Index29> cpu_out(input.size());
    REQUIRE(IdentityTransform{}
                .apply_into(input, cpu_out, nlohmann::json::object(), TransformDirection::Decrypt)
                .ok());

    std::vector<Index29> cuda_out(input.size());
    REQUIRE(CudaBackend::apply_into(TransformId::identity(), input, cuda_out,
                                    nlohmann::json::object(), TransformDirection::Encrypt)
                .ok());
    REQUIRE(cuda_out == cpu_out);
    REQUIRE(cuda_out == input);
}

TEST_CASE("CUDA identity CudaBackend random and empty", "[cuda][parity][identity]") {
    REQUIRE(CudaBackend::available());

    std::mt19937 rng(0x1DE771u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> input;
    input.reserve(128);
    for (std::size_t i = 0; i < 128; ++i) {
        input.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }

    StatusOr<std::vector<Index29>> cuda_out = CudaBackend::apply(
        TransformId::identity(), input, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(cuda_out.ok());
    REQUIRE(cuda_out.value() == input);

    std::vector<Index29> empty_in;
    std::vector<Index29> empty_out;
    REQUIRE(CudaBackend::apply_into(TransformId::identity(), empty_in, empty_out,
                                    nlohmann::json::object(), TransformDirection::Decrypt)
                .ok());
}

TEST_CASE("CUDA identity rejects non-empty params", "[cuda][parity][identity]") {
    REQUIRE(CudaBackend::available());
    std::vector<Index29> in{Index29{1}};
    std::vector<Index29> out(1);
    REQUIRE_FALSE(CudaBackend::apply_into(TransformId::identity(), in, out,
                                          nlohmann::json{{"extra", 1}}, TransformDirection::Decrypt)
                      .ok());
}

#else

TEST_CASE("CUDA identity skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][identity]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise identity via CudaBackend");
}

#endif
