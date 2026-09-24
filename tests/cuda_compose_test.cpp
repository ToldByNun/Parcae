#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "compose_driver.hpp"
#include "interrupt_device_view.hpp"
#include "params.hpp"
#include "params_json.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

namespace {

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

[[nodiscard]] std::vector<Index29> from_bytes(const std::vector<std::uint8_t>& bytes) {
    std::vector<Index29> out;
    out.reserve(bytes.size());
    for (std::uint8_t value : bytes) {
        out.push_back(Index29{value});
    }
    return out;
}

} // namespace

TEST_CASE("CUDA compose driver atbash_then_caesar matches CPU", "[cuda][parity][compose]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{10}, Index29{28}, Index29{14}};
    const nlohmann::json params = ComposeTransform::atbash_then_caesar_params(3);

    StatusOr<ComposeParamsHost> recipe = CudaParamsJson::compose_from_json(params);
    REQUIRE(recipe.ok());
    REQUIRE(recipe.value().stages.size() == 2);

    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), plain.size());
    REQUIRE(view.ok());

    const std::vector<std::uint8_t> host_in = to_bytes(plain);

    SECTION("decrypt") {
        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(ComposeTransform{}
                    .apply_into(plain, cpu_out, params, TransformDirection::Decrypt)
                    .ok());

        std::vector<std::uint8_t> host_out(host_in.size(), 0xFFu);
        REQUIRE(ComposeDriver::apply_host(host_in, host_out, recipe.value(), view.value(),
                                          CudaDir::Decrypt)
                    .ok());
        REQUIRE(from_bytes(host_out) == cpu_out);
    }

    SECTION("encrypt") {
        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(ComposeTransform{}
                    .apply_into(plain, cpu_out, params, TransformDirection::Encrypt)
                    .ok());

        std::vector<std::uint8_t> host_out(host_in.size(), 0xFFu);
        REQUIRE(ComposeDriver::apply_host(host_in, host_out, recipe.value(), view.value(),
                                          CudaDir::Encrypt)
                    .ok());
        REQUIRE(from_bytes(host_out) == cpu_out);
    }

    SECTION("round-trip") {
        std::vector<std::uint8_t> mid(host_in.size());
        REQUIRE(
            ComposeDriver::apply_host(host_in, mid, recipe.value(), view.value(), CudaDir::Encrypt)
                .ok());
        std::vector<std::uint8_t> back(host_in.size());
        REQUIRE(ComposeDriver::apply_host(mid, back, recipe.value(), view.value(), CudaDir::Decrypt)
                    .ok());
        REQUIRE(back == host_in);
    }
}

TEST_CASE("CUDA compose driver single identity stage", "[cuda][parity][compose]") {
    REQUIRE(ParcaeCuda::available());

    const nlohmann::json params{
        {"stages", nlohmann::json::array({nlohmann::json{
                       {"transform_id", "identity"},
                       {"params", nlohmann::json::object()},
                   }})},
    };
    StatusOr<ComposeParamsHost> recipe = CudaParamsJson::compose_from_json(params);
    REQUIRE(recipe.ok());

    const std::vector<std::uint8_t> host_in{3, 1, 4, 1, 5};
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), host_in.size());
    REQUIRE(view.ok());

    std::vector<std::uint8_t> host_out(host_in.size(), 0);
    REQUIRE(
        ComposeDriver::apply_host(host_in, host_out, recipe.value(), view.value(), CudaDir::Decrypt)
            .ok());
    REQUIRE(host_out == host_in);
}

TEST_CASE("CUDA compose driver rejects empty stages and size mismatch", "[cuda][parity][compose]") {
    REQUIRE(ParcaeCuda::available());

    ComposeParamsHost empty;
    std::vector<std::uint8_t> in{1};
    std::vector<std::uint8_t> out{0};
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), 1);
    REQUIRE(view.ok());
    REQUIRE_FALSE(ComposeDriver::apply_host(in, out, empty, view.value(), CudaDir::Decrypt).ok());

    StatusOr<ComposeParamsHost> recipe =
        CudaParamsJson::compose_from_json(ComposeTransform::atbash_then_caesar_params(1));
    REQUIRE(recipe.ok());
    std::vector<std::uint8_t> short_out(0);
    REQUIRE_FALSE(
        ComposeDriver::apply_host(in, short_out, recipe.value(), view.value(), CudaDir::Decrypt)
            .ok());
}

#else

TEST_CASE("CUDA compose skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][compose]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise ComposeDriver");
}

#endif
