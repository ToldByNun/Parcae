#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"

#include "beaufort_key_kernel.hpp"
#include "interrupt_device_view.hpp"
#include "parcae_cuda.hpp"

#include <cstdint>
#include <random>
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

TEST_CASE("CUDA beaufort parity involution and interrupts", "[cuda][parity][beaufort]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> key{5, 7, 11};
    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{10}, Index29{28}, Index29{14}};
    const std::vector<std::uint8_t> host_in = to_bytes(plain);

    SECTION("involution no interrupts") {
        StatusOr<InterruptDeviceView> view =
            InterruptDeviceView::from_policy(InterruptPolicy::none(), host_in.size());
        REQUIRE(view.ok());

        std::vector<Index29> cpu_once(plain.size());
        REQUIRE(BeaufortKeyTransform::kernel(plain, cpu_once, from_bytes(key), {}).ok());

        std::vector<std::uint8_t> once(host_in.size(), 0xFFu);
        REQUIRE(BeaufortKeyKernel::apply_host(host_in, once, key, view.value()).ok());
        REQUIRE(from_bytes(once) == cpu_once);
        REQUIRE(once == std::vector<std::uint8_t>{5, 6, 1, 6, 22});

        std::vector<std::uint8_t> twice(once.size());
        REQUIRE(BeaufortKeyKernel::apply_host(once, twice, key, view.value()).ok());
        REQUIRE(twice == host_in);
    }

    SECTION("with interrupts") {
        StatusOr<InterruptPolicy> policy =
            InterruptPolicy::from_skip_indices(std::vector<std::size_t>{1, 3});
        REQUIRE(policy.ok());
        StatusOr<InterruptDeviceView> view =
            InterruptDeviceView::from_policy(policy.value(), host_in.size());
        REQUIRE(view.ok());

        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(BeaufortKeyTransform::kernel(plain, cpu_out, from_bytes(key),
                                             policy.value().skip_indices())
                    .ok());

        std::vector<std::uint8_t> host_out(host_in.size());
        REQUIRE(BeaufortKeyKernel::apply_host(host_in, host_out, key, view.value()).ok());
        REQUIRE(from_bytes(host_out) == cpu_out);

        std::vector<std::uint8_t> recovered(host_out.size());
        REQUIRE(BeaufortKeyKernel::apply_host(host_out, recovered, key, view.value()).ok());
        REQUIRE(recovered == host_in);
    }
}

TEST_CASE("CUDA beaufort constant key 28 matches Atbash identity", "[cuda][parity][beaufort]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> key{28};
    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{14}, Index29{28}};
    const std::vector<std::uint8_t> host_in = to_bytes(plain);

    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), host_in.size());
    REQUIRE(view.ok());

    std::vector<Index29> cpu_out(plain.size());
    REQUIRE(BeaufortKeyTransform::kernel(plain, cpu_out, from_bytes(key), {}).ok());

    std::vector<std::uint8_t> host_out(host_in.size());
    REQUIRE(BeaufortKeyKernel::apply_host(host_in, host_out, key, view.value()).ok());
    REQUIRE(from_bytes(host_out) == cpu_out);
    // 28 - x
    REQUIRE(host_out == std::vector<std::uint8_t>{28, 27, 14, 0});
}

TEST_CASE("CUDA beaufort random and sorted-skips encoding", "[cuda][parity][beaufort]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xBEA701u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> plain;
    plain.reserve(48);
    for (std::size_t i = 0; i < 48; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }
    const std::vector<std::uint8_t> key{
        static_cast<std::uint8_t>(dist(rng)),
        static_cast<std::uint8_t>(dist(rng)),
    };
    StatusOr<InterruptPolicy> policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{0, 5, 20});
    REQUIRE(policy.ok());

    std::vector<Index29> cpu_out(plain.size());
    REQUIRE(
        BeaufortKeyTransform::kernel(plain, cpu_out, from_bytes(key), policy.value().skip_indices())
            .ok());

    const std::vector<std::uint8_t> host_in = to_bytes(plain);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(policy.value(), host_in.size());
    REQUIRE(view.ok());

    std::vector<std::uint8_t> host_out(host_in.size());
    REQUIRE(BeaufortKeyKernel::apply_host(host_in, host_out, key, view.value()).ok());
    REQUIRE(from_bytes(host_out) == cpu_out);

    const std::size_t large_T = 5000;
    std::vector<Index29> large_plain(large_T, Index29{4});
    StatusOr<InterruptPolicy> large_policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{11, 4096});
    REQUIRE(large_policy.ok());
    StatusOr<InterruptDeviceView> large_view =
        InterruptDeviceView::from_policy(large_policy.value(), large_T);
    REQUIRE(large_view.ok());
    REQUIRE(large_view.value().encoding() == InterruptDeviceView::Encoding::SortedSkips);

    std::vector<Index29> large_cpu(large_T);
    REQUIRE(BeaufortKeyTransform::kernel(large_plain, large_cpu, from_bytes(key),
                                         large_policy.value().skip_indices())
                .ok());

    const std::vector<std::uint8_t> large_in = to_bytes(large_plain);
    std::vector<std::uint8_t> large_out(large_T);
    REQUIRE(BeaufortKeyKernel::apply_host(large_in, large_out, key, large_view.value()).ok());
    REQUIRE(from_bytes(large_out) == large_cpu);
}

TEST_CASE("CUDA beaufort rejects empty key", "[cuda][parity][beaufort]") {
    REQUIRE(ParcaeCuda::available());
    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> out(2);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), 2);
    REQUIRE(view.ok());
    REQUIRE_FALSE(BeaufortKeyKernel::apply_host(in, out, {}, view.value()).ok());
}

#else

TEST_CASE("CUDA beaufort skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][beaufort]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise BeaufortKeyKernel");
}

#endif
