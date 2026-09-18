#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "interrupt_device_view.hpp"
#include "parcae_cuda.hpp"
#include "params.hpp"
#include "vigenere_key_kernel.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

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

[[nodiscard]] std::vector<Index29> key_from_bytes(const std::vector<std::uint8_t>& bytes) {
    return from_bytes(bytes);
}

}  // namespace

TEST_CASE("CUDA vigenere parity hand vectors with and without interrupts", "[cuda][parity][vigenere]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> key{1, 2};
    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
    const std::vector<std::uint8_t> host_in = to_bytes(plain);

    SECTION("no interrupts") {
        StatusOr<InterruptDeviceView> view =
            InterruptDeviceView::from_policy(InterruptPolicy::none(), host_in.size());
        REQUIRE(view.ok());
        REQUIRE(view.value().encoding() == InterruptDeviceView::Encoding::Bitmask);

        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(VigenereKeyTransform::kernel(
                    plain, cpu_out, key_from_bytes(key), {}, TransformDirection::Encrypt)
                    .ok());

        std::vector<std::uint8_t> host_out(host_in.size(), 0xFFu);
        REQUIRE(VigenereKeyKernel::apply_host(
                    host_in, host_out, key, view.value(), CudaDir::Encrypt)
                    .ok());
        REQUIRE(from_bytes(host_out) == cpu_out);
        REQUIRE(
            host_out == std::vector<std::uint8_t>{1, 3, 3, 5});
    }

    SECTION("with interrupts") {
        StatusOr<InterruptPolicy> policy =
            InterruptPolicy::from_skip_indices(std::vector<std::size_t>{1, 3});
        REQUIRE(policy.ok());
        StatusOr<InterruptDeviceView> view =
            InterruptDeviceView::from_policy(policy.value(), host_in.size());
        REQUIRE(view.ok());

        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(VigenereKeyTransform::kernel(
                    plain,
                    cpu_out,
                    key_from_bytes(key),
                    policy.value().skip_indices(),
                    TransformDirection::Encrypt)
                    .ok());

        std::vector<std::uint8_t> host_out(host_in.size(), 0xFFu);
        REQUIRE(VigenereKeyKernel::apply_host(
                    host_in, host_out, key, view.value(), CudaDir::Encrypt)
                    .ok());
        REQUIRE(from_bytes(host_out) == cpu_out);
        REQUIRE(host_out == std::vector<std::uint8_t>{1, 1, 4, 3});

        std::vector<std::uint8_t> recovered(host_out.size());
        REQUIRE(VigenereKeyKernel::apply_host(
                    host_out, recovered, key, view.value(), CudaDir::Decrypt)
                    .ok());
        REQUIRE(recovered == host_in);
    }
}

TEST_CASE("CUDA vigenere parity random and sorted-skips encoding", "[cuda][parity][vigenere]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0x716E57u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> plain;
    plain.reserve(64);
    for (std::size_t i = 0; i < 64; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }
    const std::vector<std::uint8_t> key{
        static_cast<std::uint8_t>(dist(rng)),
        static_cast<std::uint8_t>(dist(rng)),
        static_cast<std::uint8_t>(dist(rng)),
    };
    StatusOr<InterruptPolicy> policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{2, 7, 33});
    REQUIRE(policy.ok());

    std::vector<Index29> cpu_out(plain.size());
    REQUIRE(VigenereKeyTransform::kernel(
                plain,
                cpu_out,
                key_from_bytes(key),
                policy.value().skip_indices(),
                TransformDirection::Decrypt)
                .ok());

    const std::vector<std::uint8_t> host_in = to_bytes(plain);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(policy.value(), host_in.size());
    REQUIRE(view.ok());
    REQUIRE(view.value().encoding() == InterruptDeviceView::Encoding::Bitmask);

    std::vector<std::uint8_t> host_out(host_in.size());
    REQUIRE(VigenereKeyKernel::apply_host(
                host_in, host_out, key, view.value(), CudaDir::Decrypt)
                .ok());
    REQUIRE(from_bytes(host_out) == cpu_out);

    // Force sorted-skips encoding with T > 4096.
    const std::size_t large_T = 5000;
    std::vector<Index29> large_plain(large_T, Index29{3});
    StatusOr<InterruptPolicy> large_policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{10, 100, 4096});
    REQUIRE(large_policy.ok());
    StatusOr<InterruptDeviceView> large_view =
        InterruptDeviceView::from_policy(large_policy.value(), large_T);
    REQUIRE(large_view.ok());
    REQUIRE(large_view.value().encoding() == InterruptDeviceView::Encoding::SortedSkips);

    std::vector<Index29> large_cpu(large_T);
    REQUIRE(VigenereKeyTransform::kernel(
                large_plain,
                large_cpu,
                key_from_bytes(key),
                large_policy.value().skip_indices(),
                TransformDirection::Encrypt)
                .ok());

    const std::vector<std::uint8_t> large_in = to_bytes(large_plain);
    std::vector<std::uint8_t> large_out(large_T);
    REQUIRE(VigenereKeyKernel::apply_host(
                large_in, large_out, key, large_view.value(), CudaDir::Encrypt)
                .ok());
    REQUIRE(from_bytes(large_out) == large_cpu);
}

TEST_CASE("CUDA vigenere rejects empty key and size mismatch", "[cuda][parity][vigenere]") {
    REQUIRE(ParcaeCuda::available());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> out(2);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), 2);
    REQUIRE(view.ok());

    REQUIRE_FALSE(VigenereKeyKernel::apply_host(in, out, {}, view.value(), CudaDir::Encrypt).ok());

    std::vector<std::uint8_t> short_out(1);
    const std::vector<std::uint8_t> key{1};
    REQUIRE_FALSE(
        VigenereKeyKernel::apply_host(in, short_out, key, view.value(), CudaDir::Encrypt).ok());
}

#else

TEST_CASE("CUDA vigenere skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][vigenere]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise VigenereKeyKernel");
}

#endif
