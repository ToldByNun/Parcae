#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/math/totient_keystream.hpp"
#include "parcae/transform/totient_prime_stream_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "interrupt_device_view.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"
#include "totient_prime_stream_kernel.hpp"

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

[[nodiscard]] StatusOr<std::vector<std::uint8_t>> host_shifts(std::size_t consumable,
                                                              std::size_t prime_start_index = 0) {
    std::vector<Index29> shifts(consumable);
    Status filled = TotientKeystream::shifts_into(shifts, prime_start_index);
    if (!filled.ok()) {
        return filled;
    }
    return to_bytes(shifts);
}

} // namespace

TEST_CASE("CUDA totient parity hand vectors with and without interrupts",
          "[cuda][parity][totient]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<Index29> plain{Index29{0}, Index29{1}, Index29{2}, Index29{3}};
    const std::vector<std::uint8_t> host_in = to_bytes(plain);

    SECTION("no interrupts") {
        StatusOr<InterruptDeviceView> view =
            InterruptDeviceView::from_policy(InterruptPolicy::none(), host_in.size());
        REQUIRE(view.ok());
        StatusOr<std::vector<std::uint8_t>> shifts = host_shifts(4);
        REQUIRE(shifts.ok());

        std::vector<Index29> cpu_shifts(4);
        REQUIRE(TotientKeystream::shifts_into(cpu_shifts, 0).ok());
        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(TotientPrimeStreamTransform::kernel(plain, cpu_out, cpu_shifts, {},
                                                    TransformDirection::Encrypt)
                    .ok());

        std::vector<std::uint8_t> host_out(host_in.size(), 0xFFu);
        REQUIRE(TotientPrimeStreamKernel::apply_host(host_in, host_out, shifts.value(),
                                                     view.value(), CudaDir::Encrypt)
                    .ok());
        REQUIRE(from_bytes(host_out) == cpu_out);
        REQUIRE(host_out == std::vector<std::uint8_t>{1, 3, 6, 9});

        std::vector<std::uint8_t> recovered(host_out.size());
        REQUIRE(TotientPrimeStreamKernel::apply_host(host_out, recovered, shifts.value(),
                                                     view.value(), CudaDir::Decrypt)
                    .ok());
        REQUIRE(recovered == host_in);
    }

    SECTION("with interrupts") {
        StatusOr<InterruptPolicy> policy =
            InterruptPolicy::from_skip_indices(std::vector<std::size_t>{1});
        REQUIRE(policy.ok());
        StatusOr<InterruptDeviceView> view =
            InterruptDeviceView::from_policy(policy.value(), host_in.size());
        REQUIRE(view.ok());
        StatusOr<std::vector<std::uint8_t>> shifts = host_shifts(3);
        REQUIRE(shifts.ok());

        std::vector<Index29> cpu_shifts(3);
        REQUIRE(TotientKeystream::shifts_into(cpu_shifts, 0).ok());
        std::vector<Index29> cpu_out(plain.size());
        REQUIRE(TotientPrimeStreamTransform::kernel(plain, cpu_out, cpu_shifts,
                                                    policy.value().skip_indices(),
                                                    TransformDirection::Encrypt)
                    .ok());

        std::vector<std::uint8_t> host_out(host_in.size());
        REQUIRE(TotientPrimeStreamKernel::apply_host(host_in, host_out, shifts.value(),
                                                     view.value(), CudaDir::Encrypt)
                    .ok());
        REQUIRE(from_bytes(host_out) == cpu_out);
        REQUIRE(host_out == std::vector<std::uint8_t>{1, 1, 4, 7});
    }
}

TEST_CASE("CUDA totient random and sorted-skips encoding", "[cuda][parity][totient]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0x7011E7u);
    std::uniform_int_distribution<int> dist(0, 28);

    std::vector<Index29> plain;
    plain.reserve(40);
    for (std::size_t i = 0; i < 40; ++i) {
        plain.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }
    StatusOr<InterruptPolicy> policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{3, 8, 21});
    REQUIRE(policy.ok());

    const std::size_t consumable = 37;
    std::vector<Index29> cpu_shifts(consumable);
    REQUIRE(TotientKeystream::shifts_into(cpu_shifts, 2).ok());
    std::vector<Index29> cpu_out(plain.size());
    REQUIRE(TotientPrimeStreamTransform::kernel(plain, cpu_out, cpu_shifts,
                                                policy.value().skip_indices(),
                                                TransformDirection::Decrypt)
                .ok());

    const std::vector<std::uint8_t> host_in = to_bytes(plain);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(policy.value(), host_in.size());
    REQUIRE(view.ok());
    StatusOr<std::vector<std::uint8_t>> shifts = host_shifts(consumable, 2);
    REQUIRE(shifts.ok());

    std::vector<std::uint8_t> host_out(host_in.size());
    REQUIRE(TotientPrimeStreamKernel::apply_host(host_in, host_out, shifts.value(), view.value(),
                                                 CudaDir::Decrypt)
                .ok());
    REQUIRE(from_bytes(host_out) == cpu_out);

    const std::size_t large_T = 5000;
    std::vector<Index29> large_plain(large_T, Index29{5});
    StatusOr<InterruptPolicy> large_policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{0, 4096});
    REQUIRE(large_policy.ok());
    StatusOr<InterruptDeviceView> large_view =
        InterruptDeviceView::from_policy(large_policy.value(), large_T);
    REQUIRE(large_view.ok());
    REQUIRE(large_view.value().encoding() == InterruptDeviceView::Encoding::SortedSkips);

    const std::size_t large_consumable = large_T - 2;
    std::vector<Index29> large_cpu_shifts(large_consumable);
    REQUIRE(TotientKeystream::shifts_into(large_cpu_shifts, 0).ok());
    std::vector<Index29> large_cpu(large_T);
    REQUIRE(TotientPrimeStreamTransform::kernel(large_plain, large_cpu, large_cpu_shifts,
                                                large_policy.value().skip_indices(),
                                                TransformDirection::Encrypt)
                .ok());

    StatusOr<std::vector<std::uint8_t>> large_shifts = host_shifts(large_consumable);
    REQUIRE(large_shifts.ok());
    const std::vector<std::uint8_t> large_in = to_bytes(large_plain);
    std::vector<std::uint8_t> large_out(large_T);
    REQUIRE(TotientPrimeStreamKernel::apply_host(large_in, large_out, large_shifts.value(),
                                                 large_view.value(), CudaDir::Encrypt)
                .ok());
    REQUIRE(from_bytes(large_out) == large_cpu);
}

TEST_CASE("CUDA totient rejects wrong shifts length", "[cuda][parity][totient]") {
    REQUIRE(ParcaeCuda::available());
    std::vector<std::uint8_t> in{1, 2, 3};
    std::vector<std::uint8_t> out(3);
    StatusOr<InterruptDeviceView> view =
        InterruptDeviceView::from_policy(InterruptPolicy::none(), 3);
    REQUIRE(view.ok());
    std::vector<std::uint8_t> bad_shifts{1, 2}; // need 3
    REQUIRE_FALSE(
        TotientPrimeStreamKernel::apply_host(in, out, bad_shifts, view.value(), CudaDir::Encrypt)
            .ok());
}

#else

TEST_CASE("CUDA totient skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][totient]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise TotientPrimeStreamKernel");
}

#endif
