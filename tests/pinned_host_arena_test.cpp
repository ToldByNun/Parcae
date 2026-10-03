#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "device_buffer.hpp"
#include "parcae_cuda.hpp"
#include "pinned_host_arena.hpp"

#include <cstdint>
#include <utility>
#include <vector>

TEST_CASE("CUDA PinnedHostArena empty allocate", "[cuda][pinned]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<PinnedHostArena> arena = PinnedHostArena::allocate(0);
    REQUIRE(arena.ok());
    REQUIRE(arena.value().empty());
    REQUIRE(arena.value().capacity() == 0);
    REQUIRE(arena.value().fill_from({}).ok());
}

TEST_CASE("CUDA PinnedHostArena allocate and fill", "[cuda][pinned]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> payload{0, 1, 2, 28, 7, 13, 255, 29};
    StatusOr<PinnedHostArena> arena = PinnedHostArena::allocate(payload.size());
    REQUIRE(arena.ok());
    REQUIRE(arena.value().capacity() == payload.size());
    REQUIRE(arena.value().fill_from(payload).ok());

    StatusOr<std::span<std::uint8_t>> view = arena.value().span_prefix(payload.size());
    REQUIRE(view.ok());
    REQUIRE(std::vector<std::uint8_t>(view.value().begin(), view.value().end()) == payload);
}

TEST_CASE("CUDA PinnedHostArena fill / span_prefix exceed capacity", "[cuda][pinned]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<PinnedHostArena> arena = PinnedHostArena::allocate(4);
    REQUIRE(arena.ok());

    const std::vector<std::uint8_t> too_big(8, 1);
    REQUIRE_FALSE(arena.value().fill_from(too_big).ok());
    REQUIRE_FALSE(arena.value().span_prefix(5).ok());
    REQUIRE(arena.value().span_prefix(4).ok());
}

TEST_CASE("CUDA PinnedHostArena ensure_capacity grow-only", "[cuda][pinned]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<PinnedHostArena> arena = PinnedHostArena::allocate(4);
    REQUIRE(arena.ok());
    const std::vector<std::uint8_t> prefix{10, 20, 30, 40};
    REQUIRE(arena.value().fill_from(prefix).ok());

    REQUIRE(arena.value().ensure_capacity(4).ok());
    REQUIRE(arena.value().capacity() == 4);

    REQUIRE(arena.value().ensure_capacity(16).ok());
    REQUIRE(arena.value().capacity() == 16);

    StatusOr<std::span<std::uint8_t>> kept = arena.value().span_prefix(4);
    REQUIRE(kept.ok());
    REQUIRE(std::vector<std::uint8_t>(kept.value().begin(), kept.value().end()) == prefix);

    // Shrink request is a no-op (grow-only).
    REQUIRE(arena.value().ensure_capacity(8).ok());
    REQUIRE(arena.value().capacity() == 16);
}

TEST_CASE("CUDA PinnedHostArena move leaves source empty", "[cuda][pinned]") {
    REQUIRE(ParcaeCuda::available());

    StatusOr<PinnedHostArena> arena = PinnedHostArena::allocate(32);
    REQUIRE(arena.ok());
    PinnedHostArena moved = std::move(arena.value());
    REQUIRE(moved.capacity() == 32);
    REQUIRE(arena.value().empty());
    REQUIRE(arena.value().data() == nullptr);
}

TEST_CASE("CUDA PinnedHostArena stages DeviceBuffer H2D", "[cuda][pinned]") {
    REQUIRE(ParcaeCuda::available());

    const std::vector<std::uint8_t> host_in{3, 1, 4, 1, 5, 9, 2, 6};
    StatusOr<PinnedHostArena> pinned = PinnedHostArena::allocate(host_in.size());
    REQUIRE(pinned.ok());
    REQUIRE(pinned.value().fill_from(host_in).ok());

    StatusOr<DeviceBuffer<std::uint8_t>> device =
        DeviceBuffer<std::uint8_t>::allocate(host_in.size());
    REQUIRE(device.ok());
    REQUIRE(device.value()
                .copy_from_host(pinned.value().span_prefix(host_in.size()).value())
                .ok());

    std::vector<std::uint8_t> host_out(host_in.size());
    REQUIRE(device.value().copy_to_host(host_out).ok());
    REQUIRE(host_out == host_in);
}

#else

TEST_CASE("CUDA PinnedHostArena skipped (PARCAE_HAS_CUDA unset)", "[cuda][pinned]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise PinnedHostArena");
}

#endif
