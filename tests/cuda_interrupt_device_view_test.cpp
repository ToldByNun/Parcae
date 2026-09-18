#include "interrupt_device_view.hpp"

#include "parcae/interrupt/policy.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

TEST_CASE("InterruptDeviceView prefers bitmask for T <= 4096", "[cuda][interrupt]") {
    StatusOr<InterruptPolicy> policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{0, 3, 31, 32});
    REQUIRE(policy.ok());

    StatusOr<InterruptDeviceView> view = InterruptDeviceView::from_policy(policy.value(), 64);
    REQUIRE(view.ok());
    REQUIRE(view.value().encoding() == InterruptDeviceView::Encoding::Bitmask);
    REQUIRE(view.value().consumable_length() == 64);
    REQUIRE(view.value().bitmask_words().size() == 2);
    REQUIRE(view.value().sorted_skips().size() == 4);

    REQUIRE(view.value().bitmask_words()[0] == ((1u << 0) | (1u << 3) | (1u << 31)));
    REQUIRE(view.value().bitmask_words()[1] == (1u << 0));  // index 32

    for (std::size_t i = 0; i < 64; ++i) {
        REQUIRE(view.value().should_skip(i) == policy.value().should_skip(i));
    }
}

TEST_CASE("InterruptDeviceView empty policy is all-clear bitmask", "[cuda][interrupt]") {
    const InterruptPolicy policy = InterruptPolicy::none();
    StatusOr<InterruptDeviceView> view = InterruptDeviceView::from_policy(policy, 100);
    REQUIRE(view.ok());
    REQUIRE(view.value().encoding() == InterruptDeviceView::Encoding::Bitmask);
    REQUIRE(view.value().bitmask_words().size() == InterruptDeviceView::bitmask_word_count(100));
    for (std::uint32_t word : view.value().bitmask_words()) {
        REQUIRE(word == 0u);
    }
    REQUIRE_FALSE(view.value().should_skip(0));
    REQUIRE_FALSE(view.value().should_skip(99));
}

TEST_CASE("InterruptDeviceView uses sorted skips when T > 4096", "[cuda][interrupt]") {
    StatusOr<InterruptPolicy> policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{10, 4096, 5000});
    REQUIRE(policy.ok());

    // Must exceed kBitmaskMaxT and be > max skip (5000).
    const std::size_t T = 5001;
    REQUIRE(T > InterruptDeviceView::kBitmaskMaxT);
    StatusOr<InterruptDeviceView> view = InterruptDeviceView::from_policy(policy.value(), T);
    REQUIRE(view.ok());
    REQUIRE(view.value().encoding() == InterruptDeviceView::Encoding::SortedSkips);
    REQUIRE(view.value().bitmask_words().empty());
    REQUIRE(view.value().sorted_skips() == std::vector<std::uint32_t>{10, 4096, 5000});

    REQUIRE(view.value().should_skip(10));
    REQUIRE(view.value().should_skip(4096));
    REQUIRE(view.value().should_skip(5000));
    REQUIRE_FALSE(view.value().should_skip(11));
    for (std::size_t i = 0; i < 64; ++i) {
        REQUIRE(view.value().should_skip(i) == policy.value().should_skip(i));
    }
}

TEST_CASE("InterruptDeviceView rejects skip >= consumable_length", "[cuda][interrupt]") {
    StatusOr<InterruptPolicy> policy =
        InterruptPolicy::from_skip_indices(std::vector<std::size_t>{5});
    REQUIRE(policy.ok());
    REQUIRE_FALSE(InterruptDeviceView::from_policy(policy.value(), 5).ok());
    REQUIRE(InterruptDeviceView::from_policy(policy.value(), 6).ok());
}

TEST_CASE("InterruptDeviceView matches welcome-scale skip set", "[cuda][interrupt]") {
    // Subset of welcome interrupts; T large enough for fixture-scale bitmask.
    StatusOr<InterruptPolicy> policy = InterruptPolicy::from_skip_indices(
        std::vector<std::size_t>{48, 74, 84, 132, 159, 160, 250, 421, 443, 465, 514});
    REQUIRE(policy.ok());

    constexpr std::size_t T = 600;
    StatusOr<InterruptDeviceView> view = InterruptDeviceView::from_policy(policy.value(), T);
    REQUIRE(view.ok());
    REQUIRE(view.value().encoding() == InterruptDeviceView::Encoding::Bitmask);
    REQUIRE(view.value().sorted_skips().size() == 11);

    for (std::size_t i = 0; i < T; ++i) {
        REQUIRE(view.value().should_skip(i) == policy.value().should_skip(i));
    }
}
