#ifndef INTERRUPT_DEVICE_VIEW_HPP
#define INTERRUPT_DEVICE_VIEW_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

/// Host encoding of `InterruptPolicy` for CUDA twins (`docs/architecture/cuda-abi.md`).
///
/// - `T ≤ kBitmaskMaxT`: `uint32_t` bitmask words; bit `i` set ⇒ skip consumable `i`.
/// - else: sorted unique `uint32_t` skip list (binary search, same as CPU).
///
/// Built on the host before H2D. Device code never infers interrupts.
class InterruptDeviceView {
public:
    /// Fixture-scale threshold from `cuda-abi.md` (§ Interrupt representation).
    static constexpr std::size_t kBitmaskMaxT = 4096;

    enum class Encoding : std::uint8_t {
        Bitmask = 0,
        SortedSkips = 1,
    };

    [[nodiscard]] static StatusOr<InterruptDeviceView> from_policy(const InterruptPolicy& policy,
                                                                   std::size_t consumable_length) {
        StatusOr<std::vector<std::uint32_t>> skips = to_u32_skips(policy);
        if (!skips.ok()) {
            return skips.status();
        }

        InterruptDeviceView view;
        view.consumable_length_ = consumable_length;
        view.sorted_skips_ = std::move(skips.value());

        if (consumable_length <= kBitmaskMaxT) {
            view.encoding_ = Encoding::Bitmask;
            const std::size_t word_count = bitmask_word_count(consumable_length);
            view.bitmask_words_.assign(word_count, 0u);
            for (std::uint32_t index : view.sorted_skips_) {
                if (static_cast<std::size_t>(index) >= consumable_length) {
                    return Status::error(
                        "InterruptDeviceView: skip index out of range for consumable_length");
                }
                const std::size_t word = static_cast<std::size_t>(index) / 32u;
                const std::uint32_t bit = 1u << (index % 32u);
                view.bitmask_words_[word] |= bit;
            }
        } else {
            view.encoding_ = Encoding::SortedSkips;
            for (std::uint32_t index : view.sorted_skips_) {
                if (static_cast<std::size_t>(index) >= consumable_length) {
                    return Status::error(
                        "InterruptDeviceView: skip index out of range for consumable_length");
                }
            }
        }

        return view;
    }

    [[nodiscard]] Encoding encoding() const noexcept { return encoding_; }

    [[nodiscard]] std::size_t consumable_length() const noexcept { return consumable_length_; }

    [[nodiscard]] const std::vector<std::uint32_t>& bitmask_words() const noexcept {
        return bitmask_words_;
    }

    [[nodiscard]] const std::vector<std::uint32_t>& sorted_skips() const noexcept {
        return sorted_skips_;
    }

    /// Same predicate as `InterruptPolicy::should_skip` for indices in `[0, T)`.
    [[nodiscard]] bool should_skip(std::size_t consumable_index) const {
        if (consumable_index >= consumable_length_) {
            return false;
        }
        if (encoding_ == Encoding::Bitmask) {
            const std::size_t word = consumable_index / 32u;
            const std::uint32_t bit = 1u << (consumable_index % 32u);
            return (bitmask_words_[word] & bit) != 0u;
        }
        return std::binary_search(sorted_skips_.begin(), sorted_skips_.end(),
                                  static_cast<std::uint32_t>(consumable_index));
    }

    [[nodiscard]] static std::size_t bitmask_word_count(std::size_t consumable_length) noexcept {
        if (consumable_length == 0) {
            return 0;
        }
        return (consumable_length + 31u) / 32u;
    }

private:
    InterruptDeviceView() = default;

    [[nodiscard]] static StatusOr<std::vector<std::uint32_t>>
    to_u32_skips(const InterruptPolicy& policy) {
        std::vector<std::uint32_t> out;
        out.reserve(policy.skip_indices().size());
        for (std::size_t index : policy.skip_indices()) {
            if (index > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                return Status::error("InterruptDeviceView: skip index exceeds uint32_t");
            }
            out.push_back(static_cast<std::uint32_t>(index));
        }
        return out;
    }

    Encoding encoding_ = Encoding::Bitmask;
    std::size_t consumable_length_ = 0;
    std::vector<std::uint32_t> bitmask_words_;
    std::vector<std::uint32_t> sorted_skips_;
};

#endif // INTERRUPT_DEVICE_VIEW_HPP
