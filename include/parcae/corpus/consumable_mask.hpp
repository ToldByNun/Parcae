#ifndef CONSUMABLE_MASK_HPP
#define CONSUMABLE_MASK_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

/// Parallel Index29 + participation mask for consumable runes.
/// `mask[i] == true` means rune `indices[i]` participates in the cipher transform.
class ConsumableMask {
public:
    [[nodiscard]] static StatusOr<ConsumableMask> create(
        std::vector<Index29> indices,
        std::vector<std::uint8_t> mask) {
        if (indices.size() != mask.size()) {
            return Status::error("ConsumableMask indices/mask size mismatch");
        }
        for (const std::uint8_t bit : mask) {
            if (bit > 1) {
                return Status::error("ConsumableMask mask values must be 0 or 1");
            }
        }
        return ConsumableMask{std::move(indices), std::move(mask)};
    }

    [[nodiscard]] static ConsumableMask all_participating(std::vector<Index29> indices) {
        std::vector<std::uint8_t> mask(indices.size(), 1);
        return ConsumableMask{std::move(indices), std::move(mask)};
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return indices_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return indices_.empty();
    }

    [[nodiscard]] Index29 index_at(std::size_t i) const {
        return indices_.at(i);
    }

    [[nodiscard]] bool participates(std::size_t i) const {
        return mask_.at(i) != 0;
    }

    void set_participates(std::size_t i, bool participates) {
        mask_.at(i) = participates ? 1 : 0;
    }

    [[nodiscard]] std::span<const Index29> indices() const noexcept {
        return indices_;
    }

    [[nodiscard]] std::span<const std::uint8_t> mask() const noexcept {
        return mask_;
    }

private:
    ConsumableMask(std::vector<Index29> indices, std::vector<std::uint8_t> mask)
        : indices_(std::move(indices)), mask_(std::move(mask)) {}

    std::vector<Index29> indices_;
    std::vector<std::uint8_t> mask_;
};

#endif // CONSUMABLE_MASK_HPP
