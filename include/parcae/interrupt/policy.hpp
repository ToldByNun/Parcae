#ifndef INTERRUPT_POLICY_HPP
#define INTERRUPT_POLICY_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Explicit skip-index interrupt policy (`explicit_skip_indices_v0`).
class InterruptPolicy {
public:
    [[nodiscard]] static InterruptPolicy none() {
        return InterruptPolicy{"explicit_skip_indices_v0", 0, {}};
    }

    [[nodiscard]] static StatusOr<InterruptPolicy> from_skip_indices(
        std::vector<std::size_t> skip_indices,
        std::size_t rune_index_base = 0) {
        if (rune_index_base != 0) {
            return Status::error("rune_index_base must be 0");
        }
        Status sorted = validate_sorted_unique(skip_indices);
        if (!sorted.ok()) {
            return sorted;
        }
        return InterruptPolicy{"explicit_skip_indices_v0", 0, std::move(skip_indices)};
    }

    [[nodiscard]] static StatusOr<InterruptPolicy> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("interrupt policy must be a JSON object");
        }
        const std::string policy_id =
            root.value("policy_id", std::string{"explicit_skip_indices_v0"});
        if (policy_id != "explicit_skip_indices_v0") {
            return Status::error("Unsupported interrupt policy_id");
        }
        const std::size_t rune_index_base = root.value("rune_index_base", std::size_t{0});
        if (rune_index_base != 0) {
            return Status::error("rune_index_base must be 0");
        }

        std::vector<std::size_t> skip_indices;
        if (root.contains("skip_indices")) {
            if (!root.at("skip_indices").is_array()) {
                return Status::error("skip_indices must be an array");
            }
            for (const nlohmann::json& item : root.at("skip_indices")) {
                skip_indices.push_back(item.get<std::size_t>());
            }
        }
        return from_skip_indices(std::move(skip_indices), rune_index_base);
    }

    [[nodiscard]] nlohmann::json to_json() const {
        return nlohmann::json{
            {"policy_id", policy_id_},
            {"rune_index_base", rune_index_base_},
            {"skip_indices", skip_indices_},
        };
    }

    [[nodiscard]] const std::string& policy_id() const noexcept {
        return policy_id_;
    }

    [[nodiscard]] std::size_t rune_index_base() const noexcept {
        return rune_index_base_;
    }

    [[nodiscard]] const std::vector<std::size_t>& skip_indices() const noexcept {
        return skip_indices_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return skip_indices_.empty();
    }

    [[nodiscard]] bool should_skip(std::size_t consumable_index) const {
        return std::binary_search(skip_indices_.begin(), skip_indices_.end(), consumable_index);
    }

private:
    InterruptPolicy(
        std::string policy_id,
        std::size_t rune_index_base,
        std::vector<std::size_t> skip_indices)
        : policy_id_(std::move(policy_id)),
          rune_index_base_(rune_index_base),
          skip_indices_(std::move(skip_indices)) {}

    [[nodiscard]] static Status validate_sorted_unique(std::vector<std::size_t>& skip_indices) {
        std::sort(skip_indices.begin(), skip_indices.end());
        for (std::size_t i = 1; i < skip_indices.size(); ++i) {
            if (skip_indices[i] == skip_indices[i - 1]) {
                return Status::error("skip_indices must be unique");
            }
        }
        return Status::success();
    }

    std::string policy_id_;
    std::size_t rune_index_base_ = 0;
    std::vector<std::size_t> skip_indices_;
};

#endif // INTERRUPT_POLICY_HPP
