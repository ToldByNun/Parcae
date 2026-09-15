#ifndef FIXTURE_HPP
#define FIXTURE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class FixtureLiteralRegion {
public:
    FixtureLiteralRegion(
        std::string kind,
        std::string role,
        std::string value_file,
        std::string compare)
        : kind_(std::move(kind)),
          role_(std::move(role)),
          value_file_(std::move(value_file)),
          compare_(std::move(compare)) {}

    [[nodiscard]] const std::string& kind() const noexcept {
        return kind_;
    }

    [[nodiscard]] const std::string& role() const noexcept {
        return role_;
    }

    [[nodiscard]] const std::string& value_file() const noexcept {
        return value_file_;
    }

    [[nodiscard]] const std::string& compare() const noexcept {
        return compare_;
    }

private:
    std::string kind_;
    std::string role_;
    std::string value_file_;
    std::string compare_;
};

class Fixture {
public:
    Fixture(
        std::string id,
        std::string ciphertext,
        std::string plaintext,
        std::string transform_id,
        std::string direction,
        std::string verification_status,
        bool recomputed_ok,
        std::vector<std::size_t> skip_indices,
        std::vector<FixtureLiteralRegion> literal_regions,
        std::optional<std::string> key_latin,
        std::optional<std::vector<int>> key_indices)
        : id_(std::move(id)),
          ciphertext_(std::move(ciphertext)),
          plaintext_(std::move(plaintext)),
          transform_id_(std::move(transform_id)),
          direction_(std::move(direction)),
          verification_status_(std::move(verification_status)),
          recomputed_ok_(recomputed_ok),
          skip_indices_(std::move(skip_indices)),
          literal_regions_(std::move(literal_regions)),
          key_latin_(std::move(key_latin)),
          key_indices_(std::move(key_indices)) {}

    [[nodiscard]] const std::string& id() const noexcept {
        return id_;
    }

    [[nodiscard]] const std::string& ciphertext() const noexcept {
        return ciphertext_;
    }

    [[nodiscard]] const std::string& plaintext() const noexcept {
        return plaintext_;
    }

    [[nodiscard]] const std::string& transform_id() const noexcept {
        return transform_id_;
    }

    [[nodiscard]] const std::string& direction() const noexcept {
        return direction_;
    }

    [[nodiscard]] const std::string& verification_status() const noexcept {
        return verification_status_;
    }

    [[nodiscard]] bool recomputed_ok() const noexcept {
        return recomputed_ok_;
    }

    [[nodiscard]] const std::vector<std::size_t>& skip_indices() const noexcept {
        return skip_indices_;
    }

    [[nodiscard]] const std::vector<FixtureLiteralRegion>& literal_regions() const noexcept {
        return literal_regions_;
    }

    [[nodiscard]] const std::optional<std::string>& key_latin() const noexcept {
        return key_latin_;
    }

    [[nodiscard]] const std::optional<std::vector<int>>& key_indices() const noexcept {
        return key_indices_;
    }

    [[nodiscard]] Status validate_lock_rules() const {
        if (verification_status_ == "locked" && !recomputed_ok_) {
            return Status::error("Locked fixture requires recomputed_ok=true");
        }
        if (verification_status_ != "draft" && verification_status_ != "locked") {
            return Status::error("verification.status must be draft or locked");
        }
        return Status::success();
    }

private:
    std::string id_;
    std::string ciphertext_;
    std::string plaintext_;
    std::string transform_id_;
    std::string direction_;
    std::string verification_status_;
    bool recomputed_ok_ = false;
    std::vector<std::size_t> skip_indices_;
    std::vector<FixtureLiteralRegion> literal_regions_;
    std::optional<std::string> key_latin_;
    std::optional<std::vector<int>> key_indices_;
};

#endif // FIXTURE_HPP
