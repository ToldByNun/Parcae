#ifndef GEMATRIA_PROFILE_HPP
#define GEMATRIA_PROFILE_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/gematria/gematria_entry.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class GematriaProfile {
public:
    [[nodiscard]] static StatusOr<GematriaProfile> create(std::string profile_id,
                                                          std::vector<GematriaEntry> entries) {
        Status status = validate(entries);
        if (!status.ok()) {
            return status;
        }

        GematriaProfile profile;
        profile.profile_id_ = std::move(profile_id);
        profile.entries_ = std::move(entries);
        profile.finalize();
        return profile;
    }

    [[nodiscard]] const std::string& profile_id() const noexcept { return profile_id_; }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    [[nodiscard]] const std::vector<GematriaEntry>& entries() const noexcept { return entries_; }

    [[nodiscard]] const GematriaEntry& entry_at(Index29 index) const {
        return entries_.at(index.value());
    }

    [[nodiscard]] StatusOr<Index29> index_for_rune(const std::string& rune) const {
        const auto it = rune_to_index_.find(rune);
        if (it == rune_to_index_.end()) {
            return Status::error("Unknown rune in Gematria profile");
        }
        return Index29{it->second};
    }

    [[nodiscard]] StatusOr<std::string> rune_for_index(Index29 index) const {
        if (index.value() >= entries_.size()) {
            return Status::error("Index out of range for Gematria profile");
        }
        return entries_[index.value()].rune();
    }

    [[nodiscard]] StatusOr<Index29> index_for_prime(std::uint32_t prime) const {
        const auto it = prime_to_index_.find(prime);
        if (it == prime_to_index_.end()) {
            return Status::error("Unknown prime in Gematria profile");
        }
        return Index29{it->second};
    }

    [[nodiscard]] StatusOr<std::uint32_t> prime_for_index(Index29 index) const {
        if (index.value() >= entries_.size()) {
            return Status::error("Index out of range for Gematria profile");
        }
        return entries_[index.value()].prime();
    }

private:
    friend class GematriaProfileLoader;

    GematriaProfile() = default;

    [[nodiscard]] static Status validate(const std::vector<GematriaEntry>& entries) {
        if (entries.size() != Index29::modulus) {
            return Status::error("Gematria profile must contain exactly 29 entries");
        }

        static constexpr std::uint32_t k_first_29_primes[Index29::modulus] = {
            2,  3,  5,  7,  11, 13, 17, 19, 23, 29, 31,  37,  41,  43,  47,
            53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
        };

        std::unordered_map<std::string, std::uint8_t> seen_runes;
        std::unordered_map<std::uint32_t, std::uint8_t> seen_primes;
        std::vector<bool> seen_indices(Index29::modulus, false);

        for (const GematriaEntry& entry : entries) {
            const std::uint8_t idx = entry.index().value();
            if (seen_indices[idx]) {
                return Status::error("Gematria profile indices are not unique");
            }
            seen_indices[idx] = true;

            if (entry.prime() != k_first_29_primes[idx]) {
                return Status::error(
                    "Gematria profile primes must be the first 29 primes in index order");
            }

            if (entry.rune().empty()) {
                return Status::error("Gematria profile rune must be non-empty");
            }
            if (entry.preferred().empty()) {
                return Status::error("Gematria profile preferred label must be non-empty");
            }
            if (entry.labels().empty()) {
                return Status::error("Gematria profile labels must be non-empty");
            }

            if (!seen_runes.emplace(entry.rune(), idx).second) {
                return Status::error("Gematria profile runes are not unique");
            }
            if (!seen_primes.emplace(entry.prime(), idx).second) {
                return Status::error("Gematria profile primes are not unique");
            }
        }

        for (std::uint8_t i = 0; i < Index29::modulus; ++i) {
            if (!seen_indices[i]) {
                return Status::error("Gematria profile indices must be contiguous 0..28");
            }
        }

        if (seen_runes.size() != Index29::modulus || seen_primes.size() != Index29::modulus) {
            return Status::error("Gematria profile rune/prime maps are not bijective");
        }

        return Status::success();
    }

    void finalize() {
        std::sort(entries_.begin(), entries_.end(),
                  [](const GematriaEntry& left, const GematriaEntry& right) {
                      return left.index().value() < right.index().value();
                  });

        rune_to_index_.clear();
        prime_to_index_.clear();
        for (const GematriaEntry& entry : entries_) {
            rune_to_index_.emplace(entry.rune(), entry.index().value());
            prime_to_index_.emplace(entry.prime(), entry.index().value());
        }
    }

    std::string profile_id_;
    std::vector<GematriaEntry> entries_;
    std::unordered_map<std::string, std::uint8_t> rune_to_index_;
    std::unordered_map<std::uint32_t, std::uint8_t> prime_to_index_;
};

#endif // GEMATRIA_PROFILE_HPP
