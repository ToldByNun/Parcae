#ifndef LATIN_LABELS_HPP
#define LATIN_LABELS_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/gematria/gematria_profile.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/// Preferred-label view + alias fold for validation hashing.
/// Canonical fold: every accepted alias maps to the profile preferred label.
class LatinLabels {
public:
    explicit LatinLabels(const GematriaProfile& profile) : profile_(&profile) { rebuild_maps(); }

    [[nodiscard]] const std::string& preferred(Index29 index) const {
        return profile_->entry_at(index).preferred();
    }

    [[nodiscard]] std::string to_preferred_string(const std::vector<Index29>& indices) const {
        std::string out;
        for (Index29 index : indices) {
            out += preferred(index);
        }
        return out;
    }

    /// Resolve a single Latin label (preferred or alias) to Index29.
    [[nodiscard]] StatusOr<Index29> index_for_label(const std::string& label) const {
        const auto it = label_to_index_.find(label);
        if (it == label_to_index_.end()) {
            return Status::error("Unknown Latin label for Gematria profile");
        }
        return Index29{it->second};
    }

    /// Fold any accepted alias to the preferred spelling used for hashing.
    [[nodiscard]] StatusOr<std::string> fold_to_preferred(const std::string& label) const {
        StatusOr<Index29> index = index_for_label(label);
        if (!index.ok()) {
            return index.status();
        }
        return preferred(index.value());
    }

    /// Greedy longest-match parse of a Latin rune spelling into indices.
    [[nodiscard]] StatusOr<std::vector<Index29>> parse_indices(const std::string& latin) const {
        std::vector<Index29> out;
        std::size_t offset = 0;
        while (offset < latin.size()) {
            bool matched = false;
            for (const std::string& label : labels_by_length_desc_) {
                if (offset + label.size() <= latin.size() &&
                    latin.compare(offset, label.size(), label) == 0) {
                    out.push_back(Index29{label_to_index_.at(label)});
                    offset += label.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                return Status::error("Unable to parse Latin label sequence");
            }
        }
        return out;
    }

    /// Parse then re-emit preferred labels — canonical form for validation hashes.
    [[nodiscard]] StatusOr<std::string> canonicalize(const std::string& latin) const {
        StatusOr<std::vector<Index29>> indices = parse_indices(latin);
        if (!indices.ok()) {
            return indices.status();
        }
        return to_preferred_string(indices.value());
    }

private:
    void rebuild_maps() {
        label_to_index_.clear();
        labels_by_length_desc_.clear();

        for (const GematriaEntry& entry : profile_->entries()) {
            label_to_index_.emplace(entry.preferred(), entry.index().value());
            for (const std::string& label : entry.labels()) {
                label_to_index_.emplace(label, entry.index().value());
            }
        }

        labels_by_length_desc_.reserve(label_to_index_.size());
        for (const auto& [label, _index] : label_to_index_) {
            labels_by_length_desc_.push_back(label);
        }
        std::sort(labels_by_length_desc_.begin(), labels_by_length_desc_.end(),
                  [](const std::string& left, const std::string& right) {
                      if (left.size() != right.size()) {
                          return left.size() > right.size();
                      }
                      return left < right;
                  });
    }

    const GematriaProfile* profile_;
    std::unordered_map<std::string, std::uint8_t> label_to_index_;
    std::vector<std::string> labels_by_length_desc_;
};

#endif // LATIN_LABELS_HPP
