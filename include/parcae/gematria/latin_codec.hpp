#ifndef LATIN_CODEC_HPP
#define LATIN_CODEC_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/gematria/gematria_profile.hpp"
#include "parcae/gematria/latin_labels.hpp"

#include <string>
#include <vector>

/// Index29 ↔ preferred Latin spelling (multi-letter labels via longest-match).
class LatinCodec {
public:
    explicit LatinCodec(const GematriaProfile& profile) : labels_(profile) {}

    /// Encode indices as concatenated preferred Latin labels (e.g. TH+ING → "THING").
    [[nodiscard]] std::string latinize(const std::vector<Index29>& indices) const {
        return labels_.to_preferred_string(indices);
    }

    /// Decode a Latin spelling into indices (aliases accepted; greedy longest match).
    [[nodiscard]] StatusOr<std::vector<Index29>> delatinize(const std::string& latin) const {
        return labels_.parse_indices(latin);
    }

    /// delatinize → latinize using preferred labels (canonical round-trip form).
    [[nodiscard]] StatusOr<std::string> round_trip_preferred(const std::string& latin) const {
        StatusOr<std::vector<Index29>> indices = delatinize(latin);
        if (!indices.ok()) {
            return indices.status();
        }
        return latinize(indices.value());
    }

    [[nodiscard]] const LatinLabels& labels() const noexcept {
        return labels_;
    }

private:
    LatinLabels labels_;
};

#endif // LATIN_CODEC_HPP
