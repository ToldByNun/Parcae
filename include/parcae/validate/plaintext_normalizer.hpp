#ifndef PLAINTEXT_NORMALIZER_HPP
#define PLAINTEXT_NORMALIZER_HPP

#include "parcae/core/status_or.hpp"
#include "parcae/gematria/latin_codec.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

/// Canonical Latin projection used for `normalized_plaintext_sha256`.
/// Strips long ASCII hex literal runs, keeps letters only (A–Z), folds aliases
/// to preferred Gematria labels via greedy longest-match round-trip.
class PlaintextNormalizer {
public:
    explicit PlaintextNormalizer(const LatinCodec& codec) : codec_(&codec) {}

    [[nodiscard]] StatusOr<std::string> normalize(const std::string& plaintext) const {
        const std::string without_hex = strip_ascii_hex_literal_runs(plaintext);
        std::string letters;
        letters.reserve(without_hex.size());
        for (unsigned char ch : without_hex) {
            if (std::isalpha(ch) != 0) {
                letters.push_back(static_cast<char>(std::toupper(ch)));
            }
        }
        return codec_->round_trip_preferred(letters);
    }

    /// Preferred-label concatenation of decrypted Indices (same form as `normalize`).
    [[nodiscard]] std::string from_indices(const std::vector<Index29>& indices) const {
        return codec_->latinize(indices);
    }

private:
    /// Remove maximal `[0-9a-fA-F]+` runs that look like hex literals (contain at
    /// least one hex letter and are long enough not to eat Latin words built from
    /// A–F alone, e.g. "AN END").
    [[nodiscard]] static std::string strip_ascii_hex_literal_runs(const std::string& text) {
        constexpr std::size_t min_hex_literal_len = 16;
        std::string out;
        out.reserve(text.size());
        std::size_t i = 0;
        while (i < text.size()) {
            const unsigned char lead = static_cast<unsigned char>(text[i]);
            if (std::isxdigit(lead) == 0) {
                out.push_back(text[i]);
                ++i;
                continue;
            }

            std::size_t end = i;
            bool saw_hex_letter = false;
            while (end < text.size() && std::isxdigit(static_cast<unsigned char>(text[end])) != 0) {
                const unsigned char ch = static_cast<unsigned char>(text[end]);
                if ((ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F')) {
                    saw_hex_letter = true;
                }
                ++end;
            }

            const std::size_t run_len = end - i;
            if (saw_hex_letter && run_len >= min_hex_literal_len) {
                i = end;
                continue;
            }
            while (i < end) {
                out.push_back(text[i]);
                ++i;
            }
        }
        return out;
    }

    const LatinCodec* codec_;
};

#endif // PLAINTEXT_NORMALIZER_HPP
