#ifndef RUNE_CODEC_HPP
#define RUNE_CODEC_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/utf8.hpp"
#include "parcae/gematria/gematria_profile.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class RuneCodec {
public:
    struct DecodeResult {
        Index29 index;
        std::size_t byte_size = 0;
        std::string utf8;
    };

    explicit RuneCodec(const GematriaProfile& profile) : profile_(&profile) {}

    [[nodiscard]] StatusOr<std::string> encode(Index29 index) const {
        return profile_->rune_for_index(index);
    }

    [[nodiscard]] StatusOr<std::string> encode_all(const std::vector<Index29>& indices) const {
        std::string out;
        for (Index29 index : indices) {
            StatusOr<std::string> piece = encode(index);
            if (!piece.ok()) {
                return piece.status();
            }
            out += piece.value();
        }
        return out;
    }

    /// Decode one Gematria rune starting at `offset`. Unknown runes fail in strict mode.
    [[nodiscard]] StatusOr<DecodeResult> decode_at(const std::string& text, std::size_t offset,
                                                   bool strict = true) const {
        StatusOr<Utf8::Codepoint> cp = Utf8::decode_at(text, offset);
        if (!cp.ok()) {
            return cp.status();
        }

        const std::string rune = text.substr(offset, cp.value().size);
        StatusOr<Index29> index = profile_->index_for_rune(rune);
        if (!index.ok()) {
            if (strict) {
                return Status::error("Unknown rune in strict mode");
            }
            return index.status();
        }

        DecodeResult result;
        result.index = index.value();
        result.byte_size = cp.value().size;
        result.utf8 = rune;
        return result;
    }

    [[nodiscard]] StatusOr<std::vector<Index29>> decode_all(const std::string& text,
                                                            bool strict = true) const {
        std::vector<Index29> out;
        std::size_t offset = 0;
        while (offset < text.size()) {
            StatusOr<DecodeResult> decoded = decode_at(text, offset, strict);
            if (!decoded.ok()) {
                return decoded.status();
            }
            out.push_back(decoded.value().index);
            offset += decoded.value().byte_size;
        }
        return out;
    }

private:
    const GematriaProfile* profile_;
};

#endif // RUNE_CODEC_HPP
