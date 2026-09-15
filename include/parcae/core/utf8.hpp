#ifndef UTF8_HPP
#define UTF8_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

class Utf8 {
public:
    struct Codepoint {
        char32_t value = 0;
        std::size_t size = 0;
    };

    [[nodiscard]] static StatusOr<Codepoint> decode_at(const std::string& text, std::size_t offset) {
        if (offset >= text.size()) {
            return Status::error("UTF-8 decode past end of string");
        }

        const auto lead = static_cast<unsigned char>(text[offset]);
        Codepoint out;

        if (lead <= 0x7F) {
            out.value = lead;
            out.size = 1;
            return out;
        }

        if ((lead & 0xE0) == 0xC0) {
            if (offset + 1 >= text.size()) {
                return Status::error("Truncated 2-byte UTF-8 sequence");
            }
            const auto b1 = static_cast<unsigned char>(text[offset + 1]);
            if ((b1 & 0xC0) != 0x80) {
                return Status::error("Invalid UTF-8 continuation byte");
            }
            out.value = static_cast<char32_t>(((lead & 0x1F) << 6) | (b1 & 0x3F));
            out.size = 2;
            if (out.value < 0x80) {
                return Status::error("Overlong UTF-8 encoding");
            }
            return out;
        }

        if ((lead & 0xF0) == 0xE0) {
            if (offset + 2 >= text.size()) {
                return Status::error("Truncated 3-byte UTF-8 sequence");
            }
            const auto b1 = static_cast<unsigned char>(text[offset + 1]);
            const auto b2 = static_cast<unsigned char>(text[offset + 2]);
            if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) {
                return Status::error("Invalid UTF-8 continuation byte");
            }
            out.value = static_cast<char32_t>(((lead & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F));
            out.size = 3;
            if (out.value < 0x800) {
                return Status::error("Overlong UTF-8 encoding");
            }
            return out;
        }

        if ((lead & 0xF8) == 0xF0) {
            if (offset + 3 >= text.size()) {
                return Status::error("Truncated 4-byte UTF-8 sequence");
            }
            const auto b1 = static_cast<unsigned char>(text[offset + 1]);
            const auto b2 = static_cast<unsigned char>(text[offset + 2]);
            const auto b3 = static_cast<unsigned char>(text[offset + 3]);
            if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) {
                return Status::error("Invalid UTF-8 continuation byte");
            }
            out.value = static_cast<char32_t>(
                ((lead & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F));
            out.size = 4;
            if (out.value < 0x10000 || out.value > 0x10FFFF) {
                return Status::error("Invalid Unicode codepoint");
            }
            return out;
        }

        return Status::error("Invalid UTF-8 lead byte");
    }

    [[nodiscard]] static std::string encode(char32_t codepoint) {
        std::string out;
        if (codepoint <= 0x7F) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        return out;
    }

private:
    Utf8() = delete;
};

#endif // UTF8_HPP
