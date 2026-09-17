#ifndef TOKEN_KIND_HPP
#define TOKEN_KIND_HPP

#include <string_view>

/// Lexical kind of a Liber Primus transcript token.
enum class TokenKind {
    Rune,
    WordSep,
    ClauseSep,
    LineSep,
    ParaSep,
    PageMark,
    ChapterSep,
    Whitespace,
    Newline,
    Number,
    Hex,
    Unknown,
};

class TokenKindUtil {
public:
    [[nodiscard]] static constexpr bool is_consumable(TokenKind kind) noexcept {
        return kind == TokenKind::Rune;
    }

    [[nodiscard]] static constexpr std::string_view to_string(TokenKind kind) noexcept {
        switch (kind) {
        case TokenKind::Rune:
            return "Rune";
        case TokenKind::WordSep:
            return "WordSep";
        case TokenKind::ClauseSep:
            return "ClauseSep";
        case TokenKind::LineSep:
            return "LineSep";
        case TokenKind::ParaSep:
            return "ParaSep";
        case TokenKind::PageMark:
            return "PageMark";
        case TokenKind::ChapterSep:
            return "ChapterSep";
        case TokenKind::Whitespace:
            return "Whitespace";
        case TokenKind::Newline:
            return "Newline";
        case TokenKind::Number:
            return "Number";
        case TokenKind::Hex:
            return "Hex";
        case TokenKind::Unknown:
            return "Unknown";
        }
        return "Unknown";
    }

private:
    TokenKindUtil() = delete;
};

#endif // TOKEN_KIND_HPP
