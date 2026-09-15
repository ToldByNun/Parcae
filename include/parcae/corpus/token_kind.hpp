#ifndef TOKEN_KIND_HPP
#define TOKEN_KIND_HPP

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

private:
    TokenKindUtil() = delete;
};

#endif // TOKEN_KIND_HPP
