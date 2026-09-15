#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/utf8.hpp"
#include "parcae/corpus/separator_grammar.hpp"
#include "parcae/corpus/token.hpp"
#include "parcae/corpus/token_stream.hpp"
#include "parcae/gematria/gematria_profile.hpp"
#include "parcae/gematria/rune_codec.hpp"

#include <cctype>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class Tokenizer {
public:
    Tokenizer(const GematriaProfile& profile, const SeparatorGrammar& grammar)
        : profile_(&profile), grammar_(&grammar), rune_codec_(profile) {}

    /// ASCII grammar first. Strict mode rejects unknown non-separator symbols.
    [[nodiscard]] StatusOr<TokenStream> tokenize(const std::string& text, bool strict = true) const {
        std::vector<Token> tokens;
        std::size_t offset = 0;
        std::size_t consumable = 0;

        while (offset < text.size()) {
            if (const auto newline = match_newline(text, offset)) {
                tokens.push_back(Token::non_rune(
                    TokenKind::Newline, offset, offset + newline->size(), *newline));
                offset += newline->size();
                continue;
            }

            if (const auto whitespace = match_whitespace(text, offset)) {
                tokens.push_back(Token::non_rune(
                    TokenKind::Whitespace, offset, offset + whitespace->size(), *whitespace));
                offset += whitespace->size();
                continue;
            }

            if (const auto separator = match_separator(text, offset)) {
                StatusOr<TokenKind> kind = grammar_->kind_for_token(*separator);
                if (!kind.ok()) {
                    return kind.status();
                }
                tokens.push_back(Token::non_rune(
                    kind.value(), offset, offset + separator->size(), *separator));
                offset += separator->size();
                continue;
            }

            if (std::isdigit(static_cast<unsigned char>(text[offset])) != 0) {
                const std::size_t begin = offset;
                while (offset < text.size() &&
                       std::isdigit(static_cast<unsigned char>(text[offset])) != 0) {
                    ++offset;
                }
                tokens.push_back(Token::non_rune(
                    TokenKind::Number, begin, offset, text.substr(begin, offset - begin)));
                continue;
            }

            StatusOr<RuneCodec::DecodeResult> rune = rune_codec_.decode_at(text, offset, false);
            if (rune.ok()) {
                tokens.push_back(Token::rune(
                    rune.value().index,
                    offset,
                    offset + rune.value().byte_size,
                    rune.value().utf8,
                    consumable));
                ++consumable;
                offset += rune.value().byte_size;
                continue;
            }

            StatusOr<Utf8::Codepoint> cp = Utf8::decode_at(text, offset);
            if (!cp.ok()) {
                return cp.status();
            }

            const std::string unknown = text.substr(offset, cp.value().size);
            if (strict) {
                return Status::error("Unknown symbol in strict tokenizer mode");
            }

            tokens.push_back(Token::non_rune(
                TokenKind::Unknown, offset, offset + cp.value().size, unknown));
            offset += cp.value().size;
        }

        return TokenStream::create(std::move(tokens));
    }

private:
    [[nodiscard]] std::optional<std::string> match_newline(
        const std::string& text,
        std::size_t offset) const {
        if (offset + 1 < text.size() && text[offset] == '\r' && text[offset + 1] == '\n' &&
            grammar_->is_newline("\r\n")) {
            return std::string("\r\n");
        }
        if (offset < text.size() && text[offset] == '\n' && grammar_->is_newline("\n")) {
            return std::string("\n");
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> match_whitespace(
        const std::string& text,
        std::size_t offset) const {
        if (offset >= text.size()) {
            return std::nullopt;
        }
        const std::string one(1, text[offset]);
        if (grammar_->is_whitespace(one)) {
            return one;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> match_separator(
        const std::string& text,
        std::size_t offset) const {
        for (const std::string& token : grammar_->separator_tokens_by_length_desc()) {
            if (offset + token.size() <= text.size() &&
                text.compare(offset, token.size(), token) == 0) {
                return token;
            }
        }
        return std::nullopt;
    }

    const GematriaProfile* profile_;
    const SeparatorGrammar* grammar_;
    RuneCodec rune_codec_;
};

#endif // TOKENIZER_HPP
