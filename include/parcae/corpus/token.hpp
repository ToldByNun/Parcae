#ifndef TOKEN_HPP
#define TOKEN_HPP

#include "parcae/core/index29.hpp"
#include "parcae/corpus/token_kind.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

class Token {
public:
    [[nodiscard]] static Token rune(
        Index29 index,
        std::size_t byte_begin,
        std::size_t byte_end,
        std::string text,
        std::size_t consumable_index) {
        return Token{
            TokenKind::Rune,
            index,
            byte_begin,
            byte_end,
            std::move(text),
            consumable_index,
        };
    }

    [[nodiscard]] static Token non_rune(
        TokenKind kind,
        std::size_t byte_begin,
        std::size_t byte_end,
        std::string text) {
        return Token{
            kind,
            std::nullopt,
            byte_begin,
            byte_end,
            std::move(text),
            std::nullopt,
        };
    }

    [[nodiscard]] TokenKind kind() const noexcept {
        return kind_;
    }

    [[nodiscard]] bool is_rune() const noexcept {
        return kind_ == TokenKind::Rune;
    }

    [[nodiscard]] const std::optional<Index29>& index29() const noexcept {
        return index29_;
    }

    [[nodiscard]] std::size_t byte_begin() const noexcept {
        return byte_begin_;
    }

    [[nodiscard]] std::size_t byte_end() const noexcept {
        return byte_end_;
    }

    [[nodiscard]] const std::string& text() const noexcept {
        return text_;
    }

    [[nodiscard]] const std::optional<std::size_t>& consumable_index() const noexcept {
        return consumable_index_;
    }

private:
    friend class TokenStream;

    Token(
        TokenKind kind,
        std::optional<Index29> index29,
        std::size_t byte_begin,
        std::size_t byte_end,
        std::string text,
        std::optional<std::size_t> consumable_index)
        : kind_(kind),
          index29_(index29),
          byte_begin_(byte_begin),
          byte_end_(byte_end),
          text_(std::move(text)),
          consumable_index_(consumable_index) {}

    void set_consumable_index(std::optional<std::size_t> consumable_index) {
        consumable_index_ = consumable_index;
    }

    TokenKind kind_;
    std::optional<Index29> index29_;
    std::size_t byte_begin_;
    std::size_t byte_end_;
    std::string text_;
    std::optional<std::size_t> consumable_index_;
};

#endif // TOKEN_HPP
