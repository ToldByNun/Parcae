#ifndef TOKEN_STREAM_HPP
#define TOKEN_STREAM_HPP

#include "parcae/corpus/consumable_mask.hpp"
#include "parcae/corpus/token.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class TokenStream {
public:
    TokenStream() = default;

    /// Build a stream, renumbering consumable rune indices to `0..n-1`.
    [[nodiscard]] static StatusOr<TokenStream> create(std::vector<Token> tokens) {
        Status status = validate_raw(tokens);
        if (!status.ok()) {
            return status;
        }

        TokenStream stream;
        stream.tokens_ = std::move(tokens);
        stream.renumber_consumable_indices();
        return stream;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return tokens_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return tokens_.empty();
    }

    [[nodiscard]] const Token& at(std::size_t i) const {
        return tokens_.at(i);
    }

    [[nodiscard]] const std::vector<Token>& tokens() const noexcept {
        return tokens_;
    }

    [[nodiscard]] std::size_t consumable_count() const noexcept {
        std::size_t count = 0;
        for (const Token& token : tokens_) {
            if (token.is_rune()) {
                ++count;
            }
        }
        return count;
    }

    /// Default mask: every consumable rune participates (`true`).
    [[nodiscard]] ConsumableMask consumable_mask() const {
        std::vector<Index29> indices;
        indices.reserve(consumable_count());
        for (const Token& token : tokens_) {
            if (token.is_rune()) {
                indices.push_back(*token.index29());
            }
        }
        return ConsumableMask::all_participating(std::move(indices));
    }

    [[nodiscard]] std::string text() const {
        std::string out;
        for (const Token& token : tokens_) {
            out += token.text();
        }
        return out;
    }

private:
    static Status validate_raw(const std::vector<Token>& tokens) {
        for (const Token& token : tokens) {
            if (token.is_rune()) {
                if (!token.index29().has_value()) {
                    return Status::error("Rune token missing Index29");
                }
                if (!TokenKindUtil::is_consumable(token.kind())) {
                    return Status::error("Inconsistent rune TokenKind");
                }
            } else {
                if (token.index29().has_value()) {
                    return Status::error("Non-rune token must not carry Index29");
                }
                if (token.consumable_index().has_value()) {
                    return Status::error("Non-rune token must not carry consumable_index");
                }
                if (TokenKindUtil::is_consumable(token.kind())) {
                    return Status::error("Consumable TokenKind requires rune payload");
                }
            }

            if (token.byte_end() < token.byte_begin()) {
                return Status::error("Token byte range is inverted");
            }
        }
        return Status::success();
    }

    void renumber_consumable_indices() {
        std::size_t next = 0;
        for (Token& token : tokens_) {
            if (token.is_rune()) {
                token.set_consumable_index(next);
                ++next;
            } else {
                token.set_consumable_index(std::nullopt);
            }
        }
    }

    std::vector<Token> tokens_;
};

#endif // TOKEN_STREAM_HPP
