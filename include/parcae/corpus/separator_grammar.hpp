#ifndef SEPARATOR_GRAMMAR_HPP
#define SEPARATOR_GRAMMAR_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/token_kind.hpp"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class SeparatorGrammar {
public:
    [[nodiscard]] static StatusOr<SeparatorGrammar> load_from_string(const std::string& json_text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(json_text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid separator grammar JSON: ") + ex.what());
        }
        return load_from_json(root);
    }

    [[nodiscard]] static StatusOr<SeparatorGrammar> load_from_file(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return Status::error("Failed to open separator grammar file: " + path);
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return load_from_string(buffer.str());
    }

    [[nodiscard]] const std::string& grammar_id() const noexcept { return grammar_id_; }

    [[nodiscard]] bool is_whitespace(const std::string& token) const {
        return whitespace_.find(token) != whitespace_.end();
    }

    [[nodiscard]] bool is_newline(const std::string& token) const {
        return newlines_.find(token) != newlines_.end();
    }

    [[nodiscard]] StatusOr<TokenKind> kind_for_token(const std::string& token) const {
        const auto it = token_to_kind_.find(token);
        if (it == token_to_kind_.end()) {
            return Status::error("Unknown separator token");
        }
        return it->second;
    }

    [[nodiscard]] const std::vector<std::string>& separator_tokens_by_length_desc() const noexcept {
        return tokens_by_length_desc_;
    }

private:
    SeparatorGrammar() = default;

    [[nodiscard]] static StatusOr<TokenKind> parse_token_kind(const std::string& name) {
        if (name == "WordSep") {
            return TokenKind::WordSep;
        }
        if (name == "ClauseSep") {
            return TokenKind::ClauseSep;
        }
        if (name == "ParaSep") {
            return TokenKind::ParaSep;
        }
        if (name == "LineSep") {
            return TokenKind::LineSep;
        }
        if (name == "PageMark") {
            return TokenKind::PageMark;
        }
        if (name == "ChapterSep") {
            return TokenKind::ChapterSep;
        }
        return Status::error("Unknown token_kind in separator grammar: " + name);
    }

    [[nodiscard]] static StatusOr<SeparatorGrammar> load_from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("Separator grammar root must be an object");
        }
        if (!root.contains("schema") || root.at("schema") != "parcae.separator_grammar.v0") {
            return Status::error("Separator grammar schema must be parcae.separator_grammar.v0");
        }
        if (!root.contains("grammar_id") || !root.at("grammar_id").is_string()) {
            return Status::error("Separator grammar_id must be a string");
        }

        SeparatorGrammar grammar;
        grammar.grammar_id_ = root.at("grammar_id").get<std::string>();

        if (!root.contains("separators") || !root.at("separators").is_array()) {
            return Status::error("Separator grammar separators must be an array");
        }

        for (const nlohmann::json& item : root.at("separators")) {
            if (!item.contains("token") || !item.contains("token_kind")) {
                return Status::error("Separator entry requires token and token_kind");
            }
            const std::string token = item.at("token").get<std::string>();
            StatusOr<TokenKind> kind = parse_token_kind(item.at("token_kind").get<std::string>());
            if (!kind.ok()) {
                return kind.status();
            }
            grammar.token_to_kind_.emplace(token, kind.value());
            grammar.tokens_by_length_desc_.push_back(token);
        }

        std::sort(grammar.tokens_by_length_desc_.begin(), grammar.tokens_by_length_desc_.end(),
                  [](const std::string& left, const std::string& right) {
                      if (left.size() != right.size()) {
                          return left.size() > right.size();
                      }
                      return left < right;
                  });

        if (root.contains("whitespace") && root.at("whitespace").is_array()) {
            for (const nlohmann::json& item : root.at("whitespace")) {
                grammar.whitespace_.insert(item.get<std::string>());
            }
        }
        if (root.contains("newline") && root.at("newline").is_array()) {
            for (const nlohmann::json& item : root.at("newline")) {
                grammar.newlines_.insert(item.get<std::string>());
            }
        }

        return grammar;
    }

    std::string grammar_id_;
    std::unordered_map<std::string, TokenKind> token_to_kind_;
    std::vector<std::string> tokens_by_length_desc_;
    std::unordered_set<std::string> whitespace_;
    std::unordered_set<std::string> newlines_;
};

#endif // SEPARATOR_GRAMMAR_HPP
