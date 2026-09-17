#ifndef SCORE_CATALOG_ENTRY_HPP
#define SCORE_CATALOG_ENTRY_HPP

#include "parcae/score/score_order.hpp"

#include <string>
#include <string_view>
#include <utility>

/// Read-only catalog row for tool / agent listing of score_ids.
class ScoreCatalogEntry {
public:
    /// How many Index29 streams the score consumes besides optional tables.
    enum class Arity {
        Unary,           ///< candidate only
        Pairwise,        ///< candidate + reference
        UnaryWithTable,  ///< candidate + expected-frequency table
    };

    ScoreCatalogEntry(
        std::string id,
        std::string version,
        ScoreOrder order,
        Arity arity)
        : id_(std::move(id)),
          version_(std::move(version)),
          order_(order),
          arity_(arity) {}

    [[nodiscard]] const std::string& id() const noexcept {
        return id_;
    }

    [[nodiscard]] const std::string& version() const noexcept {
        return version_;
    }

    [[nodiscard]] ScoreOrder order() const noexcept {
        return order_;
    }

    [[nodiscard]] Arity arity() const noexcept {
        return arity_;
    }

    [[nodiscard]] static constexpr std::string_view arity_string(Arity arity) noexcept {
        switch (arity) {
        case Arity::Unary:
            return "unary";
        case Arity::Pairwise:
            return "pairwise";
        case Arity::UnaryWithTable:
            return "unary_with_table";
        }
        return "unary";
    }

private:
    std::string id_;
    std::string version_;
    ScoreOrder order_;
    Arity arity_;
};

#endif // SCORE_CATALOG_ENTRY_HPP
