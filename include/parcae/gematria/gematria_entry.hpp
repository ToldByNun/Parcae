#ifndef GEMATRIA_ENTRY_HPP
#define GEMATRIA_ENTRY_HPP

#include "parcae/core/index29.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class GematriaEntry {
public:
    GematriaEntry(
        Index29 index,
        std::uint32_t prime,
        std::string rune,
        std::string preferred,
        std::vector<std::string> labels)
        : index_(index),
          prime_(prime),
          rune_(std::move(rune)),
          preferred_(std::move(preferred)),
          labels_(std::move(labels)) {}

    [[nodiscard]] Index29 index() const noexcept {
        return index_;
    }

    [[nodiscard]] std::uint32_t prime() const noexcept {
        return prime_;
    }

    [[nodiscard]] const std::string& rune() const noexcept {
        return rune_;
    }

    [[nodiscard]] const std::string& preferred() const noexcept {
        return preferred_;
    }

    [[nodiscard]] const std::vector<std::string>& labels() const noexcept {
        return labels_;
    }

private:
    Index29 index_;
    std::uint32_t prime_;
    std::string rune_;
    std::string preferred_;
    std::vector<std::string> labels_;
};

#endif // GEMATRIA_ENTRY_HPP
