#ifndef GENERATOR_CATALOG_ENTRY_HPP
#define GENERATOR_CATALOG_ENTRY_HPP

#include <cstddef>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

/// Read-only catalog row for tool / agent listing of generator_ids.
class GeneratorCatalogEntry {
public:
    GeneratorCatalogEntry(
        std::string id,
        std::string transform_id,
        std::size_t bounded_count,
        bool requires_params)
        : id_(std::move(id)),
          transform_id_(std::move(transform_id)),
          bounded_count_(bounded_count),
          requires_params_(requires_params) {}

    [[nodiscard]] const std::string& id() const noexcept {
        return id_;
    }

    [[nodiscard]] const std::string& transform_id() const noexcept {
        return transform_id_;
    }

    /// Fixed enumeration size, or `0` when the caller supplies the bound (e.g. keys).
    [[nodiscard]] std::size_t bounded_count() const noexcept {
        return bounded_count_;
    }

    /// True when `generate` needs a non-empty params object (e.g. explicit keys).
    [[nodiscard]] bool requires_params() const noexcept {
        return requires_params_;
    }

    [[nodiscard]] nlohmann::json to_json() const {
        return nlohmann::json{
            {"generator_id", id_},
            {"transform_id", transform_id_},
            {"bounded_count", bounded_count_},
            {"requires_params", requires_params_},
        };
    }

private:
    std::string id_;
    std::string transform_id_;
    std::size_t bounded_count_ = 0;
    bool requires_params_ = false;
};

#endif  // GENERATOR_CATALOG_ENTRY_HPP
