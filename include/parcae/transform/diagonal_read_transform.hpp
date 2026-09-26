#ifndef DIAGONAL_READ_TRANSFORM_HPP
#define DIAGONAL_READ_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/transform/grid_read_order.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// Diagonal grid read over a `rows × cols` row-major stream (pure index permutation).
///
/// - `decrypt`: emit diagonal visit order from row-major input
/// - `encrypt`: inverse permutation
///
/// `anti` (default false): main diagonals (constant `r+c`); true → anti-diagonals
/// (column-mirrored). Interrupt policy is ignored.
class DiagonalReadTransform : public Transform {
public:
    DiagonalReadTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::diagonal_read(); }

    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       std::span<const std::size_t> gather,
                                       TransformDirection direction) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        return GridReadOrder::apply_gather(input, output, gather, direction);
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& /*interrupt*/ = InterruptPolicy::none()) const override {
        StatusOr<ParsedParams> parsed = parse_params(params);
        if (!parsed.ok()) {
            return parsed.status();
        }
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        if (input.size() != parsed.value().rows * parsed.value().cols) {
            return Status::error("diagonal_read input length must equal rows * cols");
        }

        std::vector<std::size_t> gather(input.size());
        Status order = GridReadOrder::fill_diagonal(gather, parsed.value().rows,
                                                    parsed.value().cols, parsed.value().anti);
        if (!order.ok()) {
            return order;
        }

        if (input.data() == output.data()) {
            std::vector<Index29> tmp(input.begin(), input.end());
            return kernel(tmp, output, gather, direction);
        }
        return kernel(input, output, gather, direction);
    }

private:
    struct ParsedParams {
        std::size_t rows = 0;
        std::size_t cols = 0;
        bool anti = false;
    };

    [[nodiscard]] static StatusOr<std::size_t> parse_dim(const nlohmann::json& params,
                                                         const char* key) {
        if (!params.contains(key)) {
            return Status::error(std::string("diagonal_read params.") + key + " is required");
        }
        if (!params.at(key).is_number_integer()) {
            return Status::error(std::string("diagonal_read params.") + key +
                                 " must be an integer");
        }
        const auto raw = params.at(key).get<std::int64_t>();
        if (raw < 1) {
            return Status::error(std::string("diagonal_read params.") + key + " must be >= 1");
        }
        return static_cast<std::size_t>(raw);
    }

    [[nodiscard]] static StatusOr<ParsedParams> parse_params(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("diagonal_read params must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "rows" && it.key() != "cols" && it.key() != "anti") {
                return Status::error("diagonal_read params contains unknown field");
            }
        }

        StatusOr<std::size_t> rows = parse_dim(params, "rows");
        if (!rows.ok()) {
            return rows.status();
        }
        StatusOr<std::size_t> cols = parse_dim(params, "cols");
        if (!cols.ok()) {
            return cols.status();
        }

        ParsedParams out;
        out.rows = rows.value();
        out.cols = cols.value();
        out.anti = false;
        if (params.contains("anti")) {
            if (!params.at("anti").is_boolean()) {
                return Status::error("diagonal_read anti must be a boolean");
            }
            out.anti = params.at("anti").get<bool>();
        }
        return out;
    }
};

#endif // DIAGONAL_READ_TRANSFORM_HPP
