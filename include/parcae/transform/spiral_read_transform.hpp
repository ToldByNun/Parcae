#ifndef SPIRAL_READ_TRANSFORM_HPP
#define SPIRAL_READ_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/transform/grid_read_order.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// Spiral grid read over a `rows × cols` row-major stream (pure index permutation).
///
/// - `decrypt`: treat input as row-major grid; emit spiral visit order
/// - `encrypt`: inverse permutation (scatter spiral stream back to row-major)
///
/// Spiral is clockwise from top-left; `spiral` param selects `inward` (default) or
/// `outward` (reverse of inward). Interrupt policy is ignored.
class SpiralReadTransform : public Transform {
public:
    SpiralReadTransform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::spiral_read(); }

    /// `gather` is spiral visit order into row-major slots. Not in-place safe.
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
            return Status::error("spiral_read input length must equal rows * cols");
        }

        std::vector<std::size_t> gather(input.size());
        Status order = parsed.value().outward
                           ? GridReadOrder::fill_spiral_outward(gather, parsed.value().rows,
                                                               parsed.value().cols)
                           : GridReadOrder::fill_spiral_inward(gather, parsed.value().rows,
                                                              parsed.value().cols);
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
        bool outward = false;
    };

    [[nodiscard]] static StatusOr<std::size_t> parse_dim(const nlohmann::json& params,
                                                         const char* key) {
        if (!params.contains(key)) {
            return Status::error(std::string("spiral_read params.") + key + " is required");
        }
        if (!params.at(key).is_number_integer()) {
            return Status::error(std::string("spiral_read params.") + key + " must be an integer");
        }
        const auto raw = params.at(key).get<std::int64_t>();
        if (raw < 1) {
            return Status::error(std::string("spiral_read params.") + key + " must be >= 1");
        }
        return static_cast<std::size_t>(raw);
    }

    [[nodiscard]] static StatusOr<ParsedParams> parse_params(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("spiral_read params must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "rows" && it.key() != "cols" && it.key() != "spiral") {
                return Status::error("spiral_read params contains unknown field");
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
        out.outward = false;
        if (params.contains("spiral")) {
            if (!params.at("spiral").is_string()) {
                return Status::error("spiral_read spiral must be a string");
            }
            const std::string spiral = params.at("spiral").get<std::string>();
            if (spiral == "inward") {
                out.outward = false;
            } else if (spiral == "outward") {
                out.outward = true;
            } else {
                return Status::error("spiral_read spiral must be inward or outward");
            }
        }
        return out;
    }
};

#endif // SPIRAL_READ_TRANSFORM_HPP
