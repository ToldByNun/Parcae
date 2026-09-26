#ifndef COLUMNAR_TRANSPOSITION_TRANSFORM_HPP
#define COLUMNAR_TRANSPOSITION_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/transform/grid_read_order.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// Columnar transposition over a `rows × cols` row-major stream.
///
/// Classical: write by rows, read columns sorted by `(key[col], col)`.
///
/// - `encrypt`: apply that columnar read order (`out[i] = in[order[i]]`)
/// - `decrypt`: inverse scatter
///
/// `cols` equals `key_indices.size()`. Input length MUST equal `rows * cols`.
/// Interrupt policy is ignored. Optional modular mix is via Compose, not this family.
class ColumnarTranspositionTransform : public Transform {
public:
    ColumnarTranspositionTransform() = default;

    [[nodiscard]] TransformId id() const override {
        return TransformId::columnar_transposition();
    }

    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       std::span<const std::size_t> gather,
                                       TransformDirection direction) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        // Classical encrypt = gather read; decrypt = inverse — flip vs spiral_read naming.
        const TransformDirection gather_dir = (direction == TransformDirection::Encrypt)
                                                  ? TransformDirection::Decrypt
                                                  : TransformDirection::Encrypt;
        return GridReadOrder::apply_gather(input, output, gather, gather_dir);
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
        const std::size_t cols = parsed.value().key.size();
        if (input.size() != parsed.value().rows * cols) {
            return Status::error("columnar_transposition input length must equal rows * cols");
        }

        std::vector<std::size_t> gather(input.size());
        Status order =
            GridReadOrder::fill_columnar(gather, parsed.value().rows, cols, parsed.value().key);
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
        std::vector<std::uint8_t> key;
    };

    [[nodiscard]] static StatusOr<ParsedParams> parse_params(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("columnar_transposition params must be an object");
        }
        if (!params.contains("rows")) {
            return Status::error("columnar_transposition params.rows is required");
        }
        if (!params.contains("key_indices")) {
            return Status::error("columnar_transposition params.key_indices is required");
        }
        if (!params.at("rows").is_number_integer()) {
            return Status::error("columnar_transposition rows must be an integer");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error("columnar_transposition key_indices must be an array");
        }

        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "rows" && it.key() != "key_indices" && it.key() != "key_latin") {
                return Status::error("columnar_transposition params contains unknown field");
            }
        }

        const auto rows_raw = params.at("rows").get<std::int64_t>();
        if (rows_raw < 1) {
            return Status::error("columnar_transposition rows must be >= 1");
        }

        ParsedParams out;
        out.rows = static_cast<std::size_t>(rows_raw);
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error(
                    "columnar_transposition key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error(
                    "columnar_transposition key_indices entries must be in 0..28");
            }
            out.key.push_back(static_cast<std::uint8_t>(raw));
        }
        if (out.key.empty()) {
            return Status::error("columnar_transposition key_indices must be non-empty");
        }
        return out;
    }
};

#endif // COLUMNAR_TRANSPOSITION_TRANSFORM_HPP
