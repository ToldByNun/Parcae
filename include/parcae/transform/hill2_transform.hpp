#ifndef HILL2_TRANSFORM_HPP
#define HILL2_TRANSFORM_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/math/z29_matrix2.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

/// Hill cipher over \(\mathbb{Z}_{29}\) with a fixed invertible \(2 \times 2\) key matrix.
/// Processes the stream in consecutive pairs; length MUST be even.
/// Encrypt: \(A\cdot v\); decrypt: \(A^{-1}\cdot v\). Interrupt policy is ignored (block cipher).
class Hill2Transform : public Transform {
public:
    Hill2Transform() = default;

    [[nodiscard]] TransformId id() const override { return TransformId::hill_2(); }

    /// Allocation-free block kernel (in-place OK). `matrix` MUST be invertible.
    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       const Z29Matrix2& matrix, TransformDirection direction) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        if (input.size() % 2u != 0) {
            return Status::error("hill_2 input length must be even");
        }

        Z29Matrix2 apply = matrix;
        if (direction == TransformDirection::Decrypt) {
            StatusOr<Z29Matrix2> inv = matrix.try_inverse();
            if (!inv.ok()) {
                return inv.status();
            }
            apply = inv.value();
        } else {
            // Encrypt still requires an invertible key so decrypt is defined.
            StatusOr<Z29Matrix2> inv = matrix.try_inverse();
            if (!inv.ok()) {
                return Status::error("hill_2 matrix must be invertible (det ≢ 0 mod 29)");
            }
        }

        for (std::size_t i = 0; i < input.size(); i += 2) {
            const std::array<Index29, 2> out = apply.mul_vec(input[i], input[i + 1]);
            output[i] = out[0];
            output[i + 1] = out[1];
        }
        return Status::success();
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& /*interrupt*/ = InterruptPolicy::none()) const override {
        StatusOr<Z29Matrix2> matrix = parse_matrix(params);
        if (!matrix.ok()) {
            return matrix.status();
        }
        return kernel(input, output, matrix.value(), direction);
    }

private:
    [[nodiscard]] static StatusOr<Index29> parse_entry(const nlohmann::json& item,
                                                       std::size_t index) {
        if (!item.is_number_integer()) {
            return Status::error("hill_2 params.matrix[" + std::to_string(index) +
                                 "] must be an integer");
        }
        const auto raw = item.get<std::int64_t>();
        if (raw < 0 || raw > 28) {
            return Status::error("hill_2 params.matrix[" + std::to_string(index) +
                                 "] must be in 0..28");
        }
        return Index29{static_cast<std::uint8_t>(raw)};
    }

    [[nodiscard]] static StatusOr<Z29Matrix2> parse_matrix(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("hill_2 params must be an object");
        }
        if (!params.contains("matrix")) {
            return Status::error("hill_2 params.matrix is required");
        }
        if (!params.at("matrix").is_array()) {
            return Status::error("hill_2 params.matrix must be an array of 4 integers");
        }
        if (params.at("matrix").size() != 4) {
            return Status::error("hill_2 params.matrix must contain exactly 4 entries");
        }
        if (params.size() != 1) {
            return Status::error("hill_2 params may only contain matrix");
        }

        std::array<Index29, 4> entries{};
        for (std::size_t i = 0; i < 4; ++i) {
            StatusOr<Index29> entry = parse_entry(params.at("matrix").at(i), i);
            if (!entry.ok()) {
                return entry.status();
            }
            entries[i] = entry.value();
        }

        const Z29Matrix2 matrix = Z29Matrix2::from_row_major(entries);
        if (matrix.det().value() == 0) {
            return Status::error("hill_2 matrix must be invertible (det ≢ 0 mod 29)");
        }
        return matrix;
    }
};

#endif // HILL2_TRANSFORM_HPP
