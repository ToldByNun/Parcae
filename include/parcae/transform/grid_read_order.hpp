#ifndef GRID_READ_ORDER_HPP
#define GRID_READ_ORDER_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <span>

/// Deterministic row-major grid visit orders for transposition-style reads.
/// Index `r * cols + c` is the row-major storage slot.
class GridReadOrder {
public:
    /// Spiral clockwise from top-left, inward. `order.size()` MUST equal `rows * cols`.
    [[nodiscard]] static Status fill_spiral_inward(std::span<std::size_t> order, std::size_t rows,
                                                   std::size_t cols) {
        if (rows == 0 || cols == 0) {
            return Status::error("grid rows and cols must be positive");
        }
        if (order.size() != rows * cols) {
            return Status::error("grid order length must equal rows * cols");
        }

        const std::ptrdiff_t R = static_cast<std::ptrdiff_t>(rows);
        const std::ptrdiff_t C = static_cast<std::ptrdiff_t>(cols);
        std::ptrdiff_t top = 0;
        std::ptrdiff_t bottom = R - 1;
        std::ptrdiff_t left = 0;
        std::ptrdiff_t right = C - 1;
        std::size_t n = 0;

        while (top <= bottom && left <= right) {
            for (std::ptrdiff_t c = left; c <= right; ++c) {
                order[n++] = static_cast<std::size_t>(top) * cols + static_cast<std::size_t>(c);
            }
            ++top;
            for (std::ptrdiff_t r = top; r <= bottom; ++r) {
                order[n++] = static_cast<std::size_t>(r) * cols + static_cast<std::size_t>(right);
            }
            --right;
            if (top <= bottom) {
                for (std::ptrdiff_t c = right; c >= left; --c) {
                    order[n++] =
                        static_cast<std::size_t>(bottom) * cols + static_cast<std::size_t>(c);
                }
                --bottom;
            }
            if (left <= right) {
                for (std::ptrdiff_t r = bottom; r >= top; --r) {
                    order[n++] =
                        static_cast<std::size_t>(r) * cols + static_cast<std::size_t>(left);
                }
                ++left;
            }
        }

        if (n != order.size()) {
            return Status::error("spiral order generation incomplete");
        }
        return Status::success();
    }

    /// Outward spiral = reverse of inward clockwise from top-left.
    [[nodiscard]] static Status fill_spiral_outward(std::span<std::size_t> order, std::size_t rows,
                                                    std::size_t cols) {
        Status inward = fill_spiral_inward(order, rows, cols);
        if (!inward.ok()) {
            return inward;
        }
        for (std::size_t i = 0; i < order.size() / 2; ++i) {
            const std::size_t j = order.size() - 1 - i;
            const std::size_t tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        return Status::success();
    }

    /// Boustrophedon: alternating row directions.
    /// `first_row_ltr`: row 0 left→right when true; otherwise right→left.
    [[nodiscard]] static Status fill_boustrophedon(std::span<std::size_t> order, std::size_t rows,
                                                   std::size_t cols, bool first_row_ltr) {
        if (rows == 0 || cols == 0) {
            return Status::error("grid rows and cols must be positive");
        }
        if (order.size() != rows * cols) {
            return Status::error("grid order length must equal rows * cols");
        }

        std::size_t n = 0;
        for (std::size_t r = 0; r < rows; ++r) {
            const bool ltr = ((r % 2u) == 0u) == first_row_ltr;
            if (ltr) {
                for (std::size_t c = 0; c < cols; ++c) {
                    order[n++] = r * cols + c;
                }
            } else {
                for (std::size_t c = cols; c-- > 0;) {
                    order[n++] = r * cols + c;
                }
            }
        }
        return Status::success();
    }

    /// Decrypt/read: `out[i] = in[gather[i]]`. Encrypt: inverse scatter.
    /// Not in-place safe (`input.data()` MUST differ from `output.data()`).
    [[nodiscard]] static Status apply_gather(std::span<const Index29> input,
                                             std::span<Index29> output,
                                             std::span<const std::size_t> gather,
                                             TransformDirection direction) {
        if (input.size() != output.size() || input.size() != gather.size()) {
            return Status::error("grid gather size mismatch");
        }
        if (input.data() == output.data()) {
            return Status::error("grid gather does not support in-place");
        }

        if (direction == TransformDirection::Decrypt) {
            for (std::size_t i = 0; i < gather.size(); ++i) {
                const std::size_t src = gather[i];
                if (src >= input.size()) {
                    return Status::error("grid gather index out of range");
                }
                output[i] = input[src];
            }
            return Status::success();
        }

        for (std::size_t i = 0; i < gather.size(); ++i) {
            const std::size_t dst = gather[i];
            if (dst >= output.size()) {
                return Status::error("grid gather index out of range");
            }
            output[dst] = input[i];
        }
        return Status::success();
    }

private:
    GridReadOrder() = delete;
};

#endif // GRID_READ_ORDER_HPP
