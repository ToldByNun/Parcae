#ifndef MATRIX729_MAP_HPP
#define MATRIX729_MAP_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

/// Community "729 matrix" helpers: \(27\times 27 = 729\) cell permutations, plus
/// full-alphabet \(29\times 29\) (one image per Index29) permutations / projections.
/// Dense matrix invert is intentionally out of scope — only bijections.
class Matrix729Map {
public:
    static constexpr std::size_t side_27 = 27;
    static constexpr std::size_t cells_729 = side_27 * side_27; // 729
    static constexpr std::size_t side_29 = Index29::modulus;     // 29

    using Perm729 = std::array<std::uint16_t, cells_729>;
    using Perm29 = std::array<std::uint8_t, side_29>;

    // --- cell indexing for the 27×27 square ---

    [[nodiscard]] static constexpr std::size_t cell_index(std::size_t row,
                                                         std::size_t col) noexcept {
        return row * side_27 + col;
    }

    [[nodiscard]] static constexpr std::pair<std::size_t, std::size_t>
    cell_rc(std::size_t index) noexcept {
        return {index / side_27, index % side_27};
    }

    // --- 729-permutation (positions 0..728) ---

    [[nodiscard]] static Perm729 identity_729() noexcept {
        Perm729 perm{};
        for (std::uint16_t i = 0; i < static_cast<std::uint16_t>(cells_729); ++i) {
            perm[i] = i;
        }
        return perm;
    }

    /// Transpose map: cell (r,c) → (c,r).
    [[nodiscard]] static Perm729 transpose_729() noexcept {
        Perm729 perm{};
        for (std::size_t r = 0; r < side_27; ++r) {
            for (std::size_t c = 0; c < side_27; ++c) {
                perm[cell_index(r, c)] = static_cast<std::uint16_t>(cell_index(c, r));
            }
        }
        return perm;
    }

    /// Validate a bijection on `{0..728}`; reject duplicates / out-of-range.
    [[nodiscard]] static StatusOr<Perm729> from_perm_729(std::span<const std::uint16_t> values) {
        if (values.size() != cells_729) {
            return Status::error("Matrix729Map::from_perm_729 requires exactly 729 entries");
        }
        Perm729 perm{};
        std::array<bool, cells_729> seen{};
        for (std::size_t i = 0; i < cells_729; ++i) {
            const std::uint16_t v = values[i];
            if (v >= cells_729) {
                return Status::error("Matrix729Map::from_perm_729 entry out of range 0..728");
            }
            if (seen[v]) {
                return Status::error("Matrix729Map::from_perm_729 is not a bijection");
            }
            seen[v] = true;
            perm[i] = v;
        }
        return perm;
    }

    [[nodiscard]] static StatusOr<Perm729> try_inverse_729(const Perm729& perm) {
        Perm729 inv{};
        std::array<bool, cells_729> seen{};
        for (std::size_t i = 0; i < cells_729; ++i) {
            const std::uint16_t v = perm[i];
            if (v >= cells_729 || seen[v]) {
                return Status::error("Matrix729Map::try_inverse_729: not a bijection");
            }
            seen[v] = true;
            inv[v] = static_cast<std::uint16_t>(i);
        }
        return inv;
    }

    /// Gather reorder: `out[i] = in[perm[i]]`. Length MUST be 729.
    template <typename T>
    [[nodiscard]] static Status apply_gather_729(const Perm729& perm, std::span<const T> input,
                                                 std::span<T> output) {
        if (input.size() != cells_729 || output.size() != cells_729) {
            return Status::error("Matrix729Map::apply_gather_729 requires length 729");
        }
        // Support in-place via scratch when buffers alias.
        if (input.data() == output.data()) {
            std::array<T, cells_729> scratch{};
            for (std::size_t i = 0; i < cells_729; ++i) {
                scratch[i] = input[perm[i]];
            }
            for (std::size_t i = 0; i < cells_729; ++i) {
                output[i] = scratch[i];
            }
            return Status::success();
        }
        for (std::size_t i = 0; i < cells_729; ++i) {
            output[i] = input[perm[i]];
        }
        return Status::success();
    }

    // --- 29-permutation (Index29 → Index29) ---

    [[nodiscard]] static Perm29 identity_29() noexcept {
        Perm29 perm{};
        for (std::uint8_t i = 0; i < side_29; ++i) {
            perm[i] = i;
        }
        return perm;
    }

    [[nodiscard]] static StatusOr<Perm29> from_perm_29(std::span<const std::uint8_t> values) {
        if (values.size() != side_29) {
            return Status::error("Matrix729Map::from_perm_29 requires exactly 29 entries");
        }
        Perm29 perm{};
        std::array<bool, side_29> seen{};
        for (std::size_t i = 0; i < side_29; ++i) {
            const std::uint8_t v = values[i];
            if (v >= side_29) {
                return Status::error("Matrix729Map::from_perm_29 entry out of range 0..28");
            }
            if (seen[v]) {
                return Status::error("Matrix729Map::from_perm_29 is not a bijection");
            }
            seen[v] = true;
            perm[i] = v;
        }
        return perm;
    }

    [[nodiscard]] static StatusOr<Perm29> try_inverse_29(const Perm29& perm) {
        Perm29 inv{};
        std::array<bool, side_29> seen{};
        for (std::size_t i = 0; i < side_29; ++i) {
            const std::uint8_t v = perm[i];
            if (v >= side_29 || seen[v]) {
                return Status::error("Matrix729Map::try_inverse_29: not a bijection");
            }
            seen[v] = true;
            inv[v] = static_cast<std::uint8_t>(i);
        }
        return inv;
    }

    [[nodiscard]] static Index29 map_29(const Perm29& perm, Index29 x) noexcept {
        return Index29{perm[x.value()]};
    }

    /// Elementwise alphabet remap: `out[i] = perm[in[i]]`.
    [[nodiscard]] static Status apply_map_29(const Perm29& perm, std::span<const Index29> input,
                                             std::span<Index29> output) {
        if (input.size() != output.size()) {
            return Status::error("Matrix729Map::apply_map_29 size mismatch");
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = map_29(perm, input[i]);
        }
        return Status::success();
    }

    // --- projection between Index29 and a 27-symbol subset ---

    /// Map `x` into `{0..26}` by skipping two excluded alphabet indices (`ea != eb`).
    /// Values equal to an excluded index fail.
    [[nodiscard]] static StatusOr<std::uint8_t> project_to_27(Index29 x, Index29 ea, Index29 eb) {
        if (ea.value() == eb.value()) {
            return Status::error("Matrix729Map::project_to_27 excluded indices must differ");
        }
        const std::uint8_t v = x.value();
        if (v == ea.value() || v == eb.value()) {
            return Status::error("Matrix729Map::project_to_27 value is excluded");
        }
        std::uint8_t rank = v;
        if (v > ea.value()) {
            --rank;
        }
        if (v > eb.value()) {
            --rank;
        }
        // After removing two indices from 0..28, rank is in 0..26.
        return rank;
    }

    /// Inverse of `project_to_27`: lift rank `0..26` back to Index29 given the same exclusions.
    [[nodiscard]] static StatusOr<Index29> lift_from_27(std::uint8_t rank, Index29 ea, Index29 eb) {
        if (ea.value() == eb.value()) {
            return Status::error("Matrix729Map::lift_from_27 excluded indices must differ");
        }
        if (rank >= side_27) {
            return Status::error("Matrix729Map::lift_from_27 rank must be in 0..26");
        }
        std::uint8_t seen = 0;
        for (std::uint8_t v = 0; v < side_29; ++v) {
            if (v == ea.value() || v == eb.value()) {
                continue;
            }
            if (seen == rank) {
                return Index29{v};
            }
            ++seen;
        }
        return Status::error("Matrix729Map::lift_from_27 failed to lift rank");
    }

private:
    Matrix729Map() = delete;
};

#endif // MATRIX729_MAP_HPP
