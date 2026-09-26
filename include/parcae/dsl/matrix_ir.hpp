#ifndef MATRIX_IR_HPP
#define MATRIX_IR_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/z29_expr.hpp"

#include <cstddef>
#include <optional>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

/// n×n matrix of `Z29Expr` (row-major) for DSL `z29_det` / `z29_matmul`.
///
/// v0 supports n = 2 and n = 3 (Hill-2 / Hill-3). `det_expr()` / `mul_vec_exprs()`
/// expand to pure scalar `Z29Expr` trees so verify + CPU emit reuse existing paths.
class MatrixIr {
public:
    [[nodiscard]] std::size_t n() const noexcept { return n_; }

    [[nodiscard]] const std::vector<Z29Expr::Ptr>& entries() const noexcept { return entries_; }

    [[nodiscard]] Z29Expr::Ptr at(std::size_t row, std::size_t col) const {
        return entries_[row * n_ + col];
    }

    [[nodiscard]] static StatusOr<MatrixIr> make(std::vector<Z29Expr::Ptr> entries,
                                                std::string source_path = {},
                                                std::optional<int> lineno = std::nullopt,
                                                std::optional<int> col = std::nullopt) {
        const std::size_t count = entries.size();
        std::size_t n = 0;
        if (count == 4) {
            n = 2;
        } else if (count == 9) {
            n = 3;
        } else {
            return fail(DslRuleId::E032_primitive_body,
                        "matrix must have 4 (2x2) or 9 (3x3) entries, got " +
                            std::to_string(count),
                        std::move(source_path), lineno, col);
        }
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if (!entries[i]) {
                return fail(DslRuleId::E032_primitive_body,
                            "matrix entry " + std::to_string(i) + " is null",
                            std::move(source_path), lineno, col);
            }
        }
        return MatrixIr{n, std::move(entries)};
    }

    /// det for 2×2: ad − bc; for 3×3: first-row cofactor expansion (same as Z29Matrix3).
    [[nodiscard]] Z29Expr::Ptr det_expr() const {
        if (n_ == 2) {
            const Z29Expr::Ptr a = at(0, 0);
            const Z29Expr::Ptr b = at(0, 1);
            const Z29Expr::Ptr c = at(1, 0);
            const Z29Expr::Ptr d = at(1, 1);
            return Z29Expr::sub(Z29Expr::mul(a, d), Z29Expr::mul(b, c));
        }
        // 3×3
        const Z29Expr::Ptr a = at(0, 0);
        const Z29Expr::Ptr b = at(0, 1);
        const Z29Expr::Ptr c = at(0, 2);
        const Z29Expr::Ptr d = at(1, 0);
        const Z29Expr::Ptr e = at(1, 1);
        const Z29Expr::Ptr f = at(1, 2);
        const Z29Expr::Ptr g = at(2, 0);
        const Z29Expr::Ptr h = at(2, 1);
        const Z29Expr::Ptr i = at(2, 2);
        const Z29Expr::Ptr t0 = minor2(e, f, h, i);
        const Z29Expr::Ptr t1 = minor2(d, f, g, i);
        const Z29Expr::Ptr t2 = minor2(d, e, g, h);
        return Z29Expr::add(Z29Expr::sub(Z29Expr::mul(a, t0), Z29Expr::mul(b, t1)),
                            Z29Expr::mul(c, t2));
    }

    /// `out[r] = sum_c M[r,c] * v[c]`.
    [[nodiscard]] StatusOr<std::vector<Z29Expr::Ptr>>
    mul_vec_exprs(const std::vector<Z29Expr::Ptr>& vec) const {
        if (vec.size() != n_) {
            return Status::error("MatrixIr::mul_vec_exprs: vector length must equal n (" +
                                 std::to_string(n_) + ")");
        }
        for (const Z29Expr::Ptr& e : vec) {
            if (!e) {
                return Status::error("MatrixIr::mul_vec_exprs: null vector entry");
            }
        }
        std::vector<Z29Expr::Ptr> out;
        out.reserve(n_);
        for (std::size_t r = 0; r < n_; ++r) {
            Z29Expr::Ptr acc = Z29Expr::mul(at(r, 0), vec[0]);
            for (std::size_t c = 1; c < n_; ++c) {
                acc = Z29Expr::add(std::move(acc), Z29Expr::mul(at(r, c), vec[c]));
            }
            out.push_back(std::move(acc));
        }
        return out;
    }

    /// One output component of `M · v` (for `z29_matmul(...)[i]`).
    [[nodiscard]] StatusOr<Z29Expr::Ptr> mul_vec_component(const std::vector<Z29Expr::Ptr>& vec,
                                                           std::size_t index) const {
        StatusOr<std::vector<Z29Expr::Ptr>> all = mul_vec_exprs(vec);
        if (!all.ok()) {
            return all.status();
        }
        if (index >= all.value().size()) {
            return Status::error("MatrixIr::mul_vec_component: index out of range");
        }
        return all.value()[index];
    }

private:
    MatrixIr(std::size_t n, std::vector<Z29Expr::Ptr> entries)
        : n_(n), entries_(std::move(entries)) {}

    [[nodiscard]] static Z29Expr::Ptr minor2(const Z29Expr::Ptr& a, const Z29Expr::Ptr& b,
                                             const Z29Expr::Ptr& c, const Z29Expr::Ptr& d) {
        return Z29Expr::sub(Z29Expr::mul(a, d), Z29Expr::mul(b, c));
    }

    [[nodiscard]] static Status fail(std::string_view rule, std::string message,
                                     std::string path, std::optional<int> lineno,
                                     std::optional<int> col) {
        return DslDiag::make(rule, std::move(message), std::move(path), lineno, col).to_status();
    }

    std::size_t n_ = 0;
    std::vector<Z29Expr::Ptr> entries_;
};

#endif // MATRIX_IR_HPP