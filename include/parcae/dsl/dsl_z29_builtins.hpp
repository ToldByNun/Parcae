#ifndef DSL_Z29_BUILTINS_HPP
#define DSL_Z29_BUILTINS_HPP

#include <cstddef>
#include <optional>
#include <string_view>

/// Allowlisted `z29_*` intrinsic call names for the theory DSL.
///
/// Used by `DslSemanticGate` (unknown `z29_*` -> E032) and as the catalog for
/// later BuildIr / emit / fuse wiring. Custom `@define_primitive` names are
/// *not* listed here (they do not use the `z29_` prefix convention for
/// intrinsics; gate only constrains names that start with `z29_`).
class DslZ29Builtins {
public:
    /// Expected positional arity, or nullopt if the name is not a builtin.
    [[nodiscard]] static std::optional<std::size_t> arity(std::string_view name) noexcept {
        // Scalar core (docs/spec/dsl.md)
        if (name == "z29_add" || name == "z29_sub" || name == "z29_mul" || name == "z29_div" ||
            name == "z29_floordiv" || name == "z29_mod" || name == "z29_pow" ||
            name == "z29_bit_and" || name == "z29_bit_or" || name == "z29_bit_xor" ||
            name == "z29_lshift" || name == "z29_rshift" || name == "z29_eq" || name == "z29_ne" ||
            name == "z29_lt" || name == "z29_le" || name == "z29_gt" || name == "z29_ge" ||
            name == "z29_bool_and" || name == "z29_bool_or") {
            return 2;
        }
        if (name == "z29_inv" || name == "z29_neg" || name == "z29_atbash" ||
            name == "z29_bit_not" || name == "z29_bool_not") {
            return 1;
        }
        if (name == "z29_select") {
            return 3;
        }
        // Matrix / stream intrinsics (Hill + autokey; BuildIr / emit / fuse wired)
        if (name == "z29_matmul") {
            return 2; // (matrix, rune_vec)
        }
        if (name == "z29_det") {
            return 1; // (matrix)
        }
        if (name == "z29_autokey_shift") {
            return 2; // (stream, lag)
        }
        return std::nullopt;
    }

    [[nodiscard]] static bool is_builtin(std::string_view name) noexcept {
        return arity(name).has_value();
    }

    /// Names beginning with `z29_` MUST be builtins; others are unconstrained here.
    [[nodiscard]] static bool is_allowed_call_name(std::string_view name) noexcept {
        if (name.size() >= 4 && name.substr(0, 4) == "z29_") {
            return is_builtin(name);
        }
        return true;
    }

private:
    DslZ29Builtins() = delete;
};

#endif // DSL_Z29_BUILTINS_HPP