"""parcae.dsl.math — IDE stubs; operators and z29_* calls fail-loud."""

from __future__ import annotations

from typing import Any

from parcae.dsl.errors import raise_stub

__parcae_dsl_stub__ = True


class Z29Expr:
    """Placeholder type for annotations. Any arithmetic/call raises."""

    __parcae_dsl_stub__ = True

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise_stub()

    def __add__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __radd__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __sub__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rsub__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __mul__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rmul__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __truediv__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rtruediv__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __floordiv__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rfloordiv__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __mod__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rmod__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __pow__(self, _other: Any, _mod: Any = None) -> Z29Expr:
        raise_stub()

    def __rpow__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __and__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rand__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __or__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __ror__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __xor__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rxor__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __lshift__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rlshift__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rshift__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __rrshift__(self, _other: Any) -> Z29Expr:
        raise_stub()

    def __invert__(self) -> Z29Expr:
        raise_stub()

    def __neg__(self) -> Z29Expr:
        raise_stub()

    def __pos__(self) -> Z29Expr:
        raise_stub()

    def __eq__(self, _other: Any) -> bool:  # type: ignore[override]
        raise_stub()

    def __ne__(self, _other: Any) -> bool:  # type: ignore[override]
        raise_stub()

    def __lt__(self, _other: Any) -> bool:
        raise_stub()

    def __le__(self, _other: Any) -> bool:
        raise_stub()

    def __gt__(self, _other: Any) -> bool:
        raise_stub()

    def __ge__(self, _other: Any) -> bool:
        raise_stub()

    def __bool__(self) -> bool:
        raise_stub()


def z29_add(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_sub(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_mul(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_div(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_floordiv(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_inv(_a: Any) -> Z29Expr:
    raise_stub()


def z29_mod(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_pow(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_bit_and(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_bit_or(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_bit_xor(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_bit_not(_x: Any) -> Z29Expr:
    raise_stub()


def z29_lshift(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_rshift(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_eq(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_ne(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_lt(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_le(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_gt(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_ge(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_bool_and(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_bool_or(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_bool_not(_x: Any) -> Z29Expr:
    raise_stub()


def z29_neg(_x: Any) -> Z29Expr:
    raise_stub()


def z29_atbash(_x: Any) -> Z29Expr:
    raise_stub()


def z29_select(_cond: Any, _a: Any, _b: Any) -> Z29Expr:
    raise_stub()


def z29_matmul(_matrix: Any, _rune_vec: Any) -> Z29Expr:
    """Hill-style matrix × vector over Z29 (intrinsic; emit/fuse follow-on)."""
    raise_stub()


def z29_det(_matrix: Any) -> Z29Expr:
    """Determinant mod 29 (intrinsic; emit/fuse follow-on)."""
    raise_stub()


def z29_autokey_shift(_stream: Any, _lag: Any) -> Z29Expr:
    """Autokey lag/ringbuffer read (intrinsic; emit/fuse follow-on)."""
    raise_stub()


__all__ = [
    "Z29Expr",
    "z29_add",
    "z29_sub",
    "z29_mul",
    "z29_div",
    "z29_floordiv",
    "z29_inv",
    "z29_mod",
    "z29_pow",
    "z29_bit_and",
    "z29_bit_or",
    "z29_bit_xor",
    "z29_bit_not",
    "z29_lshift",
    "z29_rshift",
    "z29_eq",
    "z29_ne",
    "z29_lt",
    "z29_le",
    "z29_gt",
    "z29_ge",
    "z29_bool_and",
    "z29_bool_or",
    "z29_bool_not",
    "z29_neg",
    "z29_atbash",
    "z29_select",
    "z29_matmul",
    "z29_det",
    "z29_autokey_shift",
]
