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

    def __neg__(self) -> Z29Expr:
        raise_stub()


def z29_add(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_sub(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_mul(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


def z29_inv(_a: Any) -> Z29Expr:
    raise_stub()


def z29_mod(_x: Any, _y: Any) -> Z29Expr:
    raise_stub()


__all__ = [
    "Z29Expr",
    "z29_add",
    "z29_sub",
    "z29_mul",
    "z29_inv",
    "z29_mod",
]
