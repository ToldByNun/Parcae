"""Fail-loud error for IDE-only parcae.dsl stubs."""

from __future__ import annotations


class ParcaeDslStubError(RuntimeError):
    """Raised when stub code is executed as if it were the compiler."""


_STUB_MESSAGE = (
    "This is IDE autocomplete only (parcae.dsl stubs). "
    "Run `parcae-compile <theory.py>` to compile and verify. "
    "Stubs must not be treated as a verified theory."
)


def stub_error() -> ParcaeDslStubError:
    return ParcaeDslStubError(_STUB_MESSAGE)


def raise_stub() -> None:
    raise stub_error()
