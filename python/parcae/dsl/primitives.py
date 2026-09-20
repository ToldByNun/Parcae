"""parcae.dsl.primitives — @define_primitive marker; calls fail-loud."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from parcae.dsl.errors import raise_stub

__parcae_dsl_stub__ = True

F = TypeVar("F", bound=Callable[..., Any])


def define_primitive(*, name: str, signature: str, **_kwargs: Any) -> Callable[[F], F]:
    """Function decorator: record metadata; invoking the primitive raises."""

    def decorate(fn: F) -> F:
        def wrapped(*_args: Any, **_kwargs: Any) -> Any:
            raise_stub()

        wrapped.__name__ = getattr(fn, "__name__", name)
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        wrapped.__annotations__ = getattr(fn, "__annotations__", {})
        wrapped.__parcae_dsl_stub__ = True  # type: ignore[attr-defined]
        wrapped.__parcae_dsl_primitive_name__ = name  # type: ignore[attr-defined]
        wrapped.__parcae_dsl_signature__ = signature  # type: ignore[attr-defined]
        # Keep a reference to the authored body for static tooling (never executed).
        wrapped.__parcae_dsl_source_fn__ = fn  # type: ignore[attr-defined]
        return wrapped  # type: ignore[return-value]

    return decorate


__all__ = [
    "define_primitive",
]
