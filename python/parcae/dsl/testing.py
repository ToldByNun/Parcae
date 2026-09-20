"""parcae.dsl.testing — decorator markers; running tests fails loud."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from parcae.dsl.errors import raise_stub

__parcae_dsl_stub__ = True

F = TypeVar("F", bound=Callable[..., Any])


def _mark_and_fail_loud(fn: F, **meta: Any) -> F:
    def wrapped(*_args: Any, **_kwargs: Any) -> Any:
        raise_stub()

    wrapped.__name__ = getattr(fn, "__name__", "test")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__parcae_dsl_stub__ = True  # type: ignore[attr-defined]
    wrapped.__parcae_dsl_source_fn__ = fn  # type: ignore[attr-defined]
    for key, value in meta.items():
        setattr(wrapped, key, value)
    return wrapped  # type: ignore[return-value]


def primitive_exhaustive_test(*, primitive: str, **_kwargs: Any) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        return _mark_and_fail_loud(
            fn,
            __parcae_dsl_test__="primitive_exhaustive_test",
            __parcae_dsl_primitive__=primitive,
        )

    return decorate


def primitive_property_test(
    *, primitive: str, trials: int = 100, **_kwargs: Any
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        return _mark_and_fail_loud(
            fn,
            __parcae_dsl_test__="primitive_property_test",
            __parcae_dsl_primitive__=primitive,
            __parcae_dsl_trials__=trials,
        )

    return decorate


def unit_vector(*, theory: str, **_kwargs: Any) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        return _mark_and_fail_loud(
            fn,
            __parcae_dsl_test__="unit_vector",
            __parcae_dsl_theory__=theory,
        )

    return decorate


def property_test(
    *, theory: str, trials: int = 100, seed: int | None = None, **_kwargs: Any
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        return _mark_and_fail_loud(
            fn,
            __parcae_dsl_test__="property_test",
            __parcae_dsl_theory__=theory,
            __parcae_dsl_trials__=trials,
            __parcae_dsl_seed__=seed,
        )

    return decorate


def fixture_test(*, theory: str, mode: str = "sanity_only", **_kwargs: Any) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        return _mark_and_fail_loud(
            fn,
            __parcae_dsl_test__="fixture_test",
            __parcae_dsl_theory__=theory,
            __parcae_dsl_mode__=mode,
        )

    return decorate


def negative_control(*, theory: str, **_kwargs: Any) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        return _mark_and_fail_loud(
            fn,
            __parcae_dsl_test__="negative_control",
            __parcae_dsl_theory__=theory,
        )

    return decorate


def sweep_config(
    *,
    theory: str,
    corpus: str,
    param_grid: dict[str, Any] | None = None,
    record_metrics: list[str] | None = None,
    compare_against: str | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Return metadata dict only — does not run a sweep."""
    return {
        "__parcae_dsl_stub__": True,
        "theory": theory,
        "corpus": corpus,
        "param_grid": param_grid or {},
        "record_metrics": record_metrics or [],
        "compare_against": compare_against,
    }


__all__ = [
    "primitive_exhaustive_test",
    "primitive_property_test",
    "unit_vector",
    "property_test",
    "fixture_test",
    "negative_control",
    "sweep_config",
]
