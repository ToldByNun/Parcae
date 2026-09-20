from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def primitive_exhaustive_test(*, primitive: str, **_kwargs: Any) -> Callable[[F], F]: ...
def primitive_property_test(
    *, primitive: str, trials: int = 100, **_kwargs: Any
) -> Callable[[F], F]: ...
def unit_vector(*, theory: str, **_kwargs: Any) -> Callable[[F], F]: ...
def property_test(
    *, theory: str, trials: int = 100, seed: int | None = None, **_kwargs: Any
) -> Callable[[F], F]: ...
def fixture_test(*, theory: str, mode: str = "sanity_only", **_kwargs: Any) -> Callable[[F], F]: ...
def negative_control(*, theory: str, **_kwargs: Any) -> Callable[[F], F]: ...
def sweep_config(
    *,
    theory: str,
    corpus: str,
    param_grid: dict[str, Any] | None = None,
    record_metrics: list[str] | None = None,
    compare_against: str | None = None,
    **_kwargs: Any,
) -> dict[str, Any]: ...
