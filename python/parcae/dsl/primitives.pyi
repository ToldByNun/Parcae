from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

def define_primitive(*, name: str, signature: str, **_kwargs: Any) -> Callable[[F], F]: ...
