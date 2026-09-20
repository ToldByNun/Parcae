from typing import Any, TypeVar

T = TypeVar("T")

class Param:
    min: int | None
    max: int | None
    def __init__(self, *, min: int | None = None, max: int | None = None) -> None: ...
    def __class_getitem__(cls, _item: Any) -> type[Param]: ...

class RuneStream:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...
    def rank_by_frequency(self, *_args: Any, **_kwargs: Any) -> list[int]: ...

def Theory(
    *,
    name: str,
    family: str,
    tier: str,
    interrupts: str | None = None,
    **_kwargs: Any,
) -> Any: ...

def ComposedTheory(
    *,
    name: str,
    steps: list[str],
    tier: str,
    **_kwargs: Any,
) -> Any: ...
