"""parcae.dsl.theory — IDE stubs; Theory/ComposedTheory markers, fail-loud apply."""

from __future__ import annotations

from typing import Any, TypeVar

from parcae.dsl.errors import raise_stub

__parcae_dsl_stub__ = True

T = TypeVar("T")


class Param:
    """Parameter metadata. Construction is allowed (class-body fields)."""

    __parcae_dsl_stub__ = True

    def __init__(self, *, min: int | None = None, max: int | None = None) -> None:
        self.min = min
        self.max = max

    def __class_getitem__(cls, _item: Any) -> type[Param]:
        return cls


class RuneStream:
    """Host helper stub. Construction / methods fail-loud (no fake ranking)."""

    __parcae_dsl_stub__ = True

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise_stub()

    def rank_by_frequency(self, *_args: Any, **_kwargs: Any) -> list[int]:
        raise_stub()


def Theory(
    *,
    name: str,
    family: str,
    tier: str,
    interrupts: str | None = None,
    **_kwargs: Any,
) -> Any:
    """Class decorator: mark stub; do not build IR."""

    def decorate(cls: type[T]) -> type[T]:
        setattr(cls, "__parcae_dsl_stub__", True)
        setattr(cls, "__parcae_dsl_theory_name__", name)
        setattr(cls, "__parcae_dsl_family__", family)
        setattr(cls, "__parcae_dsl_tier__", tier)
        if interrupts is not None:
            setattr(cls, "__parcae_dsl_interrupts__", interrupts)
        _install_fail_loud_semantic_methods(cls)
        return cls

    return decorate


def ComposedTheory(
    *,
    name: str,
    steps: list[str],
    tier: str,
    **_kwargs: Any,
) -> Any:
    """Class decorator for compose chains: mark stub only."""

    def decorate(cls: type[T]) -> type[T]:
        setattr(cls, "__parcae_dsl_stub__", True)
        setattr(cls, "__parcae_dsl_theory_name__", name)
        setattr(cls, "__parcae_dsl_family__", "compose")
        setattr(cls, "__parcae_dsl_tier__", tier)
        setattr(cls, "__parcae_dsl_steps__", list(steps))
        _install_fail_loud_semantic_methods(cls)
        return cls

    return decorate


def _install_fail_loud_semantic_methods(cls: type[Any]) -> None:
    """Crypto / apply entry points raise if invoked; metadata methods stay as written."""
    for attr in (
        "apply",
        "encrypt_step",
        "decrypt_step",
        "keystream_at",
        "derive_permutation",
        "interrupt_policy",
    ):
        if attr in cls.__dict__:
            original = cls.__dict__[attr]
            if callable(original) and not isinstance(original, (staticmethod, classmethod)):
                setattr(cls, attr, _FailLoudMethod(original))


class _FailLoudMethod:
    """Wraps a user method so definition is visible but invocation fails loud."""

    __parcae_dsl_stub__ = True

    def __init__(self, original: Any) -> None:
        self._original = original
        self.__name__ = getattr(original, "__name__", "method")
        self.__doc__ = getattr(original, "__doc__", None)

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self

        def bound(*_args: Any, **_kwargs: Any) -> Any:
            raise_stub()

        bound.__name__ = self.__name__
        bound.__doc__ = self.__doc__
        bound.__parcae_dsl_stub__ = True  # type: ignore[attr-defined]
        return bound


__all__ = [
    "Param",
    "RuneStream",
    "Theory",
    "ComposedTheory",
]
