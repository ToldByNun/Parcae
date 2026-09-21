"""CI-safe tests: stubs import, markers attach, semantics raise ParcaeDslStubError."""

from __future__ import annotations

import inspect

import pytest

import parcae.dsl.math as math_mod
from parcae.dsl.errors import ParcaeDslStubError, stub_error
from parcae.dsl.math import Z29Expr
from parcae.dsl.primitives import define_primitive
from parcae.dsl.testing import (
    fixture_test,
    negative_control,
    primitive_exhaustive_test,
    primitive_property_test,
    property_test,
    sweep_config,
    unit_vector,
)
from parcae.dsl.theory import ComposedTheory, Param, RuneStream, Theory

_STUB_HINT = "parcae-compile"


def _expect_stub(callable_obj, *args, **kwargs) -> None:
    with pytest.raises(ParcaeDslStubError, match=_STUB_HINT) as info:
        callable_obj(*args, **kwargs)
    assert "IDE autocomplete only" in str(info.value)


def _z29_uninitialized() -> Z29Expr:
    """Bypass fail-loud ``__init__`` so operator methods can be exercised."""
    return object.__new__(Z29Expr)


@pytest.mark.ci
def test_import_parcae_dsl_modules() -> None:
    import parcae.dsl.math as math_pkg
    import parcae.dsl.primitives as prim_mod
    import parcae.dsl.testing as testing_mod
    import parcae.dsl.theory as theory_mod

    assert math_pkg.__parcae_dsl_stub__ is True
    assert theory_mod.__parcae_dsl_stub__ is True
    assert prim_mod.__parcae_dsl_stub__ is True
    assert testing_mod.__parcae_dsl_stub__ is True


@pytest.mark.ci
def test_stub_error_message_points_at_compiler() -> None:
    err = stub_error()
    assert isinstance(err, ParcaeDslStubError)
    assert isinstance(err, RuntimeError)
    assert _STUB_HINT in str(err)
    assert "IDE autocomplete only" in str(err)


@pytest.mark.ci
def test_param_construction_allowed() -> None:
    p = Param(min=0, max=28)
    assert p.min == 0
    assert p.max == 28
    assert Param[int] is Param


@pytest.mark.ci
def test_theory_decorator_marks_without_raising() -> None:
    @Theory(name="my_affine", family="elementwise", tier="A", interrupts="none_by_design")
    class MyAffine:
        a: Param = Param(min=1, max=28)

        def decrypt_step(self, c: int, a: int) -> int:
            return c

        def apply(self, *_args: object, **_kwargs: object) -> None:
            return None

        def encrypt_step(self, *_args: object, **_kwargs: object) -> int:
            return 0

        def keystream_at(self, *_args: object, **_kwargs: object) -> int:
            return 0

        def derive_permutation(self, *_args: object, **_kwargs: object) -> list[int]:
            return []

        def interrupt_policy(self, *_args: object, **_kwargs: object) -> None:
            return None

        def step_params(self) -> dict[str, object]:
            return {}

    assert MyAffine.__parcae_dsl_stub__ is True  # type: ignore[attr-defined]
    assert MyAffine.__parcae_dsl_theory_name__ == "my_affine"  # type: ignore[attr-defined]
    assert MyAffine.__parcae_dsl_interrupts__ == "none_by_design"  # type: ignore[attr-defined]
    inst = MyAffine()
    for method_name in (
        "decrypt_step",
        "apply",
        "encrypt_step",
        "keystream_at",
        "derive_permutation",
        "interrupt_policy",
    ):
        _expect_stub(getattr(inst, method_name), 1, 2)
    # Non-semantic helpers are left as authored.
    assert inst.step_params() == {}


@pytest.mark.ci
def test_define_primitive_call_fails_loud() -> None:
    @define_primitive(name="poly2_mod29", signature="(i: Z29) -> Z29")
    def poly2_mod29(i: Z29Expr) -> Z29Expr:
        return i

    assert poly2_mod29.__parcae_dsl_primitive_name__ == "poly2_mod29"  # type: ignore[attr-defined]
    assert poly2_mod29.__parcae_dsl_signature__ == "(i: Z29) -> Z29"  # type: ignore[attr-defined]
    assert poly2_mod29.__parcae_dsl_stub__ is True  # type: ignore[attr-defined]
    _expect_stub(poly2_mod29, None)


@pytest.mark.ci
def test_all_z29_helpers_fail_loud() -> None:
    helpers = [
        name
        for name, obj in inspect.getmembers(math_mod)
        if name.startswith("z29_") and callable(obj)
    ]
    assert set(helpers) == set(math_mod.__all__) - {"Z29Expr"}
    assert len(helpers) >= 24
    for name in helpers:
        fn = getattr(math_mod, name)
        arity = len(inspect.signature(fn).parameters)
        args = tuple(range(arity))
        _expect_stub(fn, *args)


@pytest.mark.ci
def test_z29_expr_construction_and_ops_fail_loud() -> None:
    _expect_stub(Z29Expr)
    _expect_stub(Z29Expr, 1)

    expr = _z29_uninitialized()
    other = _z29_uninitialized()

    binary_ops = (
        expr.__add__,
        expr.__radd__,
        expr.__sub__,
        expr.__rsub__,
        expr.__mul__,
        expr.__rmul__,
        expr.__truediv__,
        expr.__rtruediv__,
        expr.__floordiv__,
        expr.__rfloordiv__,
        expr.__mod__,
        expr.__rmod__,
        expr.__pow__,
        expr.__rpow__,
        expr.__and__,
        expr.__rand__,
        expr.__or__,
        expr.__ror__,
        expr.__xor__,
        expr.__rxor__,
        expr.__lshift__,
        expr.__rlshift__,
        expr.__rshift__,
        expr.__rrshift__,
        expr.__eq__,
        expr.__ne__,
        expr.__lt__,
        expr.__le__,
        expr.__gt__,
        expr.__ge__,
    )
    for op in binary_ops:
        _expect_stub(op, other)

    for op in (expr.__invert__, expr.__neg__, expr.__pos__, expr.__bool__):
        _expect_stub(op)


@pytest.mark.ci
def test_rune_stream_fails_loud() -> None:
    _expect_stub(RuneStream)
    # Construction already raises; also cover the method via bypass.
    stream = object.__new__(RuneStream)
    _expect_stub(stream.rank_by_frequency)


@pytest.mark.ci
def test_composed_theory_and_testing_markers() -> None:
    @ComposedTheory(name="koan", steps=["atbash", "caesar"], tier="A")
    class Koan:
        caesar_shift: Param = Param(min=0, max=28)

        def step_params(self) -> dict:
            return {"atbash": {}, "caesar": {"shift": self.caesar_shift}}

        def apply(self, *_args: object, **_kwargs: object) -> None:
            return None

    assert Koan.__parcae_dsl_steps__ == ["atbash", "caesar"]  # type: ignore[attr-defined]
    assert Koan.__parcae_dsl_family__ == "compose"  # type: ignore[attr-defined]
    _expect_stub(Koan().apply)

    @property_test(theory="koan", trials=10, seed=1)
    def test_rt() -> None:
        return None

    assert test_rt.__parcae_dsl_test__ == "property_test"  # type: ignore[attr-defined]
    assert test_rt.__parcae_dsl_trials__ == 10  # type: ignore[attr-defined]
    assert test_rt.__parcae_dsl_seed__ == 1  # type: ignore[attr-defined]
    _expect_stub(test_rt)

    cfg = sweep_config(
        theory="koan",
        corpus="lp2_unsolved",
        compare_against="hypotheses.md#tier-c",
        param_grid={"shift": [0, 1]},
        record_metrics=["chi2"],
    )
    assert cfg["__parcae_dsl_stub__"] is True
    assert cfg["theory"] == "koan"
    assert cfg["param_grid"] == {"shift": [0, 1]}
    assert cfg["record_metrics"] == ["chi2"]


@pytest.mark.ci
def test_all_testing_decorators_fail_loud_on_invoke() -> None:
    @primitive_exhaustive_test(primitive="poly2_mod29")
    def t_ex() -> None:
        return None

    @primitive_property_test(primitive="poly2_mod29", trials=3)
    def t_pp() -> None:
        return None

    @unit_vector(theory="koan")
    def t_uv() -> None:
        return None

    @fixture_test(theory="koan", mode="sanity_only")
    def t_fx() -> None:
        return None

    @negative_control(theory="koan")
    def t_nc() -> None:
        return None

    cases = (
        (t_ex, "primitive_exhaustive_test", "poly2_mod29"),
        (t_pp, "primitive_property_test", "poly2_mod29"),
        (t_uv, "unit_vector", None),
        (t_fx, "fixture_test", None),
        (t_nc, "negative_control", None),
    )
    for fn, kind, primitive in cases:
        assert fn.__parcae_dsl_stub__ is True  # type: ignore[attr-defined]
        assert fn.__parcae_dsl_test__ == kind  # type: ignore[attr-defined]
        if primitive is not None:
            assert fn.__parcae_dsl_primitive__ == primitive  # type: ignore[attr-defined]
        _expect_stub(fn)

    assert t_fx.__parcae_dsl_mode__ == "sanity_only"  # type: ignore[attr-defined]
    assert t_pp.__parcae_dsl_trials__ == 3  # type: ignore[attr-defined]
