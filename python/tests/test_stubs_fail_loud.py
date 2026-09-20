"""CI-safe tests: stubs import, markers attach, semantics fail loud."""

from __future__ import annotations

import pytest

from parcae.dsl.errors import ParcaeDslStubError
from parcae.dsl.math import Z29Expr, z29_add, z29_mul
from parcae.dsl.primitives import define_primitive
from parcae.dsl.testing import property_test, sweep_config
from parcae.dsl.theory import ComposedTheory, Param, Theory


@pytest.mark.ci
def test_import_parcae_dsl_modules() -> None:
    import parcae.dsl.math as math_mod
    import parcae.dsl.primitives as prim_mod
    import parcae.dsl.testing as testing_mod
    import parcae.dsl.theory as theory_mod

    assert math_mod.__parcae_dsl_stub__ is True
    assert theory_mod.__parcae_dsl_stub__ is True
    assert prim_mod.__parcae_dsl_stub__ is True
    assert testing_mod.__parcae_dsl_stub__ is True


@pytest.mark.ci
def test_param_construction_allowed() -> None:
    p = Param(min=0, max=28)
    assert p.min == 0
    assert p.max == 28
    assert Param[int] is Param


@pytest.mark.ci
def test_theory_decorator_marks_without_raising() -> None:
    @Theory(name="my_affine", family="elementwise", tier="A")
    class MyAffine:
        a: Param = Param(min=1, max=28)

        def decrypt_step(self, c: int, a: int) -> int:
            return c

    assert MyAffine.__parcae_dsl_stub__ is True  # type: ignore[attr-defined]
    assert MyAffine.__parcae_dsl_theory_name__ == "my_affine"  # type: ignore[attr-defined]
    inst = MyAffine()
    with pytest.raises(ParcaeDslStubError, match="parcae-compile"):
        inst.decrypt_step(1, 2)


@pytest.mark.ci
def test_define_primitive_call_fails_loud() -> None:
    @define_primitive(name="poly2_mod29", signature="(i: Z29) -> Z29")
    def poly2_mod29(i: Z29Expr) -> Z29Expr:
        return i

    assert poly2_mod29.__parcae_dsl_primitive_name__ == "poly2_mod29"  # type: ignore[attr-defined]
    with pytest.raises(ParcaeDslStubError, match="parcae-compile"):
        poly2_mod29(None)  # type: ignore[arg-type]


@pytest.mark.ci
def test_z29_ops_fail_loud() -> None:
    with pytest.raises(ParcaeDslStubError):
        z29_add(1, 2)
    with pytest.raises(ParcaeDslStubError):
        z29_mul(1, 2)
    with pytest.raises(ParcaeDslStubError):
        Z29Expr()


@pytest.mark.ci
def test_composed_theory_and_testing_markers() -> None:
    @ComposedTheory(name="koan", steps=["atbash", "caesar"], tier="A")
    class Koan:
        caesar_shift: Param = Param(min=0, max=28)

        def step_params(self) -> dict:
            return {"atbash": {}, "caesar": {"shift": self.caesar_shift}}

    assert Koan.__parcae_dsl_steps__ == ["atbash", "caesar"]  # type: ignore[attr-defined]

    @property_test(theory="koan", trials=10, seed=1)
    def test_rt() -> None:
        return None

    with pytest.raises(ParcaeDslStubError):
        test_rt()

    cfg = sweep_config(
        theory="koan",
        corpus="lp2_unsolved",
        compare_against="hypotheses.md#tier-c",
    )
    assert cfg["__parcae_dsl_stub__"] is True
    assert cfg["theory"] == "koan"
