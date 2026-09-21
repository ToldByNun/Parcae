"""Canonical Tier-B new-math theory example (docs/spec/dsl.md).

Demonstrates:
  - @define_primitive with arity-4 poly2_mod29 (exhaustive-verify sized)
  - @Theory keyed_stream with interrupts="none_by_design"
  - required structural_claim() for Tier B (research stance != verify outcome)

Compile (not IDE stubs):

    parcae-compile theories/examples/new_math_example.py

URI after compile: parcae://theories/quadratic_polynomial_stream@1
"""

from parcae.dsl.math import Z29Expr
from parcae.dsl.primitives import define_primitive
from parcae.dsl.theory import Theory, Param


@define_primitive(
    name="poly2_mod29",
    signature="(i: Z29, c2: Z29, c1: Z29, c0: Z29) -> Z29",
)
def poly2_mod29(i: Z29Expr, c2: Z29Expr, c1: Z29Expr, c0: Z29Expr) -> Z29Expr:
    return (c2 * i * i) + (c1 * i) + c0


@Theory(
    name="quadratic_polynomial_stream",
    family="keyed_stream",
    tier="B",
    interrupts="none_by_design",
)
class QuadraticPolynomialStream:
    c2: Param[int] = Param(min=0, max=28)
    c1: Param[int] = Param(min=0, max=28)
    c0: Param[int] = Param(min=0, max=28)

    def structural_claim(self) -> str:
        return (
            "Speculative. Quadratic keystream on Z29; exhaustive verify proves "
            "totality/determinism only — not yet verified against real LP "
            "statistics (Tier B, not Tier A)."
        )

    def keystream_at(self, i: Z29Expr) -> Z29Expr:
        return poly2_mod29(i, self.c2, self.c1, self.c0)

    def encrypt_step(self, x: Z29Expr, i: Z29Expr) -> Z29Expr:
        return x + self.keystream_at(i)

    def decrypt_step(self, x: Z29Expr, i: Z29Expr) -> Z29Expr:
        return x - self.keystream_at(i)
