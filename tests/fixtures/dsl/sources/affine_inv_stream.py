"""Affine-style theory with inv(a) for DslOptimize hoist-in-compile tests.

Compile:

    parcae-compile tests/fixtures/dsl/sources/affine_inv_stream.py

URI: parcae://theories/affine_inv_stream@1
"""

from parcae.dsl.math import Z29Expr, z29_inv
from parcae.dsl.theory import Theory, Param


@Theory(
    name="affine_inv_stream",
    family="elementwise",
    tier="A",
)
class AffineInvStream:
    a: Param[int] = Param(min=1, max=28)
    b: Param[int] = Param(min=0, max=28)

    def encrypt_step(self, x: Z29Expr) -> Z29Expr:
        return (self.a * x) + self.b

    def decrypt_step(self, x: Z29Expr) -> Z29Expr:
        return z29_inv(self.a) * (x - self.b)
