"""Matrix / autokey builtin example (docs/spec/dsl.md z29_det / z29_matmul).

Demonstrates:
  - @define_primitive wrapping z29_det (BuildIr → MatrixIr::det_expr)
  - @define_primitive wrapping z29_matmul(...)[i]
  - @Theory elementwise mix via det key material
  - @ComposedTheory chaining catalog ``atbash`` with the module theory
    (DslFuse inlines; DSL-only leaf → fused emit, no ComposeTransform staged twin)

Compile:

    parcae-compile theories/examples/matrix_builtins_example.py

URIs after compile:
  parcae://theories/matrix_mix_stream@1
  parcae://theories/atbash_then_matrix_mix@1
"""

from parcae.dsl.math import Z29Expr, z29_det, z29_matmul
from parcae.dsl.primitives import define_primitive
from parcae.dsl.theory import ComposedTheory, Param, Theory


@define_primitive(
    name="det2_mod29",
    signature="(a: Z29, b: Z29, c: Z29, d: Z29) -> Z29",
)
def det2_mod29(a: Z29Expr, b: Z29Expr, c: Z29Expr, d: Z29Expr) -> Z29Expr:
    return z29_det((a, b, c, d))


@define_primitive(
    name="hill2_y0",
    signature="(a: Z29, b: Z29, c: Z29, d: Z29, x0: Z29, x1: Z29) -> Z29",
)
def hill2_y0(
    a: Z29Expr, b: Z29Expr, c: Z29Expr, d: Z29Expr, x0: Z29Expr, x1: Z29Expr
) -> Z29Expr:
    return z29_matmul((a, b, c, d), (x0, x1))[0]


@Theory(
    name="matrix_mix_stream",
    family="elementwise",
    tier="A",
)
class MatrixMixStream:
    a: Param[int] = Param(min=0, max=28)
    b: Param[int] = Param(min=0, max=28)
    c: Param[int] = Param(min=0, max=28)
    d: Param[int] = Param(min=0, max=28)

    def encrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x + det2_mod29(self.a, self.b, self.c, self.d)

    def decrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x - det2_mod29(self.a, self.b, self.c, self.d)


@ComposedTheory(
    name="atbash_then_matrix_mix",
    steps=["atbash", "matrix_mix_stream"],
    tier="A",
)
class AtbashThenMatrixMix:
    a: Param[int] = Param(min=0, max=28)
    b: Param[int] = Param(min=0, max=28)
    c: Param[int] = Param(min=0, max=28)
    d: Param[int] = Param(min=0, max=28)

    def step_params(self) -> dict:
        return {
            "atbash": {},
            "matrix_mix_stream": {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "d": self.d,
            },
        }
