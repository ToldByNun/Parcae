"""Smart-compiler golden: Param-uniform HotLoop if → Select mux (no ignore).

Compiles without --allow-dsl-ignores. Emits Z29::select / Z29Device::select.
"""

from parcae.dsl.math import Z29Expr
from parcae.dsl.primitives import define_primitive
from parcae.dsl.theory import Theory, Param


@define_primitive(
    name="param_branch_add",
    signature="(x: Z29, shift: Z29, mode: Z29) -> Z29",
)
def param_branch_add(x: Z29Expr, shift: Z29Expr, mode: Z29Expr) -> Z29Expr:
    # Uniform / Param predicate → W011 + Select (not E033).
    if mode == 1:
        return x + shift
    else:
        return x + shift


@Theory(
    name="smart_select_param",
    family="elementwise",
    tier="A",
)
class SmartSelectParam:
    shift: Param[int] = Param(min=0, max=28)
    mode: Param[int] = Param(min=0, max=1)

    def encrypt_step(self, x: Z29Expr) -> Z29Expr:
        return param_branch_add(x, self.shift, self.mode)

    def decrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x - self.shift
