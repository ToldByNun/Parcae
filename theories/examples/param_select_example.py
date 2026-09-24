"""Example: Param-uniform HotLoop `if` → `Z29Expr::Select` (no `#ignore`).

Compiles with default `parcae-compile` (no `--allow-dsl-ignores`).
The predicate uses a Param (`mode`), so `DslDivergenceGate` accepts it as
uniform / LoopInvariant (**W011**). Emit uses `Z29::select` / `Z29Device::select`.

See docs/spec/dsl.md § Execution scopes and docs/architecture/python-transpiler.md.
"""

from parcae.dsl.math import Z29Expr
from parcae.dsl.primitives import define_primitive
from parcae.dsl.theory import Theory, Param


@define_primitive(
    name="param_branch_add",
    signature="(x: Z29, shift: Z29, mode: Z29) -> Z29",
)
def param_branch_add(x: Z29Expr, shift: Z29Expr, mode: Z29Expr) -> Z29Expr:
    if mode == 1:
        return x + shift
    else:
        return x + shift


@Theory(
    name="param_select_example",
    family="elementwise",
    tier="A",
)
class ParamSelectExample:
    shift: Param[int] = Param(min=0, max=28)
    mode: Param[int] = Param(min=0, max=1)

    def encrypt_step(self, x: Z29Expr) -> Z29Expr:
        return param_branch_add(x, self.shift, self.mode)

    def decrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x - self.shift
