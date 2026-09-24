"""Research example: ThreadVarying HotLoop `if` with `#ignore DSL_FLAG`.

**Not** compiled by default CI. Requires an explicit opt-in:

    parcae-compile theories/examples/ignore_divergent_example.py --allow-dsl-ignores

Without the flag, `parcae-compile` fails with **E031** (ignore not allowed).
With the flag, **W010** is emitted and the artifact records
`dsl_ignores_applied: ["divergent_branch"]`. Emit may use a real C++/CUDA
conditional (`Select.prefer_branch`) — warp-divergence risk on device.

Prefer Param/const predicates (see `param_select_example.py`) for portable
theories. Normative: docs/spec/dsl.md § Execution scopes, docs/spec/dsl-ast-json.md
§ Directives.
"""

from parcae.dsl.math import Z29Expr
from parcae.dsl.primitives import define_primitive
from parcae.dsl.theory import Theory


@define_primitive(
    name="id_divergent_branch",
    signature="(x: Z29) -> Z29",
)
def id_divergent_branch(x: Z29Expr) -> Z29Expr:
    #ignore DSL_FLAG:divergent_branch
    if x == 0:
        return x
    else:
        return x


@Theory(
    name="ignore_divergent_example",
    family="elementwise",
    tier="A",
)
class IgnoreDivergentExample:
    def encrypt_step(self, x: Z29Expr) -> Z29Expr:
        return id_divergent_branch(x)

    def decrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x
