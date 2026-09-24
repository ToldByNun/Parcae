"""Smart-compiler golden: ThreadVarying if with honored #ignore DSL_FLAG.

Requires --allow-dsl-ignores. Emits prefer_branch conditional + W010.
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
    name="smart_ignore_divergent",
    family="elementwise",
    tier="A",
)
class SmartIgnoreDivergent:
    def encrypt_step(self, x: Z29Expr) -> Z29Expr:
        return id_divergent_branch(x)

    def decrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x
