"""Full-lifecycle compose example (docs/spec/dsl.md @ComposedTheory).

Authoring path:
  1. Write this file (IDE stubs only — semantic calls raise).
  2. ``parcae-compile theories/examples/full_lifecycle_example.py``
  3. Artifact under ``data/theories/koan1_style/<ver>/`` with fusion status,
     CPU/CUDA emit, and ``envelope.json`` (theory URI bridge).

Steps reference frozen catalog transform ids (``atbash``, ``caesar``).
"""

from parcae.dsl.theory import ComposedTheory, Param


@ComposedTheory(
    name="koan1_style",
    steps=["atbash", "caesar"],
    tier="A",
)
class Koan1Style:
    caesar_shift: Param[int] = Param(min=0, max=28)

    def step_params(self) -> dict:
        return {"atbash": {}, "caesar": {"shift": self.caesar_shift}}
