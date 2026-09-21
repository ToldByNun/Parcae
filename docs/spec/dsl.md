# Spec: Parcae Theory DSL

**Status:** Normative  
**`dsl_spec_version`:** `1.0.0`  
**Schema (related):** `parcae.theory_artifact.v0` — see [theory-artifact.md](theory-artifact.md)  
**Related:** [z29.md](z29.md), [transforms.md](transforms.md), [interrupts.md](interrupts.md),
[hypotheses.md](../research/hypotheses.md)

This document is the binding language contract for authoring theories in
Python-looking source files (`.py`) that `parcae-compile` accepts. Crypto
semantics, verification, optimization, fusion, and code emission **MUST** be
implemented in the C++20 toolkit. CPython **MAY** be used only as a one-shot
syntax frontend (`ast.parse` → JSON); it **MUST NOT** run verification or emit
loops.

Architecture overview (non-normative implementation guide):
[`docs/architecture/python-transpiler.md`](../architecture/python-transpiler.md).

---

## `dsl_spec_version` policy

### Definition

`dsl_spec_version` is a SemVer triple `MAJOR.MINOR.PATCH` published at the top of
this file and mirrored by the compiler as `DslSpecVersion::current`.

| Component | Bump when |
|-----------|-----------|
| **MAJOR** | Breaking language or verify semantics: new required fields, removed constructs, stricter gates that invalidate previously accepted theories, changes to \(\mathbb{Z}_{29}\) op meaning in the DSL |
| **MINOR** | Backward-compatible extensions: new optional families, additional diagnostics, new optional metadata |
| **PATCH** | Clarifications, typos, non-semantic wording — **MUST NOT** change acceptance of previously valid sources |

### Binding rules

1. Every conforming `parcae-compile` **MUST** embed the compiler’s current
   `dsl_spec_version` into each written theory artifact manifest.
2. `TheoryRegistry` load, `parcae-validate` on a theory URI/path, and
   `parcae-sweep` **MUST** compare `artifact.dsl_spec_version` to
   `DslSpecVersion::current`:
   - If **MAJOR** differs → artifact is **invalid**; tools **MUST** fail with a
     clear message requiring re-compile. **MUST NOT** silently continue.
   - If MAJOR matches and artifact MINOR/PATCH is older → artifact **MAY** remain
     valid (non-breaking evolution).
   - If artifact MAJOR.MINOR is **newer** than the running compiler → **MUST**
     fail (forward-incompatible).
3. `parcae-catalog --theories` **MUST** mark stale-MAJOR artifacts
   (`stale_spec: true`) and **MUST NOT** present them as ready-to-run.
4. After a **MAJOR** bump, committed example theories under `theories/examples/`
   **MUST** be recompiled before release; CI **SHOULD** fail if examples are
   stale.

Normative artifact field details: [theory-artifact.md](theory-artifact.md).

---

## Pipeline roles (normative split)

| Stage | Implementation | Requirement |
|-------|----------------|-------------|
| Authoring | `.py` sources | Human/IDE-editable; valid Python syntax for the allowed subset |
| Syntax frontend | CPython `ast.parse` → `parcae.dsl_ast_json.v0` | One shot per compile; wire format in [dsl-ast-json.md](dsl-ast-json.md) |
| Semantic gate, IR, verify, optimize, fuse, emit | C++20 | **MUST NOT** call CPython inside verify/emit loops |
| IDE stubs (`parcae.dsl.*`) | Thin Python package | Import OK; **fail-loud** on semantic runtime — see below |

A conforming toolchain **MUST NOT** ship a hand-written full Python lexer/parser
in C++ as the primary syntax path.

---

## IDE stubs (fail-loud)

Stub packages that provide `parcae.dsl.*` for editor autocomplete:

| Action | Requirement |
|--------|-------------|
| `import parcae.dsl…` | **MUST** succeed |
| Decorator application at import/definition time | **MAY** attach a marker attribute (e.g. `__parcae_dsl_stub__`); **MUST NOT** build executable IR or claim verification |
| Any call that simulates theory/primitive semantics (operators on stub exprs, invoking a primitive, `apply`, running test decorators as executors) | **MUST** raise immediately with a message that stubs are IDE-only and the user **MUST** run `parcae-compile` |

Stubs **MUST NOT** be documented or behave as a substitute for `parcae-compile`.

Contributor-facing clarification (non-normative guide):
[`docs/architecture/dsl-stubs.md`](../architecture/dsl-stubs.md).

---

## Allowed imports

Source files **MUST** import only from:

```text
parcae.dsl.math
parcae.dsl.theory
parcae.dsl.primitives
parcae.dsl.testing
```

Relative re-exports **within** those modules are allowed for the stub package.
Any other import (including `numpy`, `os`, stdlib crypto, etc.) **MUST** produce
a compile diagnostic with source location.

---

## Primitives

### `@define_primitive`

A primitive is a pure \(\mathbb{Z}_{29}\) function with a string signature and a
body that returns a `Z29Expr` tree (not general Python).

```python
@define_primitive(
    name="poly2_mod29",
    signature="(i: Z29, c2: Z29, c1: Z29, c0: Z29) -> Z29",
)
def poly2_mod29(i: Z29Expr, c2: Z29Expr, c1: Z29Expr, c0: Z29Expr) -> Z29Expr:
    return (c2 * i * i) + (c1 * i) + c0
```

| Rule | Requirement |
|------|-------------|
| `name` | Non-empty; stable id for registry / artifacts |
| `signature` | Must match parameter list arity and `Z29` / `-> Z29` shape |
| Body | Expression tree only — see subset table |
| Recursion | **MUST NOT** call itself (directly or indirectly) |
| Side effects | **MUST NOT** |

### Primitive body subset


| Allowed | Forbidden (compile error + lineno) |
|---------|-------------------------------------|
| `return` of a `Z29Expr` | `for` / `while` / `async` / `with` / `try` |
| Full operator set on `Z29Expr` (see table below) | List/dict/set comprehensions, `lambda`, `yield` |
| Calls to `z29_*` and other registered primitives | `eval` / `exec` / `open` / arbitrary attributes |
| Local bindings to expressions only | Hidden state, RNG, I/O |

### Core math ops (`parcae.dsl.math`)

Semantics **MUST** match [z29.md](z29.md). All results are `Index29` in `0..28`.
Bitwise / shift ops run on integer representatives then reduce `mod 29`.
Comparisons and bool-ish ops yield `0` or `1`. `/` is **modular** division
(`mul(x, inv(y))`), not IEEE float. `//` is integer floor-division of
representatives. `**` is modular exponentiation (`0**0` → `1`). `~x` is
`(-x-1) mod 29` (equals Atbash `28-x`).

| DSL / Python | Meaning |
|--------------|---------|
| `z29_add` / `+` | `(x + y) mod 29` |
| `z29_sub` / `-` | `(x - y) mod 29` |
| `z29_mul` / `*` | `(x * y) mod 29` |
| `z29_div` / `/` | modular `mul(x, inv(y))`; `y=0` → E040 |
| `z29_floordiv` / `//` | integer `x // y`; `y=0` → E040 |
| `z29_mod` / `%` | remainder of representatives; `y=0` → E040 |
| `z29_pow` / `**` | `x^y mod 29` |
| `z29_inv` | Inverse on `1..28`; `inv(0)` → E040 |
| `z29_neg` / unary `-` | additive inverse |
| `z29_bit_and` / `&` | `(x & y) mod 29` |
| `z29_bit_or` / `\|` | `(x \| y) mod 29` |
| `z29_bit_xor` / `^` | `(x ^ y) mod 29` |
| `z29_bit_not` / `~` | `(-x-1) mod 29` |
| `z29_lshift` / `<<` | `(x << y) mod 29` |
| `z29_rshift` / `>>` | `x >> y` (already in-domain) |
| `z29_eq`…`z29_ge` / `== != < <= > >=` | `0` or `1` |
| `z29_bool_and` / `and` | nonzero ∧ nonzero → `1` else `0` |
| `z29_bool_or` / `or` | nonzero ∨ nonzero → `1` else `0` |
| `z29_bool_not` / `not` | zero → `1` else `0` |
| `z29_atbash` | `28 - x` |

`Z29Expr` operator overloads **MUST** build IR in the compiler path, not execute
arithmetic in the stub package (stubs fail-loud). `MatMult` (`@`) and
identity/container compares (`is` / `in`) **MUST** be rejected.

---

## Theories

### `@Theory`

```python
@Theory(
    name="my_affine",
    family="elementwise",
    tier="A",
)
class MyAffine:
    a: Param[int] = Param(min=1, max=28)
    b: Param[int] = Param(min=0, max=28)
    ...
```

| Field | Requirement |
|-------|-------------|
| `name` | Stable theory id (`[a-z][a-z0-9_]*` recommended) |
| `family` | One of the families below |
| `tier` | `"A"` \| `"B"` \| `"C"` (research tiers; see hypotheses.md) |
| `interrupts` | Optional; `"none_by_design"` when no interrupt policy is intentional |

### `@ComposedTheory`

```python
@ComposedTheory(
    name="my_koan1_style",
    steps=["atbash", "caesar"],
    tier="A",
)
class MyKoan1Style:
    caesar_shift: Param[int] = Param(min=0, max=28)

    def step_params(self) -> dict:
        return {"atbash": {}, "caesar": {"shift": self.caesar_shift}}
```

Compose decrypt order **MUST** follow `steps` array order, matching
[transforms.md](transforms.md) `compose` (including per-stage direction
overrides where specified by referenced theories).

### Parameters

`Param[int] = Param(min=…, max=…)`:

- Bounds **MUST** be integers.
- For \(\mathbb{Z}_{29}\) value domains, bounds **MUST** lie in `0..28` unless
  documented otherwise (e.g. affine multiplier `a` in `1..28`).
- Domain violations relative to `z29_inv` **MUST** be rejected at compile or
  verify time (hard fail).

### Tier obligations

| Tier | `structural_claim() -> str` |
|------|------------------------------|
| `A` | Optional |
| `B` | **Required**, non-empty |
| `C` | **Required**, non-empty |

`tier` is an **author declaration** (research/process stance toward LP2), **not**
a verify outcome. `parcae-compile` **MUST** persist the declared `tier` into the
artifact and **MUST NOT** promote/demote it because exhaustive/fuzz verify
passed. Totality / determinism / CPU↔CUDA mirror live under
`verification.{mode,passed,…}` — a separate manifest object. Catalog/tools
**MUST NOT** treat `verification.passed` as evidence that a theory is Tier A.

Mis-tiering (e.g. claiming `A` without meeting Tier-A research bar in
[hypotheses.md](../research/hypotheses.md)) is a research/process concern; the
compiler **MUST** still enforce the structural claim presence rules above.
Silent downgrade or warning-only acceptance **MUST NOT** occur for missing
required claims.

### Interrupt obligations

| Situation | Requirement |
|-----------|-------------|
| `family == "elementwise"` | `interrupt_policy` **MAY** be omitted |
| `@Theory(interrupts="none_by_design")` | Explicit opt-out; no policy method required |
| `keyed_stream` / other interrupt-relevant families without the opt-out | **MUST** define `interrupt_policy(plaintext_rune: int) -> bool` |
| Raising `NotImplementedError` inside `interrupt_policy` | **MUST** fail compile (forces an explicit design choice) |

When `interrupts="none_by_design"`, runtime CPU apply (`DslIrApplicator` /
emitted Transform) **MUST** hard-reject a non-empty `InterruptPolicy` (Status
error). Emitted CUDA twins for that mode **MUST NOT** apply skip indices
either — empty policy only. This keeps CPU↔CUDA parity when agents pass a
default interrupt object.

DSL `interrupt_policy` is a value predicate for generated kernels. Bridge export
to fixture `explicit_skip_indices_v0` ([interrupts.md](interrupts.md)) **MAY**
be emitted when cleartext/fixture indices are known; tools **MUST NOT** invent
skip lists silently.

### Families (v0)


| Family | Required methods | Device notes |
|--------|------------------|--------------|
| `elementwise` | `encrypt_step` / `decrypt_step` | Single-stream + batch apply |
| `keyed_stream` | `keystream_at` + encrypt/decrypt steps + interrupt rules above | Keystream host-precompute allowed when not closed under primitives |
| `compose` (via `ComposedTheory`) | `step_params()` | Fusion pass or staged fallback |
| `keyed_permutation` | `derive_permutation` (host) + encrypt/decrypt steps | Device sees materialized tables only |

Host-only helpers (e.g. `RuneStream.rank_by_frequency`) **MUST** be deterministic
(stable tie-break). CUDA **MUST NOT** re-derive unstable orderings on device.

---

## Verification gates (compile-time, hard fail)

Before an artifact is written, the compiler **MUST** verify each new primitive
(and theory wiring that depends on it):

| Gate | Rule |
|------|------|
| Totality | Every sample result ∈ `0..28` |
| Determinism | Double evaluation identical; no hidden state |
| CPU ↔ CUDA mirror | IR CPU eval equals CUDA op-sequence mirror for every sample |
| Mode | Arity ≤ 4 \(\mathbb{Z}_{29}\) parameters: **full enumeration** (\(29^n\), e.g. \(29^4 = 707281\)). Larger: randomized property fuzz with fixed seed `0xC1CADA` |

Failure **MUST** abort compilation (non-zero exit). Conforming tools
**MUST NOT** offer “compile with warnings” / `--allow-unverified` that writes a
ready artifact.

Testing decorators in `parcae.dsl.testing` (`primitive_exhaustive_test`,
`property_test`, `fixture_test`, `negative_control`, `sweep_config`, …) declare
intent; the compiler **MAY** treat them as metadata. Stub execution of those
decorators **MUST** fail-loud (see IDE stubs).

---

## Diagnostics

User-facing compile errors **MUST**:

1. Include `path:line:col` when location is known.
2. Include a stable `rule_id` (e.g. `E013`, `E021`, `E030`).
3. Prefer a short actionable hint over compiler/CPython stack traces as the
   primary message.

Examples (informative):

```text
theories/x.py:42:4: E013 tier B requires structural_claim()
theories/x.py:3:1: E021 import 'numpy' not in parcae.dsl.* whitelist
theories/x.py:88:4: E030 interrupt_policy missing; set interrupts='none_by_design' if intentional
```

---

## Compose fusion (normative behavior)

For `ComposedTheory` chains the compiler **SHOULD** attempt a fused kernel
(no host round-trip between stages).

| Outcome | Requirement |
|---------|-------------|
| Fused throughput ≥ staged path (same benchmark protocol as toolkit compose suites) | Manifest `fusion.status = "fused"` |
| Fused &lt; staged | Manifest `fusion.status = "fallback_staged"`; use staged ping-pong compose; **MUST NOT** fail compile solely for this |

Verification gates remain hard regardless of fusion status.

---

## Non-goals

- Arbitrary Python as primitive bodies
- CPython inside exhaustive/fuzz verify loops
- Replacing locked fixture oracles with DSL scores
- Closed-loop search/agent scheduling (separate tooling)
- Fast-math or nondeterministic score reductions in emitted kernels
