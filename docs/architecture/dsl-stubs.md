# Theory DSL stubs vs compiler

**Status:** Operator / contributor guide  
**Normative stubs rules:** [`docs/spec/dsl.md`](../spec/dsl.md) § IDE stubs  
**Compiler architecture:** [`python-transpiler.md`](python-transpiler.md)  
**Package:** [`python/`](../../python/) (`parcae.dsl.*`)

## One-sentence rule

**Only `parcae-compile` compiles and verifies a theory.** Importing or running
`parcae.dsl` stubs never produces a verified artifact and must not be treated as
proof that a theory is correct.

## What each piece does

| Piece | Role | Verifies math? | Writes `parcae://theories/…`? |
|-------|------|----------------|-------------------------------|
| `parcae.dsl.*` stubs | Editor autocomplete + importable markers | **No** (fail-loud on semantic calls) | **No** |
| `python -m parcae.dsl.ast_dump` | One-shot CPython `ast.parse` → JSON | **No** (syntax only) | **No** |
| `parcae-compile` | C++ pipeline: ingest → IR → verify → emit | **Yes** (hard gates) | **Yes** (when pipeline is ready) |

```text
theory.py  ──import──►  parcae.dsl stubs     (IDE only; calls raise)
     │
     ├── ast_dump ──►  dsl_ast_json.v0       (syntax wire format)
     │
     └── parcae-compile ──►  data/theories/… (only verified path)
```

## Common mistakes

1. **“It imported, so it works.”**  
   Decorators may attach `__parcae_dsl_stub__` at definition time. That is not
   verification.

2. **“I ran the test function / called `decrypt_step`.”**  
   Stubs **MUST** raise `ParcaeDslStubError` and tell you to run
   `parcae-compile`. If a call appears to succeed, the stub package is broken —
   report it; do not trust the result.

3. **“`ast_dump` succeeded, so the theory is valid.”**  
   A successful dump means the file parsed as Python and serialized. It does
   **not** mean the DSL subset, tiers, interrupts, or \(\mathbb{Z}_{29}\) gates
   passed. Those gates run only in the C++ compiler.

4. **“I can use stub `Z29Expr` as a calculator.”**  
   Forbidden. Stubs must not build executable IR or evaluate \(\mathbb{Z}_{29}\)
   arithmetic.

## What to run

```text
# IDE / authoring (optional editable install)
cd python && pip install -e ".[dev]"

# Stub runtime contract (must raise ParcaeDslStubError)
cd python && pytest -m ci -q

# Syntax dump only (for debugging the wire format)
python -m parcae.dsl.ast_dump path/to/theory.py

# Authoritative compile + verify (C++ CLI)
parcae-compile --status --json
parcae-compile path/to/theory.py
```

Stub unit tests live in [`python/tests/test_stubs_fail_loud.py`](../../python/tests/test_stubs_fail_loud.py).
If a semantic call returns without raising, the stub package is broken.

## Spec anchors

| Topic | Doc |
|-------|-----|
| Fail-loud stub obligations | [dsl.md](../spec/dsl.md) |
| AST JSON protocol | [dsl-ast-json.md](../spec/dsl-ast-json.md) |
| Ready artifacts only after verify | [theory-artifact.md](../spec/theory-artifact.md) |
| `parcae-compile` CLI | [tools.md](../spec/tools.md) § `parcae-compile` |
