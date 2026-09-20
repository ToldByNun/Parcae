# parcae-dsl (IDE stubs only)

Python package providing `parcae.dsl.*` for **editor autocomplete** when
authoring theory sources.

**This package is not the compiler.** It does not verify \(\mathbb{Z}_{29}\)
totality, does not emit CUDA/CPU twins, and does not write
`parcae://theories/…` artifacts. The only authoritative path is:

```text
parcae-compile path/to/theory.py
```

Operator guide: [`docs/architecture/dsl-stubs.md`](../docs/architecture/dsl-stubs.md).

| Action | Behavior |
|--------|----------|
| `import parcae.dsl…` | OK |
| `@Theory` / `@define_primitive` / … at definition time | Attaches `__parcae_dsl_stub__`; no IR |
| Calling primitives, Z29Expr ops, `apply`, test runners | Raises `ParcaeDslStubError` |

## AST dump frontend (syntax only)

One-shot CPython `ast.parse` → JSON (still **not** verification):

```text
python -m parcae.dsl.ast_dump path/to/theory.py
# or: parcae-dsl-ast-dump path/to/theory.py
```

Emits `parcae.dsl_ast_json.v0` on stdout ([`docs/spec/dsl-ast-json.md`](../docs/spec/dsl-ast-json.md)).
Exit `0` on success, `1` on failure envelope (`ok: false`).

A green dump only means the file parsed as Python. DSL rules and math gates run
inside `parcae-compile`.

## Docs

| Doc | Role |
|-----|------|
| [`docs/architecture/dsl-stubs.md`](../docs/architecture/dsl-stubs.md) | Stubs vs compiler (read this first) |
| [`docs/spec/dsl.md`](../docs/spec/dsl.md) | Normative language + stub MUST/MUST NOT |
| [`docs/architecture/python-transpiler.md`](../docs/architecture/python-transpiler.md) | C++20 compiler architecture |

## Install (dev)

```text
cd python
pip install -e ".[dev]"
pytest -q
```
