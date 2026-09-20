# parcae-dsl (IDE stubs only)

Python package providing `parcae.dsl.*` for **editor autocomplete** when
authoring theory sources. This is **not** the compiler.

| Action | Behavior |
|--------|----------|
| `import parcae.dsl…` | OK |
| `@Theory` / `@define_primitive` / … at definition time | Attaches `__parcae_dsl_stub__`; no IR |
| Calling primitives, Z29Expr ops, `apply`, test runners | Raises `ParcaeDslStubError` |

Compile and verify with:

```text
parcae-compile path/to/theory.py
```

## AST dump frontend (B7)

One-shot CPython syntax frontend (no `exec`):

```text
python -m parcae.dsl.ast_dump path/to/theory.py
# or: parcae-dsl-ast-dump path/to/theory.py
```

Emits `parcae.dsl_ast_json.v0` on stdout (see `docs/spec/dsl-ast-json.md`).
Exit `0` on success, `1` on failure envelope (`ok: false`).

Normative: [`docs/spec/dsl.md`](../docs/spec/dsl.md) § IDE stubs.  
Architecture: [`docs/architecture/python-transpiler.md`](../docs/architecture/python-transpiler.md).

## Install (dev)

```text
cd python
pip install -e ".[dev]"
pytest -q
```
