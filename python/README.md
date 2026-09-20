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

Normative: [`docs/spec/dsl.md`](../docs/spec/dsl.md) § IDE stubs.  
Architecture: [`docs/architecture/python-transpiler.md`](../docs/architecture/python-transpiler.md).

## Install (dev)

```text
cd python
pip install -e ".[dev]"
pytest -q
```
