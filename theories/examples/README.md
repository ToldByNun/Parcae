# Theory examples (`theories/examples/`)

Authoring sources for the theory DSL. Normative language:
[`docs/spec/dsl.md`](../../docs/spec/dsl.md). Architecture:
[`docs/architecture/python-transpiler.md`](../../docs/architecture/python-transpiler.md).

IDE stubs under `python/parcae/dsl/` are **not** a compiler — they raise
`ParcaeDslStubError` on semantic calls. Compile with:

```text
parcae-compile theories/examples/<file>.py
```

Artifacts land under `data/theories/<name>/<version>/`
([theory-artifact.md](../../docs/spec/theory-artifact.md)).

| File | Theory URI (after compile @1) | Notes |
|------|-------------------------------|-------|
| `new_math_example.py` | `parcae://theories/quadratic_polynomial_stream@1` | Tier B keyed_stream; `poly2_mod29` arity 4 |
| `full_lifecycle_example.py` | `parcae://theories/koan1_style@1` | Tier A `@ComposedTheory` (atbash+caesar); fuse/staged emit |
