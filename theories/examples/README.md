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
| `matrix_builtins_example.py` | `matrix_mix_stream` + `atbash_then_matrix_mix` | `z29_det` / `z29_matmul` primitives; fuse DSL-only leaf |
| `param_select_example.py` | `parcae://theories/param_select_example@1` | Param-uniform `if` → `Select` mux (**W011**); CI-safe |
| `ignore_divergent_example.py` | `parcae://theories/ignore_divergent_example@1` | Research only — needs `--allow-dsl-ignores` (**W010**) |

### Smart-compiler scopes (short)

| Region | Typical code | Control flow |
|--------|--------------|--------------|
| **HotLoop** | `encrypt_step` / primitives | Prefer `Select` / Param flags; ThreadVarying `if` → **E033** |
| **OuterControl** | module / `step_params` | Bounded `for` / finite `while`; else **E035** |

`#ignore DSL_FLAG:…` (grammar in [dsl-ast-json.md](../../docs/spec/dsl-ast-json.md)
§ Directives) is **off by default**. Research theories that need it:

```text
parcae-compile theories/examples/ignore_divergent_example.py --allow-dsl-ignores
```

CI (`scripts/check-dsl-examples.sh`) compiles the portable examples only — not
`ignore_divergent_example.py`.