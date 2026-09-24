# DSL test fixtures

Committed theory-artifact trees used by `[dsl][registry][stale]` tests and the
CI `dsl-examples-cli` job (`scripts/check-dsl-examples.sh`).

| Path | Purpose |
|------|---------|
| `theories/stale_major_demo/1/` | Manifest stamped `dsl_spec_version: 0.9.0` (pre-1.0 MAJOR). Must remain rejected by `TheoryRegistry` / `parcae-validate` while toolchain current is `1.x`. |
| `sources/quadratic_polynomial_stream.py` | Minimal `@define_primitive` + `@Theory` (Tier **B** + `structural_claim`) for `DslCompile` / `parcae-compile` end-to-end. |
| `sources/smart_select_param.py` | Param-uniform HotLoop `if` → Select mux (no ignore); `[dsl][golden]`. |
| `sources/smart_ignore_divergent.py` | `#ignore DSL_FLAG:divergent_branch` + ThreadVarying `if`; needs `--allow-dsl-ignores`. |

Normative policy: [`docs/spec/dsl.md`](../../../docs/spec/dsl.md) § `dsl_spec_version`, [`docs/spec/theory-artifact.md`](../../../docs/spec/theory-artifact.md).
