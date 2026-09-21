# DSL test fixtures

Committed theory-artifact trees used by `[dsl][registry][stale]` tests.

| Path | Purpose |
|------|---------|
| `theories/stale_major_demo/1/` | Manifest stamped `dsl_spec_version: 0.9.0` (pre-1.0 MAJOR). Must remain rejected by `TheoryRegistry` while toolchain current is `1.x`. |
| `sources/quadratic_polynomial_stream.py` | Minimal `@define_primitive` + `@Theory` (Tier **B** + `structural_claim`) for `DslCompile` / `parcae-compile` end-to-end. |

Normative policy: [`docs/spec/dsl.md`](../../../docs/spec/dsl.md) § `dsl_spec_version`, [`docs/spec/theory-artifact.md`](../../../docs/spec/theory-artifact.md).
