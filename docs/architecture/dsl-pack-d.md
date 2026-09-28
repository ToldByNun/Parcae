# DSL Pack D — language MINOR 1.1.0

**Status:** Exit freeze companion for toolkit **1.1.0** / tag `v1.1.0`  
**Language SemVer (locked):** `dsl_spec_version` **`1.1.0`** (MINOR; MAJOR stays **1**)  
**Umbrella exit:** [`cuda-catalog-parity.md`](cuda-catalog-parity.md)

**Upstream:** Smart DSL + console cut (`v0.8.0-dsl-console`) deferred Pack D and
`parcae.corpus.load_page` on purpose — see [`dsl-console-exit.md`](dsl-console-exit.md).  
**Theme:** small authoring-surface pack — high leverage, no dumps/cache stretch  
**Cut procedure:** [`release.md`](release.md) (same tag as catalog CUDA parity)

```text
ast_dump → ingest → DirectiveTable
  → SemanticGate / DivergenceGate / HostGlue   # multi-diag collects
  → DslBuildIr
  → DslOptimize                                # required in DslCompile
  → DslVerifier
  → DslEmitCpu / DslEmitCuda                   # honor inv hoists
  → TheoryArtifact  (dsl_spec_version 1.1.0)
```

---

## Locked decisions

| Topic | Decision |
|-------|----------|
| Ships on | Toolkit **1.1.0** / `v1.1.0` (with CUDA catalog parity) |
| `dsl_spec_version` | **`1.1.0`** — additive language/tooling; older **1.0.0** artifacts remain loadable |
| `dsl_ast_json_version` | Unchanged **`1.1.0`** |
| C++ style (HARD) | No namespaces; one top-level class per header; `#ifndef` / `#endif` guards |
| Naming | No “phase*” paths or titles |
| Agent git | User owns commits / tags / release pushes |
| Required Pack D items | Multi-diag, `z29_match`, `DslOptimize` in `DslCompile`, `parcae.corpus.load_page`, `dsl_spec` **1.1.0** |
| Explicitly deferred | `--dump-ir` / `--dump-scopes`, IR source maps, `--strict-portable`, testing-decorator verify schedules, Param `where=`, OuterControl compose step toggles, compile cache, richer E033 counterexamples |

Architecture companions: [`python-transpiler.md`](python-transpiler.md),
[`dsl-stubs.md`](dsl-stubs.md), [`dsl-console-exit.md`](dsl-console-exit.md).  
Headers: [`include/parcae/dsl/`](../../include/parcae/dsl/).  
Python: [`python/parcae/`](../../python/parcae/).

---

## Required features (why)

| Feature | Problem today | Locked design |
|---------|---------------|---------------|
| **Multi-diagnostic compile** | Gates fail on the first `Status::error` | `DslDiagList` accumulates errors; CLI prints all; JSON `diagnostics[]`; any error → non-zero exit, **no** ready artifact. Warnings (W010/W011) never alone fail. |
| **`DslOptimize` in `DslCompile`** | Pass exists (`[dsl][optimize]`) but compile emits unoptimized IR | After `DslBuildIr`, before verify/emit: `optimize_theory` / body optimize; emit writes `__parcae_inv_N` prelude; verifier binds hoists into eval env. |
| **`z29_match`** | Only binary `Select` / `z29_select` | Builtin: `z29_match(x, ((c0,e0), …), default)` lowers to cascaded `Z29Expr::select` (HotLoop-safe, branch-free). Stub fail-loud; BuildIr / optimize / emit / verify. |
| **`parcae.corpus.load_page`** | Deferred Python helper; C++ `WorkspaceCipher::load_page` already exists for search | Pure Python corpus API over `data/`; LP2 index map; warn on locked/solved fixtures unless `allow_plaintext=True`. |
| **`dsl_spec_version` 1.1.0** | Still **1.0.0** | Bump compiler + [`dsl.md`](../spec/dsl.md); stamp new artifacts; registry keeps MAJOR-mismatch reject and older-MINOR accept. |

### `z29_match` semantics (locked)

```python
z29_match(x, ((c0, e0), (c1, e1), ...), default)
# → select(x == c0, e0, select(x == c1, e1, … default))
```

Malformed arms → **E032** (or documented sibling rule). Const scrutinee / equal arms fold via existing `DslOptimize` Select passes.

### `load_page` semantics (locked)

| `page` | Resolution |
|--------|------------|
| `0..55` | Prefer `data/workspaces/lp2-page-{page}-explore/` ciphertext when present |
| `56` | Fixture `an-end` |
| `57` | Fixture `lp2-57-identity` |
| else | `ValueError` |

Locked/solved fixture → `warnings.warn(...)` unless `allow_plaintext=True`. Default return is ciphertext-oriented (plaintext only when opted in). Distinct from C++ `WorkspaceCipher::load_page` (search workspace `pages/NN.txt`).

---

## Exit criteria → Pack D on `v1.1.0`

All boxes required to cut toolkit **1.1.0** (same tag as
[`cuda-catalog-parity.md`](cuda-catalog-parity.md)).

### Multi-diag

- [ ] `DslDiagList` (or equivalent class) collects gate errors without stopping at the first
- [ ] `DslCompile` / `parcae-compile` report all errors; JSON includes `diagnostics[]`
- [ ] Any error → non-zero exit; **no** ready theory artifact
- [ ] Catch2: multi-error source produces ≥2 diagnostics in one compile response

### Optimize in compile

- [ ] `DslCompile` calls `DslOptimize` after BuildIr (theories / primitives as applicable)
- [ ] `DslEmitCpu` / `DslEmitCuda` emit hoist temps for `__parcae_inv_N`
- [ ] `DslVerifier` (or apply path) binds hoist env so optimized bodies verify
- [ ] Catch2: compile path shows const-fold and/or inv-hoist in emitted text
      (`[dsl][compile]` / `[dsl][optimize]` integration)

### `z29_match`

- [ ] Listed in `DslZ29Builtins` + normative [`dsl.md`](../spec/dsl.md)
- [ ] Python stub fail-loud in `parcae.dsl.math`
- [ ] BuildIr → cascaded `Select`; optimize / emit / applicator / verifier green
- [ ] Example under `theories/examples/` (e.g. `match_select_example.py`)
- [ ] Catch2 `[dsl][match]` (or documented sibling tags)

### `parcae.corpus.load_page`

- [ ] Package `python/parcae/corpus/` with `load_page` / `LoadedPage` / LP2 map
- [ ] Locked-fixture warning contract
- [ ] Pytest coverage for resolution + warning
- [ ] Docs cross-links: [`search-handbook.md`](search-handbook.md),
      [`python-transpiler.md`](python-transpiler.md), [`dsl-stubs.md`](dsl-stubs.md)

### Language version

- [ ] `DslSpecVersion::current_string == "1.1.0"`
- [ ] [`docs/spec/dsl.md`](../spec/dsl.md) header **1.1.0**
- [ ] `parcae-compile --status` reports `dsl_spec_version` **1.1.0**
- [ ] Registry/tests: `1.0.0` artifacts still compatible; MAJOR mismatch still rejected
- [ ] Example theories recompiled / stamped as needed for CI

---

## Out of scope (explicit)

- `--dump-ir` / `--dump-scopes` / IR source maps / `--strict-portable`
- Wiring `@property_test` / `@fixture_test` / … into `DslVerifier` schedules
- Param `where=` / dependent domains; OuterControl compose step toggles at launch
- Compile cache (`--cache`); LSP beyond fail-loud stubs
- Replacing hand CUDA twins with DSL-only kernels
- Agent git / tags / GitHub release assets

---

## Document history

Pack D was named as post-`v0.8.0-dsl-console` language work in
[`dsl-console-exit.md`](dsl-console-exit.md). This document freezes the **required**
Pack D subset for toolkit **1.1.0**. Umbrella release checklist:
[`cuda-catalog-parity.md`](cuda-catalog-parity.md).
