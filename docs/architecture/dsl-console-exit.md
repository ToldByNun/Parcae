# Toolkit exit: smart DSL + console progress

**Status:** Toolkit **0.8.0** cut in progress (README / CI gates landed; SemVer
bump + annotated tag still pending)  
**Exit tag (locked):** `v0.8.0-dsl-console`  
**Toolkit SemVer (locked):** `0.8.0`

**Upstream:** Theory DSL (`v0.5.0-theory-dsl`) + agent tools (`v0.6.0-agent-tools`) +
search engine (`v0.7.0-search-engine`)  
**Themes:** scope-aware smart compiler (OuterControl vs HotLoop) **and** human
console progress (`ConsoleDashboard` on search CLIs)  
**Cut procedure:** [`release.md`](release.md)

Combined milestone: authors get a portable HotLoop/`Select` language surface;
operators get live stderr progress without polluting agent JSON.

```text
theory.py  →  parcae-compile  →  ready theory URI
workspace  →  parcae-search-cycle [--quiet|panel|lines]  →  batches + hypotheses
```

---

## Locked decisions

| Topic | Decision |
|-------|----------|
| Exit tag | `v0.8.0-dsl-console` (toolkit **0.8.0**) |
| Version order | `0.5 theory-dsl` → `0.6 agent-tools` → `0.7 search-engine` → **`0.8 dsl-console`** |
| C++ style (HARD) | No namespaces; one top-level class per header; `#ifndef` / `#endif` guards |
| Naming | Descriptive kebab paths only — **no** “phase*” filenames, tags, or CI job names |
| Agent git | User owns commits, annotated tags, and release pushes |
| `dsl_spec_version` | Stays MAJOR **`1.0.0`** for this cut; language MINOR **`1.1.0`** is pack D (post-exit / optional on-tag) |
| `dsl_ast_json_version` | **`1.1.0`** (directives wire format) recorded in artifacts |
| `load_page` | **Deferred** — not required to cut `v0.8.0-dsl-console` |
| Pack D (multi-diag, dumps, `z29_match`, …) | Nice-to-have on the same tag if already green; **do not** block 0.8.0 on `dsl_spec` 1.1 |

Architecture companions: [`python-transpiler.md`](python-transpiler.md) (smart
compiler), [`search-handbook.md`](search-handbook.md) § Console progress,
[`search-engine.md`](search-engine.md) (prior exit pattern). Headers:
[`include/parcae/dsl/`](../../include/parcae/dsl/).

---

## Exit criteria → `v0.8.0-dsl-console`

Declare toolkit **0.8.0** complete when all **required** boxes below are green
(then the user cuts the annotated tag). Items marked *deferred* are documented
out of the cut on purpose.

### Smart DSL transpiler

- [x] OuterControl vs HotLoop normative in [`dsl.md`](../spec/dsl.md)
- [x] `DslScopeAnalyzer` + scope-aware `DslSemanticGate` (**E034**)
- [x] `Z29Expr` Select + applicator + optimize fold
- [x] `DslDivergenceGate` (**E033** / **W011**); HotLoop relaxed `if` → Select
- [x] OuterControl `for`/`while` via HostGlue (const unroll / bounded host); **E035**
- [x] `#ignore DSL_FLAG` via `ast_dump` + `DslDirectiveTable`; **W010**; `--allow-dsl-ignores`
- [x] CPU/CUDA emit for Select; ignored divergent `if` documented + warned
- [x] Catch2: `[dsl][scope]`, `[dsl][divergence]`, `[dsl][directive]` / `[dsl][directives]`, `[dsl][golden]`
- [x] `dsl_ast_json_version` **1.1.0** recorded in artifacts (directives)
- [x] Portable examples + ignore-denied CI (`scripts/check-dsl-examples.sh`,
      `theories/examples/param_select_example.py`)
- [ ] ~~`parcae.corpus.load_page` warns on locked/solved fixtures~~ — **deferred**
      (not required for this tag)

### Console progress / dashboard

- [x] `ConsoleDashboard` live path on `parcae-search-cycle` (stderr-only; JSON stdout clean)
- [x] Flags: `--quiet` > `--plain-progress` > `--progress auto|panel|lines|off`
- [x] ToolBridge injects `--quiet` on `search_cycle` cycle runs
- [x] Digest invariance: progress on vs `--quiet`
      (`[tool][search_cycle][progress][determinism]`)
- [x] VT panel → lines fallback covered (`[cli][dashboard]`)
- [x] `SearchRunConsole` / BlindCrack human reports share dashboard vocabulary
- [x] Progress contract docs in tools / search-loop / search-handbook / agent-handbook

### Gates & release cut

- [x] Hosted CI gates for smart-dsl + cli-progress tags
      (`Gate [dsl-smart]` →
      `"[dsl][scope],[dsl][divergence],[dsl][directive],[dsl][directives]"`;
      `Gate [cli-progress]` →
      `"[cli][dashboard],[tool][search_cycle][progress]"`;
      documented in [`cuda-build.md`](cuda-build.md) § Catch2 tags)
- [x] Root README roadmap marks **dsl-console** done with exit tag
- [ ] [`RELEASE_NOTES.md`](../../release/RELEASE_NOTES.md) for 0.8.0
- [ ] Toolkit version **0.8.0** (CMake + `version.hpp` + smoke asserts)
- [ ] Smoke: `Version` + `parcae-search-cycle --status` /
      `parcae-compile --status` → toolkit **0.8.0**
- [ ] Annotated tag `v0.8.0-dsl-console` (**user**); Release workflow publishes
      `Parcae-v0.8.0-…` artifacts

**Nice-to-have on the same tag** (language/tooling pack; do not block):
multi-diagnostic compile, `--dump-ir` / `--dump-scopes`, IR source maps,
`--strict-portable`. Language MINOR `dsl_spec_version` **1.1.0** may follow under
toolkit 0.8.x.

---

## Out of scope (explicit)

- Replacing hand CUDA Tier-A twins with DSL-only kernels
- Full Python runtime inside CUDA kernels / arbitrary OuterControl I/O
- Numeric LP1 page indices beyond a future documented LP2 map (`load_page`)
- Agent creating/pushing annotated tags or GitHub release assets
- Reworking the historical `v0.7.0-search-engine` exit narrative
- Treating toolkit **0.8.0** as a `dsl_spec_version` **MAJOR** bump

---

## Document history

Freeze checklist for the combined smart-compiler + console-dashboard milestone.
Engineering narrative stays in [`python-transpiler.md`](python-transpiler.md) and
[`search-handbook.md`](search-handbook.md); packaging steps in
[`release.md`](release.md).
