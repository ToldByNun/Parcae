# Parcae release notes

Canonical notes for toolkit SemVer cuts. Packaging filenames use `X.Y.Z` from
CMake (milestone suffixes on git tags are **not** in artifact names). See
[`docs/architecture/release.md`](../docs/architecture/release.md).

---

# Parcae 0.8.0 — Smart DSL + console progress

**Tag:** `v0.8.0-dsl-console`  
**Toolkit:** 0.8.0  
**Date:** 2026-09-24

## Highlights

Combined milestone: a **scope-aware theory DSL smart compiler** plus **human
console progress** on search CLIs.

- **OuterControl vs HotLoop** — host setup vs kernel math; HotLoop `for`/`while`
  → **E034**; unbounded OuterControl `while` → **E035**
- **Param / const HotLoop `if` → `Z29Expr::Select`** (branch-free mux); thread-
  varying predicates → **E033**; relaxed accepts → **W011**
- **`#ignore DSL_FLAG:…`** (research only) via `ast_dump` + `DslDirectiveTable`;
  requires `--allow-dsl-ignores` (**W010**); denied without the flag → **E031**
- **`ConsoleDashboard`** live stderr progress on `parcae-search-cycle`, shared
  vocabulary on `SearchRunConsole` / BlindCrack; JSON stdout stays clean
- **Agent path:** ToolBridge injects `--quiet` on `search_cycle` so transcripts
  stay free of panel noise; digests identical with progress on vs off

`parcae.corpus.load_page` (Liber Primus page helper with solved/locked warning)
is **deferred** — not part of this cut.

Exit freeze checklist:
[`docs/architecture/dsl-console-exit.md`](../docs/architecture/dsl-console-exit.md).

## What's new since 0.7.0

### Smart DSL compiler
- Normative scopes in [`docs/spec/dsl.md`](../docs/spec/dsl.md)
- `DslScopeAnalyzer`, scope-aware `DslSemanticGate`, `DslDivergenceGate`,
  `DslHostGlue` / `HostGlueIr`, `DslDirectiveTable`
- HotLoop relaxed `if` → Select in `DslBuildIr`; CPU/CUDA emit for Select
- `dsl_ast_json_version` **1.1.0** (directives wire format)
- Examples: `theories/examples/param_select_example.py` (portable);
  `ignore_divergent_example.py` (research / `--allow-dsl-ignores`)
- CI: **`Gate [dsl-smart]`** + existing `[dsl][examples]` /
  `[dsl][registry][stale]`

### Console progress
- Shared `ConsoleProgressSnapshot` / `ConsoleDashboard` / ANSI helpers
- Flags: `--quiet` > `--plain-progress` > `--progress auto|panel|lines|off`
- Digest invariance: `[tool][search_cycle][progress][determinism]`
- CI: **`Gate [cli-progress]`**

### Docs
- [`docs/architecture/python-transpiler.md`](../docs/architecture/python-transpiler.md)
  § Execution scopes
- [`docs/architecture/search-handbook.md`](../docs/architecture/search-handbook.md)
  § Console progress
- [`docs/architecture/cuda-build.md`](../docs/architecture/cuda-build.md)
  Catch2 tag map for smart DSL + dashboard

## Packaging matrix (0.8.0)

Filenames use toolkit SemVer **0.8.0** (tag `v0.8.0-dsl-console` is not in the
name):

| Pattern | Flavor |
|---------|--------|
| `Parcae-v0.8.0-windows-x64{-cpu,-cuda,}.exe` | Windows installers |
| `Parcae-v0.8.0-linux-x86_64{-cpu,-cuda,}.tar.gz` | Linux tarballs |
| `Parcae-v0.8.0-linux-x86_64{-cpu,-cuda,}.AppImage` | AppImage |
| `parcae_0.8.0_amd64{_cpu,_cuda,}.deb` | Debian |
| `parcae-0.8.0-1.x86_64{.cpu,.cuda,}.rpm` | RPM |
| `parcae-0.8.0-source.tar.gz` | Source |
| `SHA256SUMS` | Digests |

## Install / smoke

Same packaging flow as 0.7.0 ([`release.md`](../docs/architecture/release.md)).

After the SemVer bump lands:

```text
parcae-search-cycle --status --json   →  "toolkit_version":"0.8.0"
parcae-compile --status --json        →  same toolkit version
```

Quiet vs panel: `--quiet` (agent) · `--plain-progress` (CI logs) · default TTY panel.

## Docs

- [`docs/architecture/dsl-console-exit.md`](../docs/architecture/dsl-console-exit.md)
- [`docs/architecture/python-transpiler.md`](../docs/architecture/python-transpiler.md)
- [`docs/architecture/search-handbook.md`](../docs/architecture/search-handbook.md)
- [`docs/architecture/release.md`](../docs/architecture/release.md)

---

# Parcae 0.7.0 — Search-engine exit

**Tag:** `v0.7.0-search-engine`  
**Toolkit:** 0.7.0  
**Date:** 2026-09-23

Closed-loop Liber Primus **search engine**: workspace ciphertext → GPU/CPU
candidate export → `BatchArtifact` → `HypothesisBridge` → `SearchPrior` → next
cycle (`SearchScheduler` / `parcae-search-cycle`). Agents use allow-listed
`search_cycle`; packaging matrix used `Parcae-v0.7.0-…` artifacts.

Details: [`docs/architecture/search-engine.md`](../docs/architecture/search-engine.md).
