# Toolkit exit: bench & diagnostics

**Status:** Exit freeze — toolkit **0.9.0** / `parcae-bench` (tag cut is user-owned)  
**Exit tag (locked):** `v0.9.0-bench`  
**Toolkit SemVer (locked):** `0.9.0`

**Upstream:** DSL console (`v0.8.0-dsl-console`) + search engine (`v0.7.0-search-engine`)  
**Themes:** umbrella bench CLI (SLO / accuracy / hardware / probe), deterministic
`Gate [bench]`, agent deny-list for timing diagnostics  
**Cut procedure:** [`release.md`](release.md)  
**Operator handbook:** [`bench-diagnostics.md`](bench-diagnostics.md)

```text
parcae-bench --suite slo|accuracy|hardware|probe|all
parcae-throughput-tiers  ≡  --suite slo --extended --allow-cuda
```

---

## Locked decisions

| Topic | Decision |
|-------|----------|
| Exit tag | `v0.9.0-bench` (toolkit **0.9.0**) |
| Version order | `0.8 dsl-console` → **`0.9 bench`** |
| C++ style (HARD) | No namespaces; one top-level class per header; `#ifndef` / `#endif` guards |
| Naming | Descriptive kebab paths only — **no** “phase*” filenames, tags, or CI job names |
| Agent git | User owns commits, annotated tags, and release pushes |
| Absolute SLO in hosted CI | **Never** — floors live in `BenchTierSpec`; local/CUDA only |
| Probe wire format | JSON **1.0.0** (`docs/spec/bench-probe.md`); spawn timeout default **120s** |
| Agent policy | `bench` / `parcae-bench` (+ throughput-tiers aliases) **deny-listed** |
| Hosted gate filter | `[bench][spec],[bench][probe],[bench][accuracy],[bench][report]` only |

Architecture companions: [`bench-diagnostics.md`](bench-diagnostics.md),
[`cuda-throughput.md`](cuda-throughput.md),
[`include/parcae/bench/README.md`](../../include/parcae/bench/README.md).

---

## Exit criteria → `v0.9.0-bench`

Declare toolkit **0.9.0** complete when all **required** boxes below are green
(then the user cuts the annotated tag).

### Core library & report

- [x] `BenchTierSpec` T1–T3 floors + practical peaks; Catch2 `[bench][spec]`
- [x] `BenchTimer` / `BenchMetric` median-of-3 (CPU + CUDA paths)
- [x] `BenchReport` + `BenchFormatter` (human + JSON + `--omit-timing`)
- [x] Catch2 `[bench][report]` digest invariance

### Suites

- [x] **slo** — fused CUDA T1–T3 (+ `--extended` F.*/C.*); local/CUDA only
- [x] **accuracy** — fixture eval / χ² / oracle; optional CUDA planted/parity
- [x] **hardware** — CPU vs GPU same tier IDs; `--require-cuda` / `--allow-skip`
- [x] **probe** — Protocol 1.0.0 + Runner (`{tier}` spawn); `[bench][probe]`
- [x] **all** — suite order accuracy → slo → hardware → probe

### CLI & compat

- [x] `parcae-bench` CLI (`--status`, `--suite`, `--json`, `--omit-timing`, …)
- [x] `parcae-throughput-tiers` thin wrapper ≡ slo + extended + allow-cuda
- [x] Tools / agent-tools / bench README suite docs
- [x] Spec: [`docs/spec/bench-probe.md`](../spec/bench-probe.md) + tools.md § bench

### Policy & gates

- [x] `AgentPolicy` + `allowlist.py` deny `bench` / `parcae-bench`
- [x] Hosted **`Gate [bench]`** filter (no absolute SLO / hardware / timer)
- [x] Tag map documented in [`cuda-build.md`](cuda-build.md) / handbook

### Release cut

- [ ] Root README roadmap marks **bench** done with exit tag
- [ ] [`RELEASE_NOTES.md`](../../release/RELEASE_NOTES.md) for 0.9.0
- [ ] Toolkit version **0.9.0** (CMake + `version.hpp` + smoke asserts)
- [ ] Smoke: `Version` + `parcae-bench --status` → toolkit **0.9.0**
- [ ] Annotated tag `v0.9.0-bench`; Release workflow publishes artifacts

---

## Out of scope (explicit)

- Absolute runes/s gates on hosted GitHub runners
- Agent-invoked bench / timing SLOs via ToolBridge
- Recalibrating CUDA practical peaks on every PR (operator + `cuda-throughput.md`)
- Replacing Catch2 accuracy/parity tests with probe-only coverage
- Agent creating/pushing annotated tags or GitHub release assets

---

## Document history

Freeze checklist for the `parcae-bench` milestone. Operator recipes stay in
[`bench-diagnostics.md`](bench-diagnostics.md); packaging steps in
[`release.md`](release.md). Prior exit pattern:
[`dsl-console-exit.md`](dsl-console-exit.md).
