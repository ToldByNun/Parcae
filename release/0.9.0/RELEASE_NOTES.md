# Parcae release notes

Canonical notes for toolkit SemVer cuts. Packaging filenames use `X.Y.Z` from
CMake (milestone suffixes on git tags are **not** in artifact names). See
[`docs/architecture/release.md`](../../docs/architecture/release.md).

---

# Parcae 0.9.0 — Bench & diagnostics

**Tag:** `v0.9.0-bench`  
**Toolkit:** 0.9.0  
**Date:** 2026-09-25

## Highlights

Umbrella **verification + diagnostics** CLI (`parcae-bench`) for fused CUDA SLO
tiers, statistical accuracy checks, CPU↔GPU hardware compare, and optional
external JSON probes — without absolute runes/s gates on hosted CI.

- **`parcae-bench`** suites: `slo` | `accuracy` | `hardware` | `probe` | `all`
- **`BenchTierSpec`** T1–T3 floors + practical peaks; median-of-3 timing
- **Probe wire format 1.0.0** (`--probe-cmd` with `{tier}`; spawn timeout 120s)
- **Compat:** `parcae-throughput-tiers` ≡ `--suite slo --extended --allow-cuda`
- **Agent deny-list:** `bench` / `parcae-bench` (+ throughput-tiers aliases)
- **Hosted `Gate [bench]`:** `[bench][spec|probe|accuracy|report]` only

Exit freeze checklist:
[`docs/architecture/bench-exit.md`](../../docs/architecture/bench-exit.md).  
Operator handbook:
[`docs/architecture/bench-diagnostics.md`](../../docs/architecture/bench-diagnostics.md).

## What's new since 0.8.0

### Core library & report
- `BenchTierSpec`, `BenchTimer` / `BenchMetric`, `BenchReport` + `BenchFormatter`
  (human + JSON + `--omit-timing`)
- Catch2: `[bench][spec]`, `[bench][report]`, `[bench][probe]`, `[bench][accuracy]`

### Suites & CLI
- Accuracy: fixture eval / χ² / oracle (+ optional CUDA planted/parity)
- Hardware: same tier IDs CPU vs CUDA; `--require-cuda` / `--allow-skip`
- Probe: Protocol 1.0.0 + Runner; docs [`docs/spec/bench-probe.md`](../../docs/spec/bench-probe.md)
- `parcae-bench --status` / `--suite` / `--json` / `--omit-timing`

### Policy & gates
- `AgentPolicy` + `allowlist.py` deny timing diagnostics
- Hosted **`Gate [bench]`** (no absolute SLO / hardware / timer in the filter)

### Docs
- [`docs/architecture/bench-diagnostics.md`](../../docs/architecture/bench-diagnostics.md)
- [`docs/architecture/bench-exit.md`](../../docs/architecture/bench-exit.md)
- Suite notes in tools / agent-tools / `include/parcae/bench/README.md`

## Packaging matrix (0.9.0)

Filenames use toolkit SemVer **0.9.0** (tag `v0.9.0-bench` is not in the name):

| Pattern | Flavor |
|---------|--------|
| `Parcae-v0.9.0-windows-x64{-cpu,-cuda,}.exe` | Windows installers |
| `Parcae-v0.9.0-linux-x86_64{-cpu,-cuda,}.tar.gz` | Linux tarballs |
| `Parcae-v0.9.0-linux-x86_64{-cpu,-cuda,}.AppImage` | AppImage |
| `parcae_0.9.0_amd64{_cpu,_cuda,}.deb` | Debian |
| `parcae-0.9.0-1.x86_64{.cpu,.cuda,}.rpm` | RPM |
| `parcae-0.9.0-source.tar.gz` | Source |
| `SHA256SUMS` | Digests |

## Install / smoke

Same packaging flow as 0.8.0 ([`release.md`](../../docs/architecture/release.md)).

After the SemVer bump lands:

```text
parcae-bench --status --json          →  "toolkit_version":"0.9.0"
parcae-search-cycle --status --json   →  same toolkit version
parcae-compile --status --json        →  same toolkit version
```

Absolute SLO / CUDA hardware compare remain **local/operator** recipes — see
[`bench-diagnostics.md`](../../docs/architecture/bench-diagnostics.md).

## Docs

- [`docs/architecture/bench-exit.md`](../../docs/architecture/bench-exit.md)
- [`docs/architecture/bench-diagnostics.md`](../../docs/architecture/bench-diagnostics.md)
- [`docs/spec/bench-probe.md`](../../docs/spec/bench-probe.md)
- [`docs/architecture/release.md`](../../docs/architecture/release.md)

---

# Parcae 0.8.0 — Smart DSL + console progress

**Tag:** `v0.8.0-dsl-console`  
**Toolkit:** 0.8.0  
**Date:** 2026-09-24

Scope-aware theory DSL smart compiler (OuterControl / HotLoop / Select /
`#ignore`) plus `ConsoleDashboard` stderr progress on search CLIs. Hosted gates
`Gate [dsl-smart]` / `Gate [cli-progress]`.

Details: [`docs/architecture/dsl-console-exit.md`](../../docs/architecture/dsl-console-exit.md)
· full notes under [`release/0.8.0/RELEASE_NOTES.md`](../0.8.0/RELEASE_NOTES.md).
