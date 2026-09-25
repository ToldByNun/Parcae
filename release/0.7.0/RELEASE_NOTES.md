# Parcae 0.7.0 — Search-engine exit

**Tag:** `v0.7.0-search-engine`  
**Toolkit:** 0.7.0  
**Date:** 2026-09-23

## Highlights

Closed-loop Liber Primus **search engine** is complete:

workspace ciphertext → GPU/CPU candidate export → `BatchArtifact` → `HypothesisBridge` → `SearchPrior` → next cycle, driven by `SearchScheduler` / `parcae-search-cycle`.

Agents call the loop via the allow-listed `search_cycle` tool; research CLIs (`search-run`, `blind-crack`) stay separate.

## What's new since 0.3.0 (CUDA parity)

### Search engine
- Core types: `SearchJob`, `SearchPrior`, `BatchArtifact` (hard size caps + stable ordering)
- `WorkspaceCipher` from `workspace.v0` (path-escape rejected)
- Candidate export (Caesar / Atbash / Affine / Vigenère; opt-in Beaufort/Totient; compose)
- Optional CUDA path in `RankCandidates` / `parcae-rank --backend cuda`
- `HypothesisBridge` with idempotent IDs and `source.batch_id` provenance
- `SearchScheduler` budgets / priors / deterministic multi-iteration tests
- CLI `parcae-search-cycle` (`--json`, `--status`, `--omit-timing`)
- Agent allowlist, handbook, LP2 `inputs/` recipe; CI gate `[search]`

### Release packaging
- Tag-only GitHub Actions workflow (not on ordinary commits)
- Windows Inno installers (winget bootstrap for MSVC / CMake / Python / CUDA)
- Linux tar.gz / AppImage / deb / rpm (cpu, cuda, full)
- Source archive + `SHA256SUMS`

## Artifacts (this local cut)

| File | Flavor |
|------|--------|
| `Parcae-v0.7.0-windows-x64-cpu.exe` | Windows CPU installer |
| `Parcae-v0.7.0-windows-x64-cuda.exe` | Windows CUDA installer |
| `Parcae-v0.7.0-windows-x64.exe` | Windows full (cpu + cuda bins) |
| `Parcae-v0.7.0-linux-x86_64-cpu.tar.gz` / `.AppImage` | Linux CPU |
| `Parcae-v0.7.0-linux-x86_64-cuda.tar.gz` / `.AppImage` | Linux CUDA |
| `Parcae-v0.7.0-linux-x86_64.tar.gz` / `.AppImage` | Linux full |
| `parcae_0.7.0_amd64{_cpu,_cuda,}.deb` | Debian |
| `parcae-0.7.0-1.x86_64{.cpu,.cuda,}.rpm` | RPM |
| `parcae-0.7.0-source.tar.gz` | Source |
| `SHA256SUMS` | Digests |

See `SHA256SUMS` for full hashes.

## Install notes

**Windows:** run the matching `.exe` as admin → bootstrap installs missing MSVC/CMake/Python/(CUDA) via winget → optional PATH + editable pip.

**Linux:** extract tarball / install deb|rpm / run AppImage. CUDA flavors need a compatible NVIDIA driver + toolkit runtime.

Smoke: `parcae-search-cycle --status --json` → `"toolkit_version":"0.7.0"`.

## Docs

- `docs/architecture/search-engine.md`
- `docs/architecture/search-handbook.md`
- `docs/architecture/release.md`
