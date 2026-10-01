# Search engine headers (`include/parcae/search/`)

C++20 closed-loop search surface. Normative:
[`docs/spec/search-loop.md`](../../../docs/spec/search-loop.md). Architecture:
[`docs/architecture/search-engine.md`](../../../docs/architecture/search-engine.md).
Operator guide:
[`docs/architecture/search-handbook.md`](../../../docs/architecture/search-handbook.md).

## Style (HARD)

Top-level classes only — **no** C++ namespaces. One class per header with
`#ifndef NAME_HPP` / `#endif // NAME_HPP`.

## Status

| Header | Class | Status |
|--------|-------|--------|
| `search_job.hpp` | `SearchJob` | Done (`parcae.search_job.v0`) |
| `search_prior.hpp` | `SearchPrior` | Done (`parcae.search_prior.v0`) |
| `batch_artifact.hpp` | `BatchArtifact` | Done (`parcae.batch_artifact.v0`) |
| `workspace_cipher.hpp` | `WorkspaceCipher` | Done (`workspace.v0` → `Index29`) |
| `gpu_candidate_export.hpp` | `GpuCandidateExport` | Done (Caesar…affine + vigenere + compose recipes + opt-in beaufort/totient + theory fused χ² / scores-only + `TheoryExportCache`; theory launch via `TheoryHistChi2Launch`) |
| `theory_export_cache.hpp` | `TheoryExportCache` | Done — host bytecode + device `ops`/`imm` reuse across theory chunks |
| `nvtx_range.hpp` | `NvtxRange` | Done — RAII NVTX for nsys (`prepare_theory`…`ingest`) |
| `cpu_candidate_export.hpp` | `CpuCandidateExport` | Done (v0 families + extended `hill_2`/`hill_3`/CTAK/PTAK + `theory` URI; hard CPU-only for hill/autokey) |
| `hypothesis_bridge.hpp` | `HypothesisBridge` | Done (ingest + idempotent ids / provenance) |
| `search_scheduler.hpp` | `SearchScheduler` | Done (`run_once` / `run_loop` + prior feedback) |
| CLI `parcae-search-cycle` | tool `search_cycle` | Done (`--status` / run / `--omit-timing` / `--allow-extended-families` / `--allow-theory-uri` / AgentPolicy + JSON goldens) |

`SearchJob` emits `allow_extended_families` / `allow_theory_uri` (default false).
`is_cpu_export_only_family` covers hill / CTAK / PTAK (hard CPU). `theory` is
**not** hard CPU-only: with χ² + decrypt,
`has_fused_cuda_chi2_export("theory")` is true and the scheduler calls
`GpuCandidateExport::theory_explicit_params`. Toolkit **1.1.0**
lifts hill/autokey off the hard list —
[`cuda-catalog-parity.md`](../../../docs/architecture/cuda-catalog-parity.md).
`SearchScheduler::run_loop` keeps a `TheoryExportCache` across iterations
(override via `Options::theory_cache` / `LoopOptions::theory_cache`). Use
`GpuCandidateExport::theory_scores_only` for score sweeps without materialize.
Family ↔ catalog `transform_id` map:
[`search-loop.md`](../../../docs/spec/search-loop.md).
Grid-read / columnar / `variable_delay_autokey` are **not** search families.

Tests: `[search][job]`, `[search][prior]`, `[search][batch]`, `[search][roundtrip]`,
`[search][cipher]`, `[search][cipher][resolve]`, `[search][export]`,
`[search][export][parity]`, `[search][export][compose][parity]`, `[search][bridge]`,
`[search][bridge][score]`, `[search][scheduler]`, `[search][scheduler][loop]`,
`[search][scheduler][prior]`, `[search][scheduler][loop][determinism]`,
`[search][adversarial]`, `[search][adversarial][job]`,
`[search][adversarial][path]`, `[search][adversarial][caps]`,
`[search][batch][limits]`, `[search][batch][fuzz]`,
`[research][falsify]`, `[tool][search_cycle]`, `[tool][golden][cli][search_cycle]`,
`[tool][policy][cli][search_cycle]` — types + export (incl. extended/hill/autokey/theory)
+ ingest + scheduler + falsify corpus load/determinism + CLI cycle run + goldens +
AgentPolicy allow path + adversarial job/path/caps + BatchArtifact limits / ordering fuzz.
CLI smoke: `ctest -R cli_search_cycle_status_json`.
Hosted CI matrix gate: `parcae_tests "[search]"` (`.github/workflows/ci.yml`).
Tag table (CPU vs CUDA): [`docs/architecture/cuda-build.md`](../../../docs/architecture/cuda-build.md)
§ Catch2 tags (search engine / scheduler).
