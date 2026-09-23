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
| `gpu_candidate_export.hpp` | `GpuCandidateExport` | Done (Caesar…affine + vigenere + opt-in beaufort/totient) |
| `cpu_candidate_export.hpp` | `CpuCandidateExport` | Done (generate + rank + prior + extended + theory URI) |
| `hypothesis_bridge.hpp` | `HypothesisBridge` | Done (ingest + idempotent ids / provenance) |
| `search_scheduler.hpp` | `SearchScheduler` | Done (`run_once` / `run_loop` + prior feedback) |
| CLI `parcae-search-cycle` | tool `search_cycle` | Done (`--status` / run / `--omit-timing` / AgentPolicy + JSON goldens) |

Tests: `[search][job]`, `[search][prior]`, `[search][batch]`, `[search][roundtrip]`,
`[search][cipher]`, `[search][cipher][resolve]`, `[search][export]`,
`[search][export][parity]`, `[search][bridge]`, `[search][bridge][score]`,
`[search][scheduler]`, `[search][scheduler][loop]`, `[search][scheduler][prior]`,
`[search][scheduler][loop][determinism]`, `[tool][search_cycle]`,
`[tool][golden][cli][search_cycle]`, `[tool][policy][cli][search_cycle]` — types +
export + ingest + scheduler + CLI cycle run + goldens + AgentPolicy allow path.
CLI smoke: `ctest -R cli_search_cycle_status_json`.
