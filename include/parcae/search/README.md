# Search engine headers (`include/parcae/search/`)

C++20 closed-loop search surface. Normative:
[`docs/spec/search-loop.md`](../../../docs/spec/search-loop.md). Architecture:
[`docs/architecture/search-engine.md`](../../../docs/architecture/search-engine.md).

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
| `gpu_candidate_export.hpp` | `GpuCandidateExport` | Done (Caesar…affine + vigenere) |
| `cpu_candidate_export.hpp` | `CpuCandidateExport` | Done (generate + rank + prior) |
| `hypothesis_bridge.hpp` | `HypothesisBridge` | Done (batch top-k → proposed) |
| `search_scheduler.hpp` | `SearchScheduler` | Planned |

Tests: `[search][job]`, `[search][prior]`, `[search][batch]`, `[search][roundtrip]`,
`[search][cipher]`, `[search][cipher][resolve]`, `[search][export]`,
`[search][export][parity]`, `[search][bridge]` — types + export + batch ingest.
