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
| `batch_artifact.hpp` | `BatchArtifact` | Planned |
| `workspace_cipher.hpp` | `WorkspaceCipher` | Planned |
| `hypothesis_bridge.hpp` | `HypothesisBridge` | Planned |
| `gpu_candidate_export.hpp` | `GpuCandidateExport` | Planned |
| `search_scheduler.hpp` | `SearchScheduler` | Planned |

Tests: `[search][job]` SearchJob JSON parse / validate / digest; `[search][prior]`
SearchPrior from hypotheses / exclusions / digest.
