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
| `hypothesis_bridge.hpp` | `HypothesisBridge` | Planned |
| `gpu_candidate_export.hpp` | `GpuCandidateExport` | Planned |
| `search_scheduler.hpp` | `SearchScheduler` | Planned |

Tests: `[search][job]`, `[search][prior]`, `[search][batch]`, `[search][roundtrip]`,
`[search][cipher]`, `[search][cipher][resolve]` — parse / digests / ciphertext
resolve (fixture + workspace_file + path-escape).
