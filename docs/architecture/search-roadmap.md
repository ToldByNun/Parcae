# Search engine commit roadmap

**Status:** Frozen granular commit list for the Liber Primus **search engine**  
**Plan freeze:** [`search-engine.md`](search-engine.md)  
**Normative contract:** [`docs/spec/search-loop.md`](../spec/search-loop.md)  
**Exit tag (planned):** `v0.7.0-search-engine`  
**Upstream:** CUDA parity (`v0.3.0-cuda-parity`), Theory DSL (`v0.5.0-theory-dsl`),
CMD agent tooling ([`agent-tooling.md`](agent-tooling.md), planned `v0.6.0-agent-tools`)

North star:

```text
workspace ciphertext
    → GPU fused / CPU fallback search
    → TransformCandidate top-k (BatchArtifact)
    → HypothesisRecord (+ optional agent)
    → SearchPrior → next cycle
```

Numbering is **local to this roadmap** (not a continuation of CUDA 1–42 or
agent-tooling 1–35).

C++ / CUDA style (HARD): **no `namespace`s** — top-level `class Name` +
`#ifndef NAME_HPP` headers only.

---

## Progress

| Band | Commits | Status |
|------|---------|--------|
| A — Spec & architecture freeze | 1–4 | **done** |
| B–C — Core types + workspace cipher | 5–10 | **done** |
| D–F — Export, rank CUDA, hypothesis bridge | 11–22 | **11–13 done**; next **14** |
| G–H — Scheduler + CLI | 23–30 | pending |
| I — Agent wiring | 31–35 | pending |
| J–K — Extended families / research CLI docs | 36–42 | pending |
| L–M — CI + exit | 43–52 | pending |

---

## A — Spec & architecture freeze

| # | Commit | Status |
|---|--------|--------|
| 1 | docs: search-engine plan freeze ([`search-engine.md`](search-engine.md)) | **done** |
| 2 | docs: normative search-loop schema v0 ([`search-loop.md`](../spec/search-loop.md)) | **done** |
| 3 | docs: search-roadmap + README roadmap pointer (**this**) | **done** |
| 4 | docs: revise agent-tools allow/deny for `search_cycle` | **done** |

---

## B — Core C++ types

| # | Commit |
|---|--------|
| 5 | feat(search): `SearchJob` class + JSON parse/validate |
| 6 | feat(search): `SearchPrior` from `HypothesisRecord` statuses |
| 7 | feat(search): `BatchArtifact` writer/loader (`parcae.batch_artifact.v0`) |
| 8 | test(search): Job / Prior / BatchArtifact round-trip |

## C — Workspace ciphertext

| # | Commit |
|---|--------|
| 9 | feat(search): `WorkspaceCipher` load `Index29` from `workspace.v0` |
| 10 | test(search): fixture + workspace_file resolve; path escape reject |

## D — Candidate export

| # | Commit |
|---|--------|
| 11 | feat(search): `GpuCandidateExport` Caesar fused top-k |
| 12 | feat(search): export atbash + atbash_caesar + affine |
| 13 | feat(search): export vigenere bounded / explicit-key grid |
| 14 | feat(search): `CpuCandidateExport` via generate + rank fallback |
| 15 | test(search): CPU vs CUDA top-k parity on `a-warning` (device skip if no CUDA) |

## E — Rank hardening

| # | Commit |
|---|--------|
| 16 | feat(tool): `RankCandidates` optional CUDA score path |
| 17 | feat(cli): `parcae-rank --backend cuda` when allow_cuda |
| 18 | test(tool): rank CUDA/CPU ordering contract |

## F — Hypothesis bridge

| # | Commit |
|---|--------|
| 19 | feat(search): `HypothesisBridge` ingest top-k → draft hypotheses |
| 20 | feat(search): idempotent ids + `source.batch_id` provenance |
| 21 | feat(hypothesis): extend source fields in schema + record |
| 22 | test(search): ingest → `hypothesis_score` / set-status |

## G — Scheduler

| # | Commit |
|---|--------|
| 23 | feat(search): `SearchScheduler::run_once` |
| 24 | feat(search): `SearchScheduler::run_loop` + budgets + stop reasons |
| 25 | feat(search): promoted seeds + rejected exclusions on next job |
| 26 | test(search): two-iteration deterministic CPU loop |

## H — CLI

| # | Commit |
|---|--------|
| 27 | feat(cli): `parcae-search-cycle` scaffold `--json` `--status` |
| 28 | feat(cli): `--workspace` `--iterations` `--backend` |
| 29 | feat(cli): `--omit-timing` + replayable digests |
| 30 | test(cli): JSON golden + AgentPolicy allow path | **done** |

## I — Agent wiring

| # | Commit |
|---|--------|
| 31 | feat(agent): allowlist + schemas for `search_cycle` | **done** |
| 32 | feat(agent): C++ `AgentPolicy` default allow sync | **done** (with H30) |
| 33 | feat(agent): prompts — `search_cycle` vs generate/rank | **done** |
| 34 | test(agent): mock LLM `search_cycle` CI-safe |
| 35 | docs: search-handbook operator guide |

## J — Extended families

| # | Commit |
|---|--------|
| 36 | feat(search): optional beaufort/totient families (explicit opt-in) |
| 37 | feat(search): optional theory-URI job step (top candidates only) |
| 38 | feat(search): compose recipe jobs (AtbashCaesar / ComposeDriver export) |
| 39 | test(search): compose job CPU/CUDA mirror smoke |
| 40 | docs: LP2 workspace recipe (`inputs/`, no fixture plaintext assumption) |

## K — Research CLIs stay separate

| # | Commit |
|---|--------|
| 41 | docs: clarify `blind-crack` vs `search-cycle` (still deny-listed) |
| 42 | chore: `search-run` remains metrics CLI; cross-link in `tools.md` |

## L — CI & quality

| # | Commit |
|---|--------|
| 43 | ci: gate `[search]` CPU scheduler tests on matrix |
| 44 | test(search): adversarial job JSON / path escape / caps |
| 45 | test(search): BatchArtifact limits + stable ordering fuzz |
| 46 | docs: `cuda-build.md` Catch2 tags `[search][scheduler]` |

## M — Exit

| # | Commit |
|---|--------|
| 47 | docs: search-engine exit checklist green |
| 48 | docs: README roadmap — search loop done (`v0.7.0-search-engine`) |
| 49 | docs: agent-tooling / handbook point here (closed-loop owned by search) |
| 50 | chore: version bump 0.7.0 |
| 51 | test: smoke Version + `parcae-search-cycle --status` |
| 52 | chore: annotated tag `v0.7.0-search-engine` |

---

## Dependency sketch

```text
Specs (A) → types + WorkspaceCipher (B,C) → export (D) → bridge (F)
         ↘ rank CUDA (E) can parallel after D11
                    → scheduler (G) → CLI (H) → agent (I) → extras (J) → CI/exit (L,M)
```

## Related

| Doc | Role |
|-----|------|
| [`search-engine.md`](search-engine.md) | Locked decisions + module map |
| [`search-loop.md`](../spec/search-loop.md) | Normative schemas |
| [`agent-tools.md`](../spec/agent-tools.md) | Allow/deny (`search_cycle`) |
| [`cuda-roadmap.md`](cuda-roadmap.md) | Prior CUDA twin roadmap |
| [`agent-tooling.md`](agent-tooling.md) | CMD agent freeze |
