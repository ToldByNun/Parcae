# Spec: Search Loop v0

**Status:** Normative (search engine)  
**Schema ids:** `parcae.search_job.v0`, `parcae.batch_artifact.v0`,
`parcae.search_prior.v0`, `parcae.search_cycle_result.v0`  
**Architecture freeze:** [`search-engine.md`](../architecture/search-engine.md)  
**Related:** [`hypothesis-workspace.md`](hypothesis-workspace.md),
[`agent-tools.md`](agent-tools.md), [`tools.md`](tools.md),
[`scores.md`](scores.md), [`parity.md`](parity.md),
[`cuda-score-reduction.md`](../architecture/cuda-score-reduction.md)

This document is the **binding contract** for the Liber Primus search loop:

```text
workspace ciphertext
    → SearchJob (+ SearchPrior)
    → GPU fused / CPU fallback candidate export
    → BatchArtifact (top-k TransformCandidate)
    → HypothesisBridge → HypothesisRecord
    → updated SearchPrior → next cycle
```

Library classes land under `include/parcae/search/` (`SearchJob`, `SearchPrior`,
`BatchArtifact`, `HypothesisBridge`, `SearchScheduler`, `GpuCandidateExport`,
`WorkspaceCipher`). CLI: `parcae-search-cycle`. Agent tool name: `search_cycle`
(allow-list update in [`agent-tools.md`](agent-tools.md) is a separate doc commit).

## Principles

1. Ciphertext for a cycle MUST come from a workspace
   [`parcae.workspace.v0`](hypothesis-workspace.md) input — never by silently
   writing fixtures.
2. Fixtures under `data/fixtures/` remain **read-only**. Batch artifacts and
   hypotheses write only under `data/workspaces/<id>/`.
3. Crypto math stays in C++/CUDA. Schedulers MUST NOT embed an LLM. Optional
   agent steps call allow-listed CLIs only.
4. Same `SearchJob` JSON + seed + workspace ciphertext bytes + prior digest ⇒
   same ranked `candidate_id` sequence and same artifact digests (CPU path;
   CUDA MUST match candidate **ids** under the score rules below).
5. Timing / throughput fields are **non-deterministic**. Agent / CI JSON MUST
   support omitting them (`--omit-timing` or equivalent).
6. Implementations MUST use top-level C++ **classes** and `#ifndef` headers —
   **no** C++/CUDA `namespace` blocks in new search code.

## Normative language

| Word | Meaning |
|------|---------|
| **MUST** | Required for a conforming implementation |
| **SHOULD** | Strong default; deviation needs a written rationale |
| **MAY** | Optional |
| **MUST NOT** | Forbidden |

---

## Directory layout (workspace extension)

In addition to [`hypothesis-workspace.md`](hypothesis-workspace.md):

```text
data/workspaces/<workspace_id>/
  workspace.json
  hypotheses/
  transcripts/
  inputs/                         # optional ciphertext copies
  batches/                        # search-loop artifacts (this spec)
    <batch_id>/
      manifest.json               # parcae.batch_artifact.v0
      candidates.jsonl
      report.json                 # optional
```

| Rule | Requirement |
|------|-------------|
| `batch_id` | `[a-z_][a-z0-9_-]{0,63}` — MUST match directory name |
| Paths | All relative paths MUST resolve under the workspace root (no `..`, no absolute escapes) |
| Encoding | UTF-8; LF preferred for JSONL |
| Size | Implementations MUST enforce a documented cap on `candidates.jsonl` lines (see `max_candidates`) |

---

## `SearchJob` — `parcae.search_job.v0`

In-memory record and JSON object used by `SearchScheduler` / CLI.

```json
{
  "schema": "parcae.search_job.v0",
  "workspace_id": "lp2-page-0-explore",
  "family": "caesar",
  "score_id": "chi2_english_gp_v0",
  "score_version": "v0",
  "k": 16,
  "seed": 1,
  "backend": "cpu",
  "max_candidates": 4096,
  "direction": "decrypt",
  "param_grid": null,
  "prior": null
}
```

### Required fields

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.search_job.v0` |
| `workspace_id` | MUST match an existing workspace `id` |
| `family` | See family table |
| `score_id` | Registered score id ([`scores.md`](scores.md)) |
| `k` | Integer ≥ 1 — top-k retained after ranking |
| `seed` | `uint32` — MUST be fixed for replay; `0` MAY mean implementation-defined random only for interactive human runs and **MUST NOT** be used in CI/agent replay fixtures |
| `backend` | `cpu` \| `cuda` |
| `max_candidates` | Integer ≥ `k` — hard cap before ranking |

### Optional fields

| Field | Rule |
|-------|------|
| `score_version` | Default `"v0"` |
| `direction` | `encrypt` \| `decrypt`; default `decrypt` |
| `param_grid` | Object; family-specific bounds. `null` ⇒ use family default grid |
| `prior` | Inline `parcae.search_prior.v0` object, **or** omit and load from workspace hypotheses |

### `family` (v0)

| Family | Default expansion (intent) | Notes |
|--------|----------------------------|-------|
| `caesar` | 29 shifts | Fused χ² preferred on CUDA |
| `atbash` | Single lane | |
| `atbash_caesar` | Atbash ∘ Caesar (29) | Compose / fused twin |
| `affine` | Bounded affine grid (e.g. 812) | |
| `vigenere` | Explicit keys / bounded grid only | MUST NOT imply unbounded dictionary search |
| `compose` | Empty grid → Atbash∘Caesar 29; or explicit `recipes` / `stages` / `template` | Reuses `AtbashCaesar` fused export when grid matches; else ComposeDriver / ComposeTransform |
| `beaufort` | Explicit keys / bounded grid (same as vigenère) | **Opt-in:** `allow_extended_families: true` |
| `totient` | Bounded `prime_start_index` list / count | **Opt-in:** `allow_extended_families: true` |
| `theory` | Explicit `param_grid.theory_uri` + `param_grid.params_list` only | **Opt-in:** `allow_theory_uri: true`; no TheorySweep expansion; CPU-only |

Loaders MUST reject unknown `family` values. Extended families (`beaufort`,
`totient`) MUST be rejected unless `allow_extended_families` is true (job JSON
and/or CLI `--allow-extended-families`). Family `theory` MUST be rejected unless
`allow_theory_uri` is true (job JSON and/or CLI `--allow-theory-uri`). Theory
jobs MUST supply a non-empty `params_list` of param objects (top candidates only;
MUST NOT expand TheorySweep grids).

### Ciphertext resolution

`WorkspaceCipher` (library) MUST:

1. Load `workspace.json`.
2. Resolve `input.kind`:
   - `fixture_ciphertext` → read ciphertext from fixture id under `data/fixtures/` (**read-only**).
   - `workspace_file` → read path under the workspace root only.
   - `inline_pending` → MUST fail the job unless the CLI supplied indices/bytes for this run.
3. Produce an `Index29` stream (via tokenize / fixture loader rules in
   [`tokens.md`](tokens.md) / [`fixtures.md`](fixtures.md)).

MUST NOT read fixture **plaintext** when building search inputs for unsolved
research workspaces.

---

## `SearchPrior` — `parcae.search_prior.v0`

Derived from workspace hypotheses (and optionally inlined on the job).

```json
{
  "schema": "parcae.search_prior.v0",
  "workspace_id": "lp2-page-0-explore",
  "seeds": [
    {
      "hypothesis_id": "h-caesar-3",
      "envelope": {
        "transform_id": "caesar",
        "direction": "decrypt",
        "params": { "shift": 3 }
      }
    }
  ],
  "exclusions": [
    {
      "param_hash": "sha256:…",
      "hypothesis_id": "h-caesar-7",
      "reason": "rejected"
    }
  ],
  "built_utc": "2026-09-21T18:00:00Z"
}
```

| Field | Rule |
|-------|------|
| `seeds` | From hypotheses with status `promoted` (SHOULD); MAY include `scored` when explicitly configured |
| `exclusions` | From status `rejected`; `param_hash` MUST be a stable digest of the canonical envelope params (same inputs ⇒ same hash) |
| Soft weights | **MUST NOT** appear in v0 |

When applying a prior:

- Seed envelopes SHOULD be forced into the candidate set (or used to narrow
  `param_grid`) before ranking.
- Exclusion hashes MUST remove matching candidates before writing the artifact.

---

## `BatchArtifact` — `parcae.batch_artifact.v0`

### `manifest.json`

```json
{
  "schema": "parcae.batch_artifact.v0",
  "batch_id": "b-caesar-20260921-0001",
  "workspace_id": "lp2-page-0-explore",
  "created_utc": "2026-09-21T18:00:01Z",
  "job_digest_sha256": "…",
  "prior_digest_sha256": "…",
  "family": "caesar",
  "score_id": "chi2_english_gp_v0",
  "score_version": "v0",
  "backend": "cpu",
  "k": 16,
  "candidate_count": 16,
  "seed": 1,
  "ordering": "batch_ordering_v0",
  "paths": {
    "candidates": "candidates.jsonl",
    "report": "report.json"
  }
}
```

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.batch_artifact.v0` |
| `batch_id` | MUST match directory name |
| `job_digest_sha256` | Digest of canonical job JSON (excluding non-replay fields) |
| `candidate_count` | MUST equal number of lines in `candidates.jsonl` |
| `ordering` | MUST be `batch_ordering_v0` — score → `candidate_id` → source index (same spirit as `BatchOrdering`) |
| `paths.report` | MAY be null if report omitted |

### `candidates.jsonl`

One JSON object per line, **best-first** rank order. Each line MUST be a
`TransformCandidate` wire object compatible with generate/rank tools:

```json
{
  "candidate_id": "caesar-shift-3",
  "envelope": {
    "transform_id": "caesar",
    "direction": "decrypt",
    "params": { "shift": 3 }
  },
  "output_indices": [19, 7, 4],
  "score": {
    "score_id": "chi2_english_gp_v0",
    "score_version": "v0",
    "value": 12.34,
    "backend": "cpu"
  },
  "rank": 0
}
```

| Field | Rule |
|-------|------|
| `candidate_id` | Stable string; MUST be unique within the batch |
| `envelope` | Valid transform envelope |
| `output_indices` | Array of integers `0…28`; v0 **SHOULD** include indices for top-k |
| `score` | REQUIRED on written artifacts |
| `rank` | `0` = best |

### `report.json` (optional)

MAY include aggregate stats. When `--omit-timing` / agent mode:

- MUST NOT include `tok_per_sec`, wall durations, or device clocks.
- MAY include digests, counts, and score summaries.

---

## Candidate export

### CPU path (CI source of truth for ids)

1. Expand family via `GenerateCandidates` / registry defaults (+ prior seeds,
   − exclusions).
2. Rank with `RankCandidates` / `BatchRunner` (`BatchOrdering`).
3. Keep top-`k` (≤ `max_candidates` expansion).

### CUDA path

1. Prefer fused χ² (`FamilyChi2Batch` / family twin) scores-only → host top-k
   lanes.
2. Materialize `TransformCandidate` envelopes for top-k only.
3. Preview `output_indices`: apply **only** for retained top-k — MUST NOT D2H
   full C×T plaintext for large grids.
4. If fused path missing for a family: staged SoA + `CudaBatchScore`.

### Parity

For the same job + seed + ciphertext on a locked Tier-A fixture:

- Ranked `candidate_id` sequences MUST match between CPU and CUDA.
- Score values MUST compare under
  [`cuda-score-reduction.md`](../architecture/cuda-score-reduction.md) /
  [`parity.md`](parity.md). Tie-breaks MUST follow `batch_ordering_v0`.

Hosted CI without CUDA MUST still run the CPU path.

---

## Hypothesis bridge

`HypothesisBridge` ingests a `BatchArtifact` into
[`parcae.hypothesis.v0`](hypothesis-workspace.md) records.

### Idempotent ids

For each candidate, the hypothesis id MUST be a deterministic function of
`(workspace_id, batch_id, candidate_id)`. Re-ingesting the same batch MUST
update or no-op without duplicating files.

**v0 algorithm** (`HypothesisBridge::hypothesis_id_for`):

1. Preimage (UTF-8): `workspace_id + "\n" + batch_id + "\n" + candidate_id`
   (`HypothesisBridge::id_preimage`).
2. Digest: lowercase hex SHA-256 of the preimage.
3. Id: `"h"` + first 32 hex characters (always matches
   `WorkspacePaths::validate_id`).

Changing any of the three fields MUST produce a different id. The same triple
MUST always resolve to the same path under
`workspaces/<workspace_id>/hypotheses/<id>.json`.

### Initial status

- Auto-ingest MUST create `draft` or `proposed` (SHOULD use `proposed` when
  `method` is complete).
- MUST NOT auto-set `promoted`.
- MUST NOT write under `data/fixtures/`.

### Extended `source` fields

Authoritative field table:
[`hypothesis-workspace.md`](hypothesis-workspace.md) `source`. Search ingest
MUST populate at least:

| Field | Rule |
|-------|------|
| `batch_id` | MUST match `BatchArtifact.batch_id` |
| `candidate_id` | MUST match the artifact line |
| `generator_id` / `family` | SHOULD record expansion provenance |
| `rank` | SHOULD be the best-first rank from the artifact line |
| `agent_run_id` | MAY be null for pure scheduler ingest |

### Scoring

Bridge MAY leave `scores` empty; callers SHOULD run `hypothesis_score` /
scheduler post-pass to attach values consistent with the batch `score` object.

---

## Cycle semantics — `SearchScheduler`

### `run_once`

1. Resolve ciphertext (`WorkspaceCipher`).
2. Build / load `SearchPrior`.
3. Export candidates (CPU or CUDA per `backend`).
4. Write `BatchArtifact` under `batches/<batch_id>/`.
5. Ingest top-k via `HypothesisBridge`.
6. Return `parcae.search_cycle_result.v0`.

### `run_loop`

Repeat `run_once` while:

- `iteration < max_iterations`, and
- wall budget not exceeded, and
- previous cycle produced new candidates or status changes (SHOULD stop early on
  zero new candidates).

**Stop reasons** (MUST be reported in the result):

| Reason | Meaning |
|--------|---------|
| `completed_iterations` | Hit `max_iterations` |
| `wall_budget` | Hit wall-time budget |
| `no_new_candidates` | Empty or duplicate-only export after prior filter |
| `success_promoted` | At least one hypothesis reached `promoted` (optional stop) |
| `success_validate` | Optional validate tool success when configured |
| `error` | Hard failure (schema, I/O, backend unavailable) |

### Optional agent step

CLI flag `--with-agent` (or equivalent) MAY invoke `parcae-agent` with a bounded
budget after ingest. Default CI cycles MUST be pure C++ (deterministic, no LLM).

---

## `parcae.search_cycle_result.v0`

Emitted inside `parcae.tool_response.v0` `result` for `parcae-search-cycle --json`:

```json
{
  "schema": "parcae.search_cycle_result.v0",
  "workspace_id": "lp2-page-0-explore",
  "iterations": 2,
  "stop_reason": "completed_iterations",
  "batches": [
    {
      "batch_id": "b-caesar-20260921-0001",
      "candidate_count": 16,
      "job_digest_sha256": "…"
    }
  ],
  "hypotheses_written": 16,
  "omit_timing": true
}
```

Success and failure of the CLI MUST still wrap in `parcae.tool_response.v0`
([`agent-tools.md`](agent-tools.md)).

---

## CLI — `parcae-search-cycle` (contract sketch)

| Flag | Rule |
|------|------|
| `--data-dir` | Parcae data root |
| `--workspace` | Workspace id (required for run) |
| `--job` | Path to `parcae.search_job.v0` JSON, or flags that build one |
| `--backend` | `cpu` \| `cuda` (MUST respect AgentPolicy / allow_cuda) |
| `--iterations` | Default `1` |
| `--json` | Envelope on stdout |
| `--omit-timing` | MUST strip timing from reports / result |
| `--status` | Versions / readiness (no cycle) |
| `--with-agent` | Optional; off by default |

`parcae-search-run` remains a **metrics / sweep dashboard** CLI and MUST stay
deny-listed for agents by default. It is not a substitute for `search-cycle`.

---

## Agent surface (normative intent)

| Tool name | CLI | Purpose |
|-----------|-----|---------|
| `search_cycle` | `parcae-search-cycle` | Run one/N workspace search cycles |

Default deny-list unchanged for `parcae-search-run`, `parcae-throughput-tiers`,
`parcae-blind-crack`. Full allow-list table update: [`agent-tools.md`](agent-tools.md)
(search-engine commit 4).

---

## Limits & abuse resistance

| Limit | v0 expectation |
|-------|----------------|
| `max_candidates` | MUST be enforced; reject with error if expansion would exceed |
| JSONL line size | MUST reject oversized candidate lines |
| Path escape | MUST reject `..` / absolute paths in workspace-relative fields |
| Unknown schema / family | MUST fail loudly (non-zero exit / tool error envelope) |

---

## Non-goals

- Multi-agent debate / beam search over hypotheses
- Unbounded dictionary / keyspace search as a catalog feature
- Hosted CI requiring NVIDIA GPU
- Storing GPU tok/s inside `HypothesisRecord.scores`
- Auto-writing fixtures on `promoted`
- Replacing hand CUDA twins with DSL-only kernels inside the scheduler

---

## Conformance (preview)

A search-loop implementation conforms when:

1. CPU `run_once` on a workspace with `fixture_ciphertext` writes a valid
   `parcae.batch_artifact.v0` and idempotent hypotheses with `source.batch_id`.
2. Two iterations with fixed seed are byte-stable for digests and `candidate_id`
   order when priors do not unexpectedly change mid-run inputs.
3. CUDA path (when built) matches CPU `candidate_id` order on a locked Tier-A
   fixture under documented score rules.
4. `--json --omit-timing` emits no timing fields.
5. Fixture directories are never written by the scheduler or bridge.

## Related

| Doc | Role |
|-----|------|
| [`search-engine.md`](../architecture/search-engine.md) | Plan freeze + commit roadmap |
| [`hypothesis-workspace.md`](hypothesis-workspace.md) | Workspace + HypothesisRecord |
| [`agent-tools.md`](agent-tools.md) | Tool envelope + allow/deny |
| [`tools.md`](tools.md) | generate / rank / search-run CLI details |
| [`parity.md`](parity.md) | CPU↔CUDA parity |
