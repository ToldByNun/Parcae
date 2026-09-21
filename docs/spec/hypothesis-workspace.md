# Spec: Hypothesis & Workspace Schema v0

**Status:** Normative (CMD agent / hypothesis workspace)
**Schema ids:** `parcae.workspace.v0`, `parcae.hypothesis.v0`, `parcae.transcript_step.v0`  
**Root path:** `data/workspaces/<workspace_id>/`  
**Related:** [`agent-tools.md`](agent-tools.md), [`fixtures.md`](fixtures.md),
[`transforms.md`](transforms.md)

Workspaces hold **mutable** agent/human research state. Fixtures under
`data/fixtures/` remain **read-only**; hypothesis tools MUST NOT write there.

## Directory layout

```text
data/workspaces/
  _example/                          # committed illustration only
    workspace.json
    hypotheses/
      <hypothesis_id>.json
    transcripts/
      <utc_stamp>_<seq>.jsonl        # optional in example
  <workspace_id>/                    # runtime (gitignored; see .gitignore)
    workspace.json
    hypotheses/
      <hypothesis_id>.json
    transcripts/
      <utc_stamp>_<seq>.jsonl
    inputs/                          # optional copies / pointers
      ciphertext.txt                 # optional
```

Rules:

| Rule | Requirement |
|------|-------------|
| `workspace_id` | `[a-z_][a-z0-9_-]{0,63}` — MUST match directory name (leading `_` allowed for reserved trees such as `_example`) |
| `hypothesis_id` | `[a-z_][a-z0-9_-]{0,63}` — MUST match filename stem |
| Paths | All relative paths inside a workspace MUST resolve under that workspace root (no `..`, no absolute escapes) |
| Encoding | UTF-8; LF preferred |
| Fixtures | References to fixtures use fixture **ids** or paths under `data/fixtures/` for **read** only |

## Workspace manifest — `parcae.workspace.v0`

File: `data/workspaces/<workspace_id>/workspace.json`

```json
{
  "schema": "parcae.workspace.v0",
  "id": "lp2-page-0-explore",
  "created_utc": "2026-09-19T12:00:00Z",
  "updated_utc": "2026-09-19T12:00:00Z",
  "title": "LP2 page 0 exploratory",
  "notes": "optional free text",
  "input": {
    "kind": "fixture_ciphertext",
    "fixture_id": "a-warning",
    "path": null
  },
  "default_score_id": "chi2_english_gp_v0",
  "default_score_version": "v0"
}
```

### Required fields

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.workspace.v0` |
| `id` | MUST equal parent directory name |
| `created_utc` / `updated_utc` | RFC 3339 UTC timestamps |
| `input.kind` | See table below |
| `default_score_id` | Registered score id (MAY be null until first score) |

### `input.kind`

| Kind | Meaning | `fixture_id` | `path` |
|------|---------|--------------|--------|
| `fixture_ciphertext` | Ciphertext from a solved/draft fixture | required | null |
| `workspace_file` | File under this workspace (e.g. `inputs/ciphertext.txt`) | null | required, workspace-relative |
| `inline_pending` | Input supplied only on CLI / agent run | null | null |

Loaders MUST reject unknown `kind` values.

## HypothesisRecord — `parcae.hypothesis.v0`

File: `data/workspaces/<workspace_id>/hypotheses/<hypothesis_id>.json`

```json
{
  "schema": "parcae.hypothesis.v0",
  "id": "h-atbash-then-caesar-3",
  "workspace_id": "lp2-page-0-explore",
  "created_utc": "2026-09-19T12:05:00Z",
  "updated_utc": "2026-09-19T12:10:00Z",
  "status": "proposed",
  "title": "Atbash then Caesar shift 3",
  "rationale": "Short agent/human note; not a proof claim.",
  "method": {
    "transform_id": "caesar",
    "direction": "decrypt",
    "params": { "shift": 3 },
    "interrupt": {
      "policy_id": "explicit_skip_indices_v0",
      "rune_index_base": 0,
      "skip_indices": []
    }
  },
  "source": {
    "generator_id": "gen_caesar_shifts",
    "candidate_id": "caesar-shift-3",
    "agent_run_id": null
  },
  "scores": [
    {
      "score_id": "chi2_english_gp_v0",
      "score_version": "v0",
      "value": 12.34,
      "backend": "cpu",
      "scored_utc": "2026-09-19T12:10:00Z"
    }
  ],
  "preview": {
    "latin_prefix": "THE...",
    "max_chars": 64
  },
  "digests": {
    "method_sha256": null,
    "output_indices_sha256": null
  },
  "promotion": {
    "target_fixture_id": null,
    "notes": "Promotion into fixtures/ is a human process; tools MUST NOT auto-write fixtures."
  }
}
```

### Required fields

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.hypothesis.v0` |
| `id` | MUST match filename stem |
| `workspace_id` | MUST match owning workspace `id` |
| `status` | See status enum |
| `method` | MUST be a valid transform envelope ([transforms.md](transforms.md) / `TransformEnvelope`) |
| `scores` | Array (MAY be empty) |
| `digests` | Object; hash fields MAY be null until computed |

### `status` enum

| Status | Meaning |
|--------|---------|
| `draft` | Stub / incomplete method |
| `proposed` | Method filled; candidate for scoring |
| `scored` | At least one score entry recorded |
| `rejected` | Explicitly discarded |
| `promoted` | Human marked worth fixture follow-up (does **not** mutate fixtures) |

Allowed transitions (v0):

```text
draft → proposed → scored → rejected
                 ↘ promoted
proposed → rejected
scored → rejected | promoted
```

Implementations MUST reject unknown transitions with `error.code = schema` (or
`validation`) under [`agent-tools.md`](agent-tools.md).

### `method`

Same shape as fixture `method` / tool `TransformEnvelope`:

- `transform_id` (string, registered)
- `direction` (`encrypt` \| `decrypt`)
- `params` (object per family schema)
- `interrupt` (optional; omit or empty skips)

Compose / multi-stage methods are **out of v0** unless represented as a single
registered `transform_id`. Do not invent ad-hoc pipeline JSON.

### `source` (optional object)

| Field | Meaning |
|-------|---------|
| `generator_id` | Id from `parcae-catalog` / GeneratorRegistry |
| `candidate_id` | From `TransformCandidate.candidate_id` |
| `agent_run_id` | Ties to a transcript series; MAY be null for human edits |

### `scores[]` entries

| Field | Rule |
|-------|------|
| `score_id` / `score_version` | Registered |
| `value` | Finite number |
| `backend` | `cpu` \| `cuda` |
| `scored_utc` | RFC 3339 UTC |

### `digests` (optional integrity)

When non-null:

| Field | Preimage |
|-------|----------|
| `method_sha256` | SHA-256 of canonical compact `method` JSON |
| `output_indices_sha256` | SHA-256 of compact JSON array of Index29 integers from a recorded apply |

Canonicalization for digests: UTF-8, compact separators, object keys sorted
lexicographically via `HypothesisRecord::canonicalize_json` /
`HypothesisRecord::method_sha256` (and `output_indices_sha256` for Index29
arrays). Callers MAY check integrity with `verify_method_digest` /
`verify_output_indices_digest` (non-null digests must match).

## Transcripts — `parcae.transcript_step.v0`

Directory: `data/workspaces/<workspace_id>/transcripts/`

Each agent (or CLI) run SHOULD append **JSON Lines** (`.jsonl`): one JSON object
per line, schema `parcae.transcript_step.v0`.

Suggested filename: `<YYYYMMDDThhmmssZ>_<run_id>.jsonl` where `run_id` is a
short opaque token (e.g. eight hex chars).

```json
{
  "schema": "parcae.transcript_step.v0",
  "workspace_id": "lp2-page-0-explore",
  "run_id": "a1b2c3d4",
  "seq": 0,
  "utc": "2026-09-19T12:05:01Z",
  "role": "tool",
  "tool": "generate",
  "ok": true,
  "summary": "generated 29 caesar candidates",
  "hypothesis_id": null,
  "envelope_ref": null
}
```

| Field | Rule |
|-------|------|
| `role` | `system` \| `user` \| `assistant` \| `tool` \| `finish` |
| `tool` | Allow-list name when `role=tool`; else null |
| `ok` | Tool outcome when `role=tool`; else null |
| `summary` | Short redacted text; MUST NOT store API keys or full ciphertext dumps by default |
| `hypothesis_id` | Set when the step created/updated a hypothesis |
| `envelope_ref` | Optional relative path to a saved tool envelope snippet; MUST stay under the workspace |

Transcripts MAY omit raw LLM payloads. If stored, they MUST live only under the
workspace and SHOULD be truncated. Secrets from provider configs MUST NEVER be
written here.

## Path safety (normative)

Implementations (CLI + AgentPolicy) MUST:

1. Resolve `data_dir` to an absolute root.
2. Allow workspace writes only under `data_dir/workspaces/<id>/`.
3. Reject any path containing `..` or escaping the workspace / data root.
4. Reject writes whose resolved path is under `data_dir/fixtures/`.
5. On denial, emit `parcae.tool_response.v0` with `error.code = policy` and exit **2**.

## Example tree

Committed illustration: [`data/workspaces/_example/`](../../data/workspaces/_example/).

Runtime workspaces (anything other than `_example/`) are local scratch and are
gitignored (see repo `.gitignore`).

## Non-goals

- Auto-promoting hypotheses into `data/fixtures/solved/`
- Multi-agent debate logs as a separate schema
- Storing GPU timing / throughput in hypothesis scores
- Embedding original puzzle JPGs

## Conformance (preview)

1. Loaders accept only the schema ids named above.
2. `id` fields match directory / file names.
3. `method` round-trips through `TransformEnvelope`.
4. Status transitions follow the table.
5. Path policy refuses fixture mutation and traversal.
