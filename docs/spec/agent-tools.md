# Spec: Agent Tool Surface v0

**Status:** Normative (Phase 4)  
**Schema id:** `parcae.tool_response.v0`  
**Consumers:** `parcae-agent` (CMD) and any future caller that treats Parcae CLIs as tools  
**Related:** [`tools.md`](tools.md) (library + CLI details),  
[`phase4-agent-tooling.md`](../architecture/phase4-agent-tooling.md) (roadmap freeze)

This document is the **only** normative agent-facing contract. Library shapes in
[`tools.md`](tools.md) remain authoritative for C++ APIs; when `--json` is used
for agent tool calls, **this** envelope and allow-list apply.

## Principles

1. Agents call **allow-listed CLIs only** via argv built from validated schemas —
   never via a free-form shell string from the LLM.
2. Crypto, scoring, and transforms stay in C++. The LLM MUST NOT reimplement
   \(\mathbb{Z}_{29}\) math or invent `transform_id` values outside the catalog.
3. Same inputs + flags ⇒ same stdout bytes (except fields explicitly marked
   non-deterministic and omitted for agent runs).
4. Success **and** failure MUST emit `parcae.tool_response.v0` on stdout when the
   tool is invoked in agent JSON mode (`--json`).
5. Fixtures under `data/fixtures/` are **read-only**. Writes go only to
   `data/workspaces/<id>/` (see hypothesis/workspace schema).
6. No GitHub Actions, CI workflows, or hosted runners are part of this contract —
   local/tests only from the agent/tool perspective.

## Allow-list (agent tools)

These are the **only** tool names `parcae-agent` MAY expose to an LLM as
functions. Each maps to one primary CLI binary (or subcommand family).

| Tool name | CLI | Purpose |
|-----------|-----|---------|
| `tokenize` | `parcae-tokenize` | UTF-8 → token / Index29 stream |
| `decode` | `parcae-decode` | Apply transform envelope / flags → indices + latin |
| `score` | `parcae-score` | Score an Index29 / latin / rune input |
| `validate` | `parcae-validate` | Fixture or workspace validation report |
| `catalog` | `parcae-catalog` | List transforms, scores, generators, backends |
| `generate` | `parcae-generate` | Emit `TransformCandidate` JSON from a generator |
| `rank` | `parcae-rank` | Score + rank candidates; stable ties |
| `hypothesis_init` | `parcae-hypothesis init` | Create workspace hypothesis stub |
| `hypothesis_propose` | `parcae-hypothesis propose` | Write / update a HypothesisRecord |
| `hypothesis_show` | `parcae-hypothesis show` | Read one hypothesis |
| `hypothesis_list` | `parcae-hypothesis list` | List hypotheses in a workspace |
| `hypothesis_score` | `parcae-hypothesis score` | Score a stored hypothesis against input |
| `hypothesis_set_status` | `parcae-hypothesis set-status` | Update status (`draft` / `proposed` / …) |

Notes:

- `catalog` / `generate` / `rank` / `parcae-hypothesis` are Phase 4 deliverables;
  until shipped, the allow-list is still binding for the agent design.
- `decode` covers `apply_transform` + `to_latin` from [`tools.md`](tools.md);
  agents SHOULD prefer `decode` over inventing a second transform path.
- Listing score/transform ids is part of `catalog` (and MAY remain on
  `parcae-score --list` / decode help for humans).

## Deny-list (default)

The agent MUST NOT expose these as tools by default:

| Binary / capability | Reason |
|---------------------|--------|
| `parcae-blind-crack` | Heavy / research CLI; not a stable agent primitive |
| `parcae-throughput-tiers` | Benchmarking; non-deterministic timing |
| `parcae-parity` / `parcae-parity-gen` | Dev / golden maintenance |
| `parcae-search-run` | Dashboard + timing; optional later with `--omit-timing` only |
| Arbitrary shell (`cmd`, `bash`, `powershell`, `python -c`, …) | Escape hatch |
| Writing under `data/fixtures/` | Locked corpus integrity |
| Path traversal outside `data_dir` / workspace | Sandbox |
| Installing packages, mutating git, network except LLM provider | Out of tool scope |

An operator MAY add a deny-listed binary to a **custom** allow-list only via
explicit local config (never implied by the Liber Primus system prompt). Default
`parcae-agent` configs MUST keep the deny-list above.

## JSON envelope — `parcae.tool_response.v0`

Every agent tool invocation with `--json` MUST print **exactly one** JSON object
to stdout (UTF-8, no leading junk). Human diagnostics go to stderr and MUST NOT
corrupt stdout.

### Success

```json
{
  "schema": "parcae.tool_response.v0",
  "ok": true,
  "tool": "score",
  "backend": "cpu",
  "result": {
    "score_id": "chi2_english_gp_v0",
    "score_version": "v0",
    "value": 12.34
  },
  "error": null
}
```

### Failure

```json
{
  "schema": "parcae.tool_response.v0",
  "ok": false,
  "tool": "decode",
  "backend": "cpu",
  "result": null,
  "error": {
    "code": "usage",
    "message": "missing --input"
  }
}
```

### Field rules

| Field | Type | Rules |
|-------|------|-------|
| `schema` | string | MUST be `parcae.tool_response.v0` |
| `ok` | bool | `true` iff the tool completed its contract successfully |
| `tool` | string | Allow-list tool name (not necessarily the argv0 basename) |
| `backend` | string \| null | `cpu`, `cuda`, or `null` when N/A (e.g. catalog list) |
| `result` | object \| null | Tool-specific payload; `null` when `ok` is false |
| `error` | object \| null | Required when `ok` is false; MUST be `null` when `ok` is true |

`error` object:

| Field | Type | Rules |
|-------|------|-------|
| `code` | string | Stable machine code (see below) |
| `message` | string | Human-readable; MUST NOT include secrets |

Stable `error.code` values (v0):

| Code | Meaning |
|------|---------|
| `usage` | Bad flags / missing required args |
| `io` | File / path not found or unreadable |
| `schema` | Invalid JSON / envelope / HypothesisRecord |
| `policy` | AgentPolicy denial (path, backend, deny-list) |
| `not_built` | Requested backend/feature not in this build (e.g. CUDA) |
| `validation` | Fixture / hypothesis validation failed |
| `internal` | Unexpected library error |

Legacy CLI JSON shapes from [`tools.md`](tools.md) (e.g. bare
`{ "score_id", "value" }`) MUST be migrated into `result` under this envelope
for agent mode. Human-oriented non-envelope output MAY remain only when `--json`
is **not** set.

## Exit codes

Shared for allow-listed CLIs in agent mode:

| Code | Meaning |
|------|---------|
| 0 | `ok: true` |
| 1 | Soft failure: validation / score / business logic (`ok: false`, often `validation`) |
| 2 | Hard failure: usage, I/O, schema, policy, not_built, internal |

Rules:

- Exit status MUST agree with `ok`: exit `0` ⇔ `ok: true`.
- On exit ≠ 0 with `--json`, the envelope MUST still be written to stdout when
  feasible (policy denials and usage errors included).
- If the process crashes before JSON can be emitted, the agent ToolBridge MUST
  synthesize an envelope with `error.code = "internal"`.

## Backends

- Default backend for agents: **`cpu`**.
- `cuda` MUST require an explicit operator opt-in (CLI flag and/or agent config
  `allow_cuda: true`). Without opt-in, requesting CUDA is `error.code = policy`
  or `not_built` as appropriate.
- `backend` in the envelope MUST reflect the backend that actually ran.

## Agent loop contract (`parcae-agent`)

Normative behavior for the CMD agent (implementation language: Python 3.11+
under `agents/parcae_agent/`):

1. **Config** (`parcae.agent_config.v0`): provider (`base_url`, `api_key_env`,
   `model`), `parcae_bin_dir`, `data_dir`, workspace id, budgets
   (`max_steps`, `max_tool_calls`, `max_wall_seconds`).
2. **LLM transport:** OpenAI-compatible `POST /v1/chat/completions` with tool /
   function calling. Same client for local servers and OpenRouter-style APIs.
3. **ToolBridge:** maps each tool call name → argv from the allow-list schemas
   only; runs as a subprocess **without** `shell=True` on raw LLM text; parses
   `parcae.tool_response.v0`.
4. **Loop:** alternate model turn ↔ tool results until:
   - the model emits a terminal `finish` (or equivalent), or
   - a success criterion is met (e.g. `validate` with `ok: true`, or hypothesis
     status promoted per workspace schema), or
   - a budget is exhausted → process exits **non-zero**.
5. **Persistence:** each step MAY be appended under
   `data/workspaces/<id>/transcripts/`; hypotheses only via `hypothesis_*` tools.
6. **No crypto in the LLM:** the system prompt MUST instruct the model to use
   `catalog` / `generate` / `rank` rather than inventing transforms.

Budgets are hard stops. Exhaustion without success criterion is a failed run.

## Relationship to library primitives

[`tools.md`](tools.md) lists five library primitives. Agent mapping:

| Library | Agent tool |
|---------|------------|
| `tokenize` | `tokenize` |
| `apply_transform` / `to_latin` | `decode` |
| `score` | `score` |
| `validate_fixture` | `validate` |
| id registries | `catalog` |
| *(new)* candidate generators | `generate` + `rank` |
| *(new)* workspace records | `hypothesis_*` |

## Non-goals

- Embedding an LLM inside C++ tools
- Closed-loop GPU search scheduling (Phase 5)
- Requiring Cursor Skill / MCP for Phase 4 exit
- Defining HypothesisRecord field layout (separate workspace schema doc)
- Authoring or modifying GitHub Actions / CI workflow files as part of this surface

## Conformance (preview)

A Phase 4 implementation conforms when:

1. Allow-listed CLIs emit `parcae.tool_response.v0` for `--json` success and failure.
2. Deny-listed binaries are absent from default agent tool schemas.
3. Exit codes match the table above.
4. ToolBridge never executes free-form shell from model output.
5. Fixture paths are not writable through hypothesis or other agent tools.
