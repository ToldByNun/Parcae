# Search handbook — closed-loop `search_cycle`

**Audience:** operators running workspace search cycles from a shell or via
`parcae-agent`  
**Product:** `parcae-search-cycle` (C++) + agent tool `search_cycle`  
**Normative contracts:** [`docs/spec/search-loop.md`](../spec/search-loop.md),
[`docs/spec/agent-tools.md`](../spec/agent-tools.md),
[`docs/spec/hypothesis-workspace.md`](../spec/hypothesis-workspace.md)  
**Plan freeze:** [`search-engine.md`](search-engine.md) · roadmap:
[`search-roadmap.md`](search-roadmap.md)

This handbook is the **how-to**. Specs remain authoritative when they disagree
with examples here. For the Liber Primus LLM loop itself, see
[`agent-handbook.md`](agent-handbook.md).

## What it is

```text
workspace.v0 ciphertext
    → SearchJob (+ SearchPrior from hypothesis statuses)
    → CpuCandidateExport | GpuCandidateExport  (top-k)
    → BatchArtifact (parcae.batch_artifact.v0)
    → HypothesisBridge → hypotheses/*.json
    → next iteration prior (promoted seeds / rejected exclusions)
```

CLI entry: `parcae-search-cycle`. Agent tool name: `search_cycle` (allow-listed).
Crypto and scoring stay in C++; the agent only chooses **when** to run a cycle
and which `family` / `job` / budgets to pass.

## What it is not

| Thing | Use instead |
|-------|-------------|
| Metrics / fused-sweep dashboard | `parcae-search-run` (**deny-listed** for agents) |
| Tiny explicit candidate lists | `generate` + `rank` |
| Locked-fixture foothold bench (χ² + oracle) | `parcae-blind-crack` (**deny-listed**; human shell only) |
| Writing under `data/fixtures/` | Never — AgentPolicy blocks it |
| Unbounded dictionary search | Out of scope for v0 |

## Prerequisites

1. **Build tools** (Release example):

   ```bash
   cmake -S . -B build -DPARCAE_BUILD_TOOLS=ON -DPARCAE_BUILD_TESTS=ON
   cmake --build build --config Release --target parcae-search-cycle
   ```

   Optional CUDA: your usual `-DPARCAE_BUILD_CUDA=ON` tree; cycle `--backend cuda`
   still needs `--allow-cuda` (or agent config `allow_cuda: true`).

2. A **workspace** under `data/workspaces/<id>/` with `workspace.json`
   (`parcae.workspace.v0`) whose `input` resolves to ciphertext (fixture id or
   workspace-relative file). See [`hypothesis-workspace.md`](../spec/hypothesis-workspace.md).

3. For agent runs: Python agent package + OpenAI-compatible endpoint — same as
   [`agent-handbook.md`](agent-handbook.md).

## LP2 workspace recipe (`inputs/`, no fixture plaintext)

Unsolved Liber Primus pages (`0`–`55`) are **not** shipped as fixtures in this
repo ([`test-material.md`](../research/test-material.md)). Practice cycles on
committed solved fixtures (`fixture_ciphertext`) are fine; **research** on an
unsolved page MUST use a local workspace whose ciphertext lives under
`inputs/` — never assume a fixture plaintext exists.

A helper `parcae.corpus.load_page` (solved-fixture warning + workspace resolve)
is **deferred** — until it lands, copy ciphertext into `inputs/ciphertext.txt`
manually as below.

### Layout

```text
data/workspaces/lp2-page-0-explore/     # id MUST match directory; gitignored except _example
  workspace.json
  inputs/
    ciphertext.txt                      # UTF-8 runes / separator text (tokenized)
  hypotheses/                           # created by search_cycle / hypothesis tools
  batches/                              # BatchArtifact trees
```

`WorkspaceCipher` tokenizes `inputs/ciphertext.txt` the same way as fixture
ciphertext files (consumable Index29 stream). Path escapes (`..`, absolute) are
rejected.

### `workspace.json` (recommended for unsolved LP2)

```json
{
  "schema": "parcae.workspace.v0",
  "id": "lp2-page-0-explore",
  "created_utc": "2026-09-23T00:00:00Z",
  "updated_utc": "2026-09-23T00:00:00Z",
  "title": "LP2 page 0 exploratory",
  "notes": "Ciphertext only under inputs/; no oracle plaintext in this tree.",
  "input": {
    "kind": "workspace_file",
    "fixture_id": null,
    "path": "inputs/ciphertext.txt"
  },
  "default_score_id": "chi2_english_gp_v0",
  "default_score_version": "v0"
}
```

| Do | Don't |
|----|-------|
| Copy / paste community ciphertext into `inputs/ciphertext.txt` | Point `input` at a solved fixture and treat its plaintext as LP2 ground truth |
| Use `kind: "workspace_file"` + workspace-relative `path` | Invent `data/fixtures/` entries with fake plaintext for unsolved pages |
| Keep fixtures read-only (`fixture_ciphertext` for drills only) | Ask tools to read fixture **plaintext** when building search inputs |
| Let `search_cycle` write batches/hypotheses under the workspace | Write under `data/fixtures/` (AgentPolicy denies) |

Normative: [`search-loop.md`](../spec/search-loop.md) § Ciphertext resolution —
`WorkspaceCipher` MUST NOT read fixture plaintext for unsolved research
workspaces. Fixture loaders used by search only consume ciphertext files
([`fixtures.md`](../spec/fixtures.md)).

### Bootstrap (shell)

```bash
# From repo root — create a local (gitignored) workspace
mkdir -p data/workspaces/lp2-page-0-explore/inputs
# Place UTF-8 ciphertext into:
#   data/workspaces/lp2-page-0-explore/inputs/ciphertext.txt
# Then write workspace.json as above (id == directory name).

parcae-search-cycle \
  --workspace lp2-page-0-explore \
  --family caesar \
  --k 16 \
  --seed 1 \
  --backend cpu \
  --json \
  --omit-timing \
  --data-dir data
```

Suggested early families: `caesar`, `atbash`, `atbash_caesar` / `compose`, then
opt-in `beaufort` / `totient` or explicit `theory` jobs as in the sections below.
Promote / reject hypotheses between iterations so `SearchPrior` seeds and
exclusions apply.

### Practice vs research

| Mode | `input.kind` | Example |
|------|--------------|---------|
| Practice / CI | `fixture_ciphertext` | `_example` → `a-warning` (has locked plaintext for **validate**, unused by search) |
| Unsolved LP2 research | `workspace_file` | `inputs/ciphertext.txt` only |

The committed `_example` workspace is illustration only — not an LP2 claim.

## Quick start (CLI)

### Readiness

```bash
parcae-search-cycle --status --json --data-dir data
```

Expect `ok: true`, `tool: "search_cycle"`, and `result.run_ready: true`.

### One CPU cycle on a family

```bash
# Assume workspace my-ws already exists and points at a fixture / input file.
parcae-search-cycle \
  --workspace my-ws \
  --family atbash \
  --k 8 \
  --seed 1 \
  --iterations 1 \
  --backend cpu \
  --created-utc 2026-09-22T20:00:01Z \
  --json \
  --omit-timing \
  --data-dir data
```

`--json` alone also omits timing. Prefer `--created-utc` + fixed `--seed` when you
need replayable digests / `batch_id`s across machines.

### Console progress

Live progress paints **stderr only** (`ConsoleDashboard`). Ranking, digests, and
`--json` stdout are unchanged whether progress is on or off.

| Mode | When | Look |
|------|------|------|
| **Panel** | Interactive TTY (default `auto`) | Fixed multi-line block rewritten in place |
| **Lines** | Pipe / CI, or `--plain-progress` | Append-only one-liners |
| **Off** | `--quiet` or `--progress off` | Silent (agent path) |

Precedence: `--quiet` > `--plain-progress` > `--progress` (`auto|panel|lines|off`).

**Line mode** (illustrative):

```text
[search_cycle] score 12/29 (41.4%) 45.00c/s 1.23k runes/s eta=0.4s best=-12.3450 shift=7 iter=1/1 t=0.27s
[search_cycle] done 29/29 (100.0%) 50.00c/s 1.45k runes/s eta=0.0s best=-8.1000 shift=3 iter=1/1 t=0.58s  [done]
```

**Panel mode** (illustrative; VT cursor-up rewrites the block):

```text
PARCAE  search_cycle  ws=my-ws  family=caesar  backend=cpu  score=chi2_english_gp_v0
stage=score  iter=1/1
[========--------------------] 41.4%  12/29
runes=87  1.23k runes/s  45.00 cand/s  eta=0.4s
best=-12.3450  shift=7
elapsed=0.27s
```

Human demo (plain lines + JSON on stdout):

```bash
parcae-search-cycle \
  --workspace my-ws \
  --family caesar \
  --k 8 \
  --seed 1 \
  --backend cpu \
  --plain-progress \
  --json \
  --data-dir data
```

Agent / script path (no progress noise):

```bash
parcae-search-cycle \
  --workspace my-ws \
  --family caesar \
  --k 8 \
  --seed 1 \
  --backend cpu \
  --json \
  --omit-timing \
  --quiet \
  --data-dir data
```

**Windows Terminal smoke (manual):** in an interactive Windows Terminal session,
run a short cycle **without** `--quiet` / `--plain-progress` (default `auto`).
Expect a live panel (VT). On older consoles where VT enable fails, the CLI falls
back to append-only lines automatically (covered by
`[cli][dashboard]` “panel falls back to lines when VT fails”). Pipe or redirect
stderr to force lines without relying on TTY detection:

```bash
parcae-search-cycle --workspace my-ws --family caesar --k 3 --seed 1 \
  --backend cpu --json --data-dir data 2> progress.log
```

Digest invariance (progress on vs `--quiet`) is locked by
`parcae_tests "[tool][search_cycle][progress][determinism]"` and the library
`[search][scheduler][loop][progress]` case.

Normative contract: [`search-loop.md`](../spec/search-loop.md) § Console progress
contract. Flags: [`tools.md`](../spec/tools.md) § `parcae-search-cycle`.
Toolkit exit checklist (smart DSL + console):
[`dsl-console-exit.md`](dsl-console-exit.md) (`v0.8.0-dsl-console`).

### Multi-iteration loop

```bash
parcae-search-cycle \
  --workspace my-ws \
  --family caesar \
  --k 16 \
  --seed 1 \
  --iterations 2 \
  --backend cpu \
  --json \
  --data-dir data
```

Iteration 2 reloads priors from hypothesis statuses (promoted → seeds, rejected →
exclusions) per [`search-loop.md`](../spec/search-loop.md).

### Job file instead of `--family`

```bash
parcae-search-cycle \
  --workspace my-ws \
  --job path/to/job.json \
  --backend cpu \
  --json \
  --data-dir data
```

Job schema: `parcae.search_job.v0` (normative in search-loop).

### CUDA (opt-in)

```bash
parcae-search-cycle \
  --workspace my-ws \
  --family caesar \
  --k 16 \
  --backend cuda \
  --allow-cuda \
  --json \
  --data-dir data
```

Without `--allow-cuda`, expect `error.code: "policy"` (exit 2).

### Extended families (beaufort / totient)

Require an explicit opt-in — either job JSON `allow_extended_families: true` or
CLI `--allow-extended-families`:

```bash
parcae-search-cycle \
  --workspace my-ws \
  --family totient \
  --k 8 \
  --allow-extended-families \
  --backend cpu \
  --json \
  --data-dir data
```

Default totient grid: `prime_start_index` in `0..31`. Override via job
`param_grid.prime_start_count` or `param_grid.prime_start_indices`. Beaufort uses
the same bounded key grid as Vigenère (`max_key_length`, default 20).

### Theory URI family (explicit params_list)

Require `allow_theory_uri: true` (job JSON) or CLI `--allow-theory-uri`, and a
job file whose `param_grid` lists the URI plus a **bounded** `params_list`
(no TheorySweep expansion). Prefer `--job` over `--family theory` (family mode
has an empty param grid and will fail validation).

Compile the theory first (`parcae-compile`); search only consumes ready
`parcae://theories/…` artifacts. Prefer Param/const HotLoop branches
([`param_select_example.py`](../../theories/examples/param_select_example.py)).
Research theories that use `#ignore DSL_FLAG:…` need
`parcae-compile … --allow-dsl-ignores` and leave `dsl_ignores_applied` on the
manifest for review — see [tools.md](../spec/tools.md) § `parcae-compile` and
[python-transpiler.md](python-transpiler.md) § Execution scopes.

```bash
parcae-search-cycle \
  --workspace my-ws \
  --job path/to/theory_job.json \
  --allow-theory-uri \
  --backend cpu \
  --json \
  --data-dir data
```

Example `param_grid`:

```json
{
  "theory_uri": "parcae://theories/quadratic_polynomial_stream@1",
  "params_list": [
    {"c2": 1, "c1": 0, "c0": 0},
    {"c2": 0, "c1": 1, "c0": 0}
  ]
}
```

Theory jobs are CPU-only (`TheoryDispatch` / `apply_ir`); `--backend cuda` falls
back to the CPU export path.

### Compose recipes (Atbash∘Caesar / ComposeDriver)

Family `compose` (no extra opt-in). Empty `param_grid` expands the Koan-1
Atbash∘Caesar 29-shift grid (same as `--family atbash_caesar`). Explicit recipes:

```json
{
  "stages": [
    {"transform_id": "atbash", "params": {}},
    {"transform_id": "caesar", "direction": "encrypt", "params": {"shift": 3}}
  ]
}
```

Or `param_grid.recipes` / `params_list` for several compose params objects, or
`template: "atbash_caesar"`. CUDA: full Atbash∘Caesar grids use the fused export;
other recipes apply via `ComposeDriver` then host χ² (top-k only).

## Outputs to inspect

After a successful cycle:

```text
data/workspaces/<id>/
  batches/<batch_id>.json     # parcae.batch_artifact.v0
  hypotheses/*.json           # ingested top-k (source.batch_id set)
  # optional report.json only when timing is not omitted
```

Useful follow-ups:

```bash
parcae-hypothesis list --data-dir data --workspace my-ws --json
parcae-hypothesis show --data-dir data --workspace my-ws --id <hyp-id> --json
parcae-hypothesis score --data-dir data --workspace my-ws --id <hyp-id> \
  --input <cipher-path> --json
```

## Agent path (`search_cycle` tool)

ToolBridge injects `--data-dir`, `--json`, `--workspace` (from config),
`--omit-timing`, and `--quiet` for cycle runs. The model MUST NOT pass
`workspace` / `data_dir` / `allow_cuda` / `quiet` / `omit_timing`.

| Model args | Meaning |
|------------|---------|
| `status: true` | Readiness only (`--status`); no family/job |
| `family` / `job` | One required for a cycle (mutually with status) |
| `k`, `seed`, `iterations`, `score_id`, `backend`, `created_utc` | Optional cycle knobs |

Prompt guidance (built into `parcae-agent`): prefer `search_cycle` for **large
family grids**; keep `generate` + `rank` for **tiny explicit** sets.

Example operator prompt:

```bash
python -m parcae_agent run -c configs/ollama.example.yaml -v --prompt \
  "On this workspace, run search_cycle family=atbash k=8 seed=1 iterations=1 backend=cpu. Summarize batches and hypotheses_written. Do not invent catalog ids."
```

Offline CI contract (no network, no real CLI): `cd agents && pytest -m ci -q`
(includes mock `search_cycle` argv + envelope checks).

## `search_cycle` vs `generate`+`rank` vs `search-run` vs `blind-crack`

| Goal | Tool | Agent default |
|------|------|---------------|
| Broad family sweep → batch → hypotheses | **`search_cycle`** | Allow-listed |
| Score a small hand-built candidate JSON | `generate` + `rank` | Allow-listed |
| Throughput / sweep dashboard / fixture eval rates | `parcae-search-run` | Deny-listed |
| Locked-fixture additive-family foothold bench (χ² + oracle) | `parcae-blind-crack` | Deny-listed |

`parcae-search-run` remains the **metrics** CLI (`SearchRun` throughput /
sweep / fixture eval) — it does not write workspace hypotheses and MUST stay
off the default agent allow-list. `parcae-blind-crack` is a **human research
CLI** (locked-fixture foothold bench): same deny rule. Prefer `search_cycle`
for Liber Primus workspace research (`inputs/` ciphertext).
Details: [`agent-tools.md`](../spec/agent-tools.md) § `search_cycle` vs
`parcae-search-run` / § `search_cycle` vs `parcae-blind-crack`,
[`tools.md`](../spec/tools.md) § `parcae-search-run` / § `parcae-blind-crack`.

## Safety checklist

- [ ] `data_dir` is the repo `data/` root — never under `fixtures/`
- [ ] Fixtures stay read-only; cycles write only under `workspaces/<id>/`
- [ ] Unsolved LP2 workspaces use `inputs/` + `workspace_file` — no fixture plaintext assumption
- [ ] Fixed `--seed` + `--created-utc` for replay / golden digests
- [ ] `--allow-cuda` / `allow_cuda: true` only when intended
- [ ] Do not expose deny-listed `search-run` / `blind-crack` to the agent

## Tests (operators / CI)

| What | Command |
|------|---------|
| CI matrix gate | `.github/workflows/ci.yml` → `Gate [search]` (`parcae_tests "[search]"`) |
| Adversarial job / path / caps | `parcae_tests "[search][adversarial]"` |
| BatchArtifact limits + ordering fuzz | `parcae_tests "[search][batch][limits]"` / `"[search][batch][fuzz]"` |
| CLI status smoke | `ctest -R cli_search_cycle_status_json` |
| Catch2 search + CLI | `parcae_tests "[search]"` / `"[tool][search_cycle]"` |
| Progress digest invariance | `parcae_tests "[tool][search_cycle][progress][determinism]"` |
| Scheduler subset | `parcae_tests "[search][scheduler]"` (see [`cuda-build.md`](cuda-build.md) § Catch2 tags) |
| JSON goldens | `parcae_tests "[tool][golden][cli][search_cycle]"` |
| AgentPolicy path | `parcae_tests "[tool][policy][cli][search_cycle]"` |
| Agent mock CI | `cd agents && pytest -m ci -q` |

Headers / tags: [`include/parcae/search/README.md`](../../include/parcae/search/README.md).

## Package map (contributors)

| Piece | Role |
|-------|------|
| `include/parcae/search/*.hpp` | Job, prior, artifact, cipher, export, bridge, scheduler |
| `tools/parcae_search_cycle/main.cpp` | CLI |
| `agents/parcae_agent/allowlist.py` | Tool → binary map |
| `agents/parcae_agent/tool_schemas.py` | LLM function schema |
| `agents/parcae_agent/tool_bridge.py` | Argv builder (`--omit-timing`, `--quiet`, workspace) |
| `agents/parcae_agent/prompts.py` | Prefer cycle vs generate/rank |

## Related docs

| Doc | Why |
|-----|-----|
| [`search-loop.md`](../spec/search-loop.md) | Schemas, scheduler semantics, result shape |
| [`search-engine.md`](search-engine.md) | Plan freeze + exit checklist (engineering green) |
| [`search-roadmap.md`](search-roadmap.md) | Commit list 1–52 |
| [`tools.md`](../spec/tools.md) | CLI flag reference |
| [`agent-handbook.md`](agent-handbook.md) | Running `parcae-agent` (tool bridge; does **not** own the scheduler) |
| [`agent-tooling.md`](agent-tooling.md) | Agent plan freeze — points here for closed-loop ownership |
| [`agent-tools.md`](../spec/agent-tools.md) | Allow/deny + envelope |
| [`cuda-score-reduction.md`](cuda-score-reduction.md) | CUDA top-k score contract |
