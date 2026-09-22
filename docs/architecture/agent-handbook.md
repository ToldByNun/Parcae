# Agent handbook (CMD) — `parcae-agent`

**Audience:** operators running the Liber Primus tool-use agent from a
shell  
**Product:** `parcae-agent` (Python) → OpenAI-compatible LLM → Parcae C++ CLIs  
**Normative contracts:** [`docs/spec/agent-tools.md`](../spec/agent-tools.md),
[`docs/spec/hypothesis-workspace.md`](../spec/hypothesis-workspace.md)  
**Plan freeze:** [`agent-tooling.md`](agent-tooling.md)

This handbook is the **how-to**. Specs remain authoritative when they disagree
with examples here.

## What it is

```text
you  →  parcae-agent run
          →  LLM (Ollama / OpenRouter / …)
          →  ToolBridge (allow-listed argv only, shell=False)
          →  parcae-* --json
          →  data/workspaces/<id>/  (transcripts + hypotheses)
```

Crypto, scoring, and transforms stay in C++. The model only chooses **which**
allow-listed tools to call and how to phrase hypotheses. It must **not**
reimplement \(\mathbb{Z}_{29}\) math or invent catalog ids.

## What it is not

- Not a Cursor Skill / MCP host (optional later; not required for agent-tools exit)
- Not a GPU search scheduler by itself — closed-loop cycles are
  [`search-engine.md`](search-engine.md) / tool `search_cycle`
  (`parcae-search-cycle`)
- Not a free-form shell agent — deny-listed binaries and raw shell are blocked
- Not allowed to write under `data/fixtures/`

## Prerequisites

1. **Parcae C++ tools** built (`parcae-tokenize`, `parcae-decode`, `parcae-score`,
   `parcae-validate`, `parcae-catalog`, `parcae-generate`, `parcae-rank`,
   `parcae-hypothesis`, `parcae-search-cycle`), e.g.:

   ```bash
   cmake -S . -B build -DPARCAE_BUILD_TOOLS=ON -DPARCAE_BUILD_TESTS=ON
   cmake --build build --config Release
   # or your usual build-cuda / Debug tree
   ```

2. **Python 3.11+**

3. **An OpenAI-compatible chat endpoint** with tool/function calling:
   - Local: Ollama, LM Studio, llama.cpp server, vLLM, …
   - Cloud: OpenRouter (or any gateway with the same `/v1/chat/completions` shape)

## Install the agent package

```bash
cd agents
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
```

Entry points:

```bash
python -m parcae_agent --help
# or, if Scripts/ is on PATH:
parcae-agent --help
```

## Configuration (`parcae.agent_config.v0`)

Copy an example and point paths at your machine:

| Example | Use |
|---------|-----|
| [`agents/configs/ollama.example.yaml`](../../agents/configs/ollama.example.yaml) | Local server, `api_key_env: null` |
| [`agents/configs/openrouter.example.yaml`](../../agents/configs/openrouter.example.yaml) | Cloud; set `OPENROUTER_API_KEY` |

Required fields (summary):

| Field | Notes |
|-------|--------|
| `provider.base_url` | Must include `/v1` (client posts `…/chat/completions`) |
| `provider.model` | Tool-calling capable model id |
| `provider.api_key_env` | Env **name** only, or `null` |
| `parcae_bin_dir` | Directory containing `parcae-*` binaries |
| `data_dir` | Repo `data/` root — **must not** sit under `fixtures/` |
| `workspace` | Id under `data/workspaces/<id>/` |
| `allow_cuda` | Default `false`; when true, ToolBridge may pass `--allow-cuda` |
| `budgets` | `max_steps`, `max_tool_calls`, `max_wall_seconds` (hard stops) |

Validate without talking to a model:

```bash
python -m parcae_agent check-config configs/ollama.example.yaml
python -m parcae_agent check-config configs/ollama.example.yaml --json
```

## Day-one checklist

```bash
# 1) Config + paths + binaries
python -m parcae_agent doctor --config configs/ollama.example.yaml

# 2) LLM reachable
python -m parcae_agent providers test --config configs/ollama.example.yaml

# 3) Short tool-using run
python -m parcae_agent run --config configs/ollama.example.yaml \
  --prompt "Call catalog with transforms=true, then list a few transform ids." \
  -v
```

OpenRouter: export `OPENROUTER_API_KEY` in the same shell, then use
`configs/openrouter.example.yaml`. Live smoke details:
[`agent-provider-smoke.md`](agent-provider-smoke.md).

## CLI reference

| Command | Purpose |
|---------|---------|
| `check-config PATH` | Validate YAML/JSON config |
| `doctor -c PATH` | Config, `data_dir`, fixtures, binaries, API key env |
| `providers test -c PATH` | One tiny chat completion |
| `run -c PATH` | Liber Primus agent loop |

### `run` options

| Flag | Meaning |
|------|---------|
| `--prompt` / `-p` | Inline user message |
| `--prompt-file` | UTF-8 file |
| *(stdin)* | Pipe prompt when not a TTY |
| `--json` | Machine-readable run summary on stdout |
| `--verbose` / `-v` | Log tool steps on stderr |
| `--temperature` | Sampling temperature (default `0`) |
| `--no-transcript` | Skip workspace JSONL transcript |

Exit codes from the loop:

| Code | Meaning |
|------|---------|
| 0 | Model finished with text, or success criterion met |
| 1 | Budget exhausted (`max_steps` / `max_tool_calls` / `max_wall_seconds`) |
| 2 | Usage / config / LLM / transcript error |

Success criteria (loop stops early):

- `validate` returns `ok: true`
- `hypothesis_set_status` with `status: promoted` and `ok: true`

## Allow-listed tools

The model only sees tools from [`agent-tools.md`](../spec/agent-tools.md):

`tokenize`, `decode`, `score`, `validate`, `catalog`, `generate`, `rank`,
`hypothesis_init`, `hypothesis_propose`, `hypothesis_show`, `hypothesis_list`,
`hypothesis_score`, `hypothesis_set_status`, and `search_cycle`.

**Operator tips for prompts:**

1. Ask for `catalog` before naming `transform_id` / `score_id` / `generator_id`.
2. Prefer `search_cycle` for large family grids; keep
   `generate` + `rank` for tiny explicit sets; use `decode` / `score` to check.
3. Persist work with `hypothesis_*` into the configured workspace (id is injected;
   do not pass `workspace` / `data_dir` in tool args).
4. When done, the model should reply with a short summary and **no** further
   tool calls.

Deny-listed by default: `parcae-blind-crack`, `parcae-throughput-tiers`,
`parcae-parity` / `parity-gen`, `parcae-search-run`, arbitrary shell.

## Workspaces, transcripts, hypotheses

Layout (see hypothesis-workspace spec):

```text
data/workspaces/<workspace_id>/
  workspace.json          # via hypothesis CLI / ensure
  hypotheses/*.json       # ONLY via hypothesis_* tools
  transcripts/
    <UTC>_<run_id>.jsonl  # parcae.transcript_step.v0 (default on run)
    envelopes/<run_id>/   # optional tool envelope snippets
```

- Runtime workspaces (anything other than committed `_example/`) are gitignored.
- Transcript summaries are redacted (no API keys). Prefer `--no-transcript` for
  throwaway smoke runs if you do not want files under `data/workspaces/`.
- Fixtures under `data/fixtures/` are read-only; AgentPolicy + ToolBridge enforce
  that (including refusing `--data-dir` pointed inside `fixtures/`).

## Recommended workflows

### A — Orient on a locked fixture

```bash
python -m parcae_agent run -c configs/ollama.example.yaml -v --prompt \
  "Use catalog (transforms+scores). Then validate fixture a-warning with require_locked."
```

### B — Generate and rank candidates

```bash
python -m parcae_agent run -c configs/ollama.example.yaml -v --prompt \
  "For fixtures/solved/a-warning/ciphertext.txt: catalog generators, generate with gen_atbash or gen_caesar (indices/runes as appropriate), rank with ic_mod29 k=5, summarize top hits. Persist a draft hypothesis if something looks promising."
```

Paths in tool args should be usable by the CLIs (absolute or relative to how you
invoke them). Prefer fixture ids for `validate`.

### C — Resume / inspect workspace

Use C++ CLIs directly when you only need I/O:

```bash
parcae-hypothesis list --data-dir data --workspace my-ws --json
parcae-hypothesis show --data-dir data --workspace my-ws --id h-1 --json
```

Then continue with `parcae-agent run` using the same `workspace` in config.

## Safety checklist

- [ ] Secrets only via env vars named in config — never in YAML or transcripts
- [ ] `data_dir` is the repo `data/` root, not a path under `fixtures/`
- [ ] Budgets are finite before long unattended runs
- [ ] `allow_cuda: true` only when you intend CUDA and binaries support it
- [ ] Do not put deny-listed research CLIs on a custom allow-list without review

## Tests

| Suite | Command | Network |
|-------|---------|---------|
| Unit + mock | `cd agents && pytest -q` | No (live cases skip) |
| CI mock contract | `pytest -m ci -q` | No |
| Live Ollama | `PARCAE_AGENT_LIVE=1 pytest -m "live and ollama" -q` | Yes (local) |
| Live OpenRouter | `PARCAE_AGENT_LIVE=1 pytest -m "live and openrouter" -q` | Yes + key |

Hosted CI must not set `PARCAE_AGENT_LIVE`. Details:
[`agent-provider-smoke.md`](agent-provider-smoke.md).

## Package map (for contributors)

| Module | Role |
|--------|------|
| `parcae_agent/config.py` | `parcae.agent_config.v0` |
| `parcae_agent/llm.py` | OpenAI-compatible client |
| `parcae_agent/allowlist.py` / `tool_bridge.py` | Argv allow-list + subprocess |
| `parcae_agent/tool_schemas.py` | LLM function schemas |
| `parcae_agent/prompts.py` / `loop.py` | System prompt + agent loop |
| `parcae_agent/transcript.py` | JSONL persistence |
| `parcae_agent/cli.py` / `__main__.py` | CMD entry |

## Related docs

| Doc | Why |
|-----|-----|
| [`agent-tools.md`](../spec/agent-tools.md) | Envelope, allow/deny lists, config schema, loop contract |
| [`hypothesis-workspace.md`](../spec/hypothesis-workspace.md) | HypothesisRecord + transcripts |
| [`tools.md`](../spec/tools.md) | C++ library / CLI contracts |
| [`agent-tooling.md`](agent-tooling.md) | Commit roadmap / exit tag |
| [`agents/README.md`](../../agents/README.md) | Short package README |
