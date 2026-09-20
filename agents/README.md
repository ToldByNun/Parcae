# Parcae agent (`parcae-agent`)

Python 3.11+ CMD agent that drives Parcae C++ CLIs via an OpenAI-compatible
LLM. Crypto stays in C++; this package only chooses allow-listed tools.

See:

- Plan freeze: [`docs/architecture/phase4-agent-tooling.md`](../docs/architecture/phase4-agent-tooling.md)
- Tool surface: [`docs/spec/agent-tools.md`](../docs/spec/agent-tools.md)
- Config schema: `parcae.agent_config.v0` (examples in [`configs/`](configs/))

## Setup

```bash
cd agents
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI

```bash
python -m parcae_agent check-config configs/ollama.example.yaml
python -m parcae_agent doctor --config configs/ollama.example.yaml
python -m parcae_agent providers test --config configs/ollama.example.yaml
python -m parcae_agent run --config configs/ollama.example.yaml \
  --prompt "List catalog transforms for a-warning"
```

`run` accepts `--prompt`, `--prompt-file`, or stdin. Exit codes follow the agent
loop (0 completed/succeeded, 1 budget, 2 error/usage).

## LLM client

`parcae_agent.llm.LlmClient` speaks OpenAI-compatible
`POST {base_url}/chat/completions` (local Ollama / LM Studio **or** OpenRouter).
Unit tests inject a fake `http_post` — no live network required.

## ToolBridge

`parcae_agent.tool_bridge.ToolBridge` builds argv from the allow-list only and
runs `subprocess` with `shell=False`. The model never supplies a shell string.

## Tool schemas

`parcae_agent.tool_schemas.openai_tools()` returns the OpenAI `tools` array
(function name + JSON Schema parameters) for chat completions. Keys stay in
lockstep with ToolBridge via an import-time assert.

## Agent loop

`parcae_agent.loop.AgentLoop` runs Liber Primus system prompt → LLM →
ToolBridge until the model stops, a success criterion hits, or a budget ends.
Unit tests use a scripted mock LLM (no network).

## Transcripts

By default `parcae-agent run` writes `parcae.transcript_step.v0` JSONL under
`data/workspaces/<id>/transcripts/` (plus optional envelope snippets). Pass
`--no-transcript` to skip. Hypotheses still go only through `hypothesis_*` tools.
