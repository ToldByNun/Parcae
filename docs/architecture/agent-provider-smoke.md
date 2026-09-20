# Provider smoke (Ollama + OpenRouter)

**Status:** Optional live checks for Phase 4 `parcae-agent`  
**CI default:** skipped — no API keys, no local GPU models on hosted runners  
**Related:** [`phase4-agent-tooling.md`](phase4-agent-tooling.md),
[`agent-handbook.md`](agent-handbook.md),
[`agents/README.md`](../../agents/README.md)

Phase 4 exit requires **documented** live paths for:

1. Local OpenAI-compatible servers (Ollama / LM Studio / llama.cpp / …)
2. OpenRouter (or compatible cloud gateways)

Hosted CI must **not** depend on either. Use the mock suite (`pytest -m ci`) for
merge gates; run the smoke steps below on a developer machine when changing
provider code.

## Gate

| Env | Meaning |
|-----|---------|
| `PARCAE_AGENT_LIVE=1` | Enable live provider smoke tests / acknowledge network use |
| `OPENROUTER_API_KEY` | Required for OpenRouter smoke (never commit) |

Without `PARCAE_AGENT_LIVE=1`, `pytest -m live` skips every live smoke case.

## Shared prep

```bash
cd agents
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
cmake --build ../build-cuda --config Debug --target parcae-catalog  # or your BIN dir
```

Adjust `parcae_bin_dir` in the example YAML if your tools live under
`build/tools/Release` instead of `build-cuda/tools/Debug`.

## Path A — Ollama (local)

1. Install and start [Ollama](https://ollama.com/); pull a chat model, e.g.
   `ollama pull llama3.1`.
2. Confirm the OpenAI-compatible endpoint:
   `http://127.0.0.1:11434/v1/chat/completions`.
3. Use [`agents/configs/ollama.example.yaml`](../../agents/configs/ollama.example.yaml)
   (`api_key_env: null`).

```bash
python -m parcae_agent doctor --config configs/ollama.example.yaml
python -m parcae_agent providers test --config configs/ollama.example.yaml
python -m parcae_agent run --config configs/ollama.example.yaml \
  --prompt "Call catalog with transforms=true, then summarize transform ids." \
  --no-transcript
```

Optional pytest (needs a running Ollama):

```bash
# Windows PowerShell
$env:PARCAE_AGENT_LIVE="1"
pytest -m "live and ollama" -q

# Unix
PARCAE_AGENT_LIVE=1 pytest -m "live and ollama" -q
```

## Path B — OpenRouter (cloud)

1. Create an OpenRouter key; export it as `OPENROUTER_API_KEY` (shell env only).
2. Use [`agents/configs/openrouter.example.yaml`](../../agents/configs/openrouter.example.yaml).
3. Pick a tool-calling capable model id in `provider.model` if the default is
   unavailable on your account.

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="sk-or-..."
python -m parcae_agent doctor --config configs/openrouter.example.yaml
python -m parcae_agent providers test --config configs/openrouter.example.yaml

# Unix
export OPENROUTER_API_KEY=sk-or-...
python -m parcae_agent doctor --config configs/openrouter.example.yaml
python -m parcae_agent providers test --config configs/openrouter.example.yaml
```

Optional pytest:

```bash
PARCAE_AGENT_LIVE=1 pytest -m "live and openrouter" -q
```

## What CI runs instead

```bash
cd agents
pytest -m ci -q          # mock LLM loop, no network
pytest -q                # full unit suite; live tests skip
```

Live smoke must never be required for merge. Secrets must never appear in
transcripts, logs, or committed configs (`api_key_env` names only).

## Troubleshooting

| Symptom | Check |
|---------|--------|
| `providers test` HTTP connection error (Ollama) | Is `ollama serve` up? Firewall? Port 11434? |
| OpenRouter HTTP 401 | `OPENROUTER_API_KEY` set in the same shell? |
| `doctor` FAIL binaries | Build `parcae-*` tools; fix `parcae_bin_dir` |
| Model ignores tools | Switch to a tool-calling model; try `providers test` first |
