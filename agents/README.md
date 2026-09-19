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

## Config check (scaffold)

```bash
python -m parcae_agent check-config configs/ollama.example.yaml
pytest
```

Full `run` / `providers test` / `doctor` land in later Phase 4 commits.
