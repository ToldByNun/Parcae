# Architecture

Guides for how the C++ toolkit is structured and how CUDA attaches.

| Doc | Contents |
|-----|----------|
| [cpu-reference.md](cpu-reference.md) | CPU module map, data flow, hot-path rules |
| [cuda-reference.md](cuda-reference.md) | CUDA twin layout, batch/fused search, exit checklist |
| [cuda-handoff.md](cuda-handoff.md) | CPU exit criteria, twin list, parity hash procedure |
| [cuda-abi.md](cuda-abi.md) | Device buffer shapes, SoA, interrupt encoding |
| [cuda-build.md](cuda-build.md) | Local CUDA Toolkit / CMake notes; CI stays CPU-default |
| [cuda-throughput.md](cuda-throughput.md) | Fused χ² throughput ceilings (`parcae-throughput-tiers`) |
| [cuda-score-reduction.md](cuda-score-reduction.md) | Score FP / histogram reduction associativity for CUDA twins |
| [cuda-roadmap.md](cuda-roadmap.md) | **Frozen** CUDA commit roadmap (VS layout) |
| [phase4-agent-tooling.md](phase4-agent-tooling.md) | **Frozen** Phase 4 CMD-agent plan (`parcae-agent`) |
| [agent-provider-smoke.md](agent-provider-smoke.md) | Optional live Ollama / OpenRouter smoke (CI skips) |

**CUDA implementation directory:** [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (Visual Studio).

**Agent (Phase 4):** Python LLM loop under [`agents/parcae_agent/`](../../agents/parcae_agent/) calling C++ CLIs only — see Phase 4 freeze.
Config schema `parcae.agent_config.v0`: [`docs/spec/agent-tools.md`](../spec/agent-tools.md).
