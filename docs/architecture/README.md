# Architecture

Guides for how the C++ toolkit is structured, how CUDA attaches, and how the
theory DSL compiles into that stack.

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
| [agent-tooling.md](agent-tooling.md) | **Frozen** CMD-agent plan (`parcae-agent`) |
| [agent-handbook.md](agent-handbook.md) | CMD operator handbook for `parcae-agent` |
| [agent-provider-smoke.md](agent-provider-smoke.md) | Optional live Ollama / OpenRouter smoke (CI skips) |
| [python-transpiler.md](python-transpiler.md) | **Theory DSL compiler** — AST-JSON → IR → verify → CPU/CUDA emit → artifacts |
| [dsl-stubs.md](dsl-stubs.md) | Stubs vs compiler — only `parcae-compile` verifies |
| [search-engine.md](search-engine.md) | **Frozen** search engine plan — GPU ↔ candidates ↔ hypotheses loop; exit checklist |
| [search-roadmap.md](search-roadmap.md) | **Frozen** search-engine commit list (1–52) |
| [search-handbook.md](search-handbook.md) | Operator guide for `parcae-search-cycle` / `search_cycle` (incl. LP2 `inputs/` recipe) |
| [release.md](release.md) | Cutting a release — tag-only packaging, installer matrix, `SHA256SUMS` |

**CUDA implementation directory:** [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (Visual Studio).

**Agent tooling:** Python LLM loop under [`agents/parcae_agent/`](../../agents/parcae_agent/) calling C++ CLIs only — see [`agent-tooling.md`](agent-tooling.md).
Operator guide: [`agent-handbook.md`](agent-handbook.md).
Config schema `parcae.agent_config.v0`: [`docs/spec/agent-tools.md`](../spec/agent-tools.md).

**Search engine:** closed loop **owned by** [`search-engine.md`](search-engine.md)
(exit label `v0.7.0-search-engine`; engineering checklist green). Commit list:
[`search-roadmap.md`](search-roadmap.md). Operator guide:
[`search-handbook.md`](search-handbook.md). Agent-tooling delivers the tool bridge
only; `SearchScheduler` / `parcae-search-cycle` land here.

---

## Theory DSL (start here)

Architecture guide: **[`python-transpiler.md`](python-transpiler.md)**.

| Piece | Location |
|-------|----------|
| Normative language | [`docs/spec/dsl.md`](../spec/dsl.md) (incl. OuterControl / HotLoop / `#ignore`) |
| AST wire format | [`docs/spec/dsl-ast-json.md`](../spec/dsl-ast-json.md) |
| Artifacts / URIs | [`docs/spec/theory-artifact.md`](../spec/theory-artifact.md) |
| IDE stubs (fail-loud) | [`python/parcae/dsl/`](../../python/parcae/dsl/) — [`dsl-stubs.md`](dsl-stubs.md) |
| Example theories | [`theories/examples/`](../../theories/examples/) (Select + research `#ignore`) |
| C++ compiler headers | [`include/parcae/dsl/`](../../include/parcae/dsl/) |
| Compiled artifacts | [`data/theories/`](../../data/theories/) (`parcae://theories/<name>@<ver>`) |

**Authoring → runtime (short path):**

```text
theories/examples/*.py
    → parcae-compile [--allow-dsl-ignores]   (ast_dump → gates → IR → verify → emit)
    → data/theories/…/        (manifest, apply_ir.json, envelope.json, CPU/CUDA text)
    → TheoryDispatch          (catalog ApplyTransform | theory URI + apply_ir)
```

Related CLIs: `parcae-compile`, `parcae-validate --theory`, `parcae-sweep`,
`parcae-catalog --theories` — contracts in [`docs/spec/tools.md`](../spec/tools.md).

CUDA DSL emit smoke: Catch2 tag `[cuda][dsl][smoke]` — see [`cuda-build.md`](cuda-build.md).
