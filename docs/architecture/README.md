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
| [cuda-throughput.md](cuda-throughput.md) | Fused χ² ceilings (`BenchTierSpec`); Remap roofs + dual rates; Kernel SLO vs campaign wall |
| [hist-alphabet-remap.md](hist-alphabet-remap.md) | **Normative** once-count remap math (Mono/Spalten/Lag/Bigram); 93.5%/896B reinterpretation |
| [cuda-micro-opts.md](cuda-micro-opts.md) | LaunchGeom default 256; shuffle/`cp.async` A/B policy (Doc-Skip unless ≥ baseline) |
| [cuda-profile-theory.md](cuda-profile-theory.md) | Theory / search-export nsys+ncu playbook; wall vs cudaEvent; 90% peak gate |
| [theory-hist-transpile.md](theory-hist-transpile.md) | **Normative** theory fused-χ² strategies S0–S5; ≥90% shape peak (Remap or 896B diary); wall ≠ Done |
| [dsl-smart-hist.md](dsl-smart-hist.md) | **Normative** smart hist: normalize → ShapeId match; customs without presets at catalog speed |
| [theory-artifact.md](../spec/theory-artifact.md) | Artifact layout: stream `emitted/` vs fused-hist `hist/` + `paths.hist_*` / `hist_plan.v0` |
| [cuda-score-reduction.md](cuda-score-reduction.md) | Score FP / histogram reduction associativity for CUDA twins |
| [cuda-roadmap.md](cuda-roadmap.md) | **Frozen** CUDA commit roadmap (VS layout) |
| [agent-tooling.md](agent-tooling.md) | **Frozen** CMD-agent plan (`parcae-agent`) |
| [agent-handbook.md](agent-handbook.md) | CMD operator handbook for `parcae-agent` |
| [agent-provider-smoke.md](agent-provider-smoke.md) | Optional live Ollama / OpenRouter smoke (CI skips) |
| [python-transpiler.md](python-transpiler.md) | **Theory DSL compiler** — AST-JSON → IR → verify → CPU/CUDA emit → artifacts |
| [dsl-console-exit.md](dsl-console-exit.md) | **0.8.0 exit freeze** — smart DSL + `ConsoleDashboard` (`v0.8.0-dsl-console`) |
| [dsl-pack-d.md](dsl-pack-d.md) | **Pack D freeze** — `dsl_spec` 1.1.0, `load_page`, multi-diag, `z29_match`, Optimize in `DslCompile` |
| [dsl-stubs.md](dsl-stubs.md) | Stubs vs compiler — only `parcae-compile` verifies |
| [cuda-catalog-parity.md](cuda-catalog-parity.md) | **1.1.0 exit freeze** — CUDA twins for 1.0 catalog + search export + Pack D (`v1.1.0`) |
| [search-engine.md](search-engine.md) | **Frozen** search engine plan — GPU ↔ candidates ↔ hypotheses loop; exit checklist |
| [search-roadmap.md](search-roadmap.md) | **Frozen** search-engine commit list (1–52) |
| [search-handbook.md](search-handbook.md) | Operator guide for `parcae-search-cycle` / `search_cycle` (incl. LP2 `inputs/` recipe) |
| [bench-diagnostics.md](bench-diagnostics.md) | Operator handbook for `parcae-bench` (SLO / accuracy / hardware / probe) |
| [bench-exit.md](bench-exit.md) | **0.9.0 exit freeze** — `parcae-bench` / `v0.9.0-bench` |
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

Exit freeze (smart compiler + console progress):
[`dsl-console-exit.md`](dsl-console-exit.md).

Pack D (`dsl_spec` **1.1.0** authoring pack): [`dsl-pack-d.md`](dsl-pack-d.md)
— ships on toolkit **1.1.0** with [`cuda-catalog-parity.md`](cuda-catalog-parity.md).

**Bench / diagnostics:** operator handbook
[`bench-diagnostics.md`](bench-diagnostics.md); exit freeze
[`bench-exit.md`](bench-exit.md) (`v0.9.0-bench`).

**1.1.0 catalog CUDA parity + Pack D:** exit freeze
[`cuda-catalog-parity.md`](cuda-catalog-parity.md) (`v1.1.0`).

CUDA DSL emit smoke: Catch2 tag `[cuda][dsl][smoke]` — see [`cuda-build.md`](cuda-build.md).
