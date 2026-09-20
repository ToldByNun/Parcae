# Specifications

This folder holds **normative contracts** for the C++20 Liber Primus toolkit:
arithmetic, tokens, transforms, scores, fixtures, CLIs, CPU↔CUDA parity, and the
theory DSL / artifact contracts.

Research background lives in [`docs/research/`](../research/README.md). Specs here
are binding for conforming implementations.

## Documents

| File | Normative topic |
|------|-----------------|
| [z29.md](z29.md) | `Index29`, \(\mathbb{Z}_{29}\) ops, inverses |
| [tokens.md](tokens.md) | Token model, separators, consumable masks |
| [interrupts.md](interrupts.md) | Cleartext-F / interrupt policy |
| [transforms.md](transforms.md) | Transform families + JSON param schemas |
| [scores.md](scores.md) | Score suite + determinism rules |
| [fixtures.md](fixtures.md) | Fixture manifest format (incl. literals) |
| [tools.md](tools.md) | Deterministic library + CLI contracts |
| [agent-tools.md](agent-tools.md) | Agent allow-list, JSON envelope, `parcae.agent_config.v0`, loop contract |
| [hypothesis-workspace.md](hypothesis-workspace.md) | Workspace + HypothesisRecord + transcripts |
| [parity.md](parity.md) | CPU↔CUDA parity contract |
| [dsl.md](dsl.md) | Theory DSL language, `dsl_spec_version`, verify gates |
| [dsl-ast-json.md](dsl-ast-json.md) | `parcae.dsl_ast_json.v0` CPython→C++ AST wire format |
| [theory-artifact.md](theory-artifact.md) | `parcae.theory_artifact.v0`, URIs, registry invalidate rules |
| [checklist.md](checklist.md) | Spec completeness checklist |

## Normative language

| Word | Meaning |
|------|---------|
| **MUST** | Required for a conforming implementation |
| **SHOULD** | Strong default; deviation needs a written rationale |
| **MAY** | Optional |
| **MUST NOT** | Forbidden |

## Non-goals

- No production C++ in this folder
- No CUDA source here
- No agent / LLM behavior inside tools
- No solve claims

## Upstream research

Where research marks a claim **unverified**, these specs either omit it from
requirements or label it optional (see scores Tier C).
