# Specifications

This folder holds **normative contracts** for the C++20 Liber Primus toolkit:
arithmetic, tokens, transforms, scores, fixtures, CLIs, CPU↔CUDA parity, and the
theory DSL / artifact contracts.

Research background lives in [`docs/research/`](../research/README.md). Specs here
are binding for conforming implementations.

## Documents

| File | Normative topic |
|------|-----------------|
| [z29.md](z29.md) | `Index29`, \(\mathbb{Z}_{29}\) ops, inverses, `Z29Matrix2`/`Z29Matrix3` |
| [tokens.md](tokens.md) | Token model, separators, consumable masks |
| [interrupts.md](interrupts.md) | Cleartext-F / interrupt policy |
| [transforms.md](transforms.md) | Catalog `transform_id` roster + JSON param schemas (Hill, autokey, grids) |
| [scores.md](scores.md) | Score suite + determinism rules |
| [fixtures.md](fixtures.md) | Fixture manifest format (incl. literals) |
| [tools.md](tools.md) | Library + CLI contracts (incl. `parcae-bench` suites) |
| [agent-tools.md](agent-tools.md) | Agent allow-list, JSON envelope, `parcae.agent_config.v0`, loop contract |
| [hypothesis-workspace.md](hypothesis-workspace.md) | Workspace + HypothesisRecord + transcripts; research layout + determinism |
| [search-loop.md](search-loop.md) | Search job / batch / prior / cycle; family ↔ transform map; extended opt-ins |
| [parity.md](parity.md) | CPU↔CUDA parity contract |
| [dsl.md](dsl.md) | Theory DSL; `z29_matmul` / `z29_det` / `z29_autokey_shift`; catalog compose leaves |
| [dsl-ast-json.md](dsl-ast-json.md) | `parcae.dsl_ast_json.v0` CPython→C++ AST wire format |
| [theory-artifact.md](theory-artifact.md) | `parcae.theory_artifact.v0`, URIs, registry invalidate rules |
| [bench-probe.md](bench-probe.md) | External bench probe JSON 1.0.0 (`--suite probe`) |
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

## Consistency (catalog vs search vs generators)

| Layer | Source of truth |
|-------|-----------------|
| Catalog `transform_id` | [transforms.md](transforms.md) § Catalog index |
| Search `family` + opt-ins | [search-loop.md](search-loop.md) § family ↔ `transform_id` |
| Bounded `gen_*` | [transforms.md](transforms.md) § Candidate generators |
| DSL matrix / autokey builtins | [dsl.md](dsl.md) § Core math ops / `DslCatalogBuiltins` |
| Matrix arithmetic | [z29.md](z29.md) § Matrices |
