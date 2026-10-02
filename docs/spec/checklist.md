# Spec checklist

Specifications are complete when the normative docs below exist and stay consistent
with [`docs/research/`](../research/README.md) (including confidence demotions).

## Checklist

### Arithmetic & tokens

- [x] [z29.md](z29.md) — Index29, ops, inv, matrices (`Z29Matrix2`/`3`), error policy, test vectors
- [x] [tokens.md](tokens.md) — kinds, consumable indices, masks, literal kinds

### Interrupts & transforms

- [x] [interrupts.md](interrupts.md) — explicit skip policy; forbidden all-F skip as oracle
- [x] [transforms.md](transforms.md) — catalog roster, Hill/autokey/grids, generators, fixture map

### Scores & fixtures

- [x] [scores.md](scores.md) — Tier A required scores; Tier C non-normative
- [x] [fixtures.md](fixtures.md) — `fixture_manifest_v0`, verification gate, **`non_rune_literal_regions`**

### Tools & parity

- [x] [tools.md](tools.md) — library + CLI contracts; agent-facing five primitives
- [x] [agent-tools.md](agent-tools.md) — allow-list, deny-list, tool_response envelope, agent loop
- [x] [hypothesis-workspace.md](hypothesis-workspace.md) — workspace layout, HypothesisRecord, transcripts
- [x] [search-loop.md](search-loop.md) — SearchJob, BatchArtifact, SearchPrior, cycle; family ↔ transform; extended opt-ins
- [x] [parity.md](parity.md) — CPU obligations for later CUDA bit-identity

### Theory DSL & artifacts

- [x] [dsl.md](dsl.md) — language subset, `z29_matmul`/`det`/`autokey_shift`, compose leaves, verify gates
- [x] [dsl-ast-json.md](dsl-ast-json.md) — `parcae.dsl_ast_json.v0`, ingest limits, node whitelist
- [x] [theory-artifact.md](theory-artifact.md) — `parcae.theory_artifact.v0`, URIs, stale-spec reject; stream `emitted/` vs fused-hist `hist/` + `paths.hist_*` / `hist_plan.v0`

### Index / process

- [x] [README.md](README.md) — map of specs + MUST/SHOULD language

## Consistency

| Topic | Spec alignment |
|-------|----------------|
| Skip-index risk | interrupts + fixtures `verification.recomputed_ok` |
| artwaste stats | scores Tier C optional only; not required |
| An End hash | fixtures `non_rune_literal_regions` |
| Why 29 runes | research; transforms assume frozen profile id |
| Crypto runtime | tools/parity C++-first; DSL authoring may be `.py`, compile/verify in C++ |
| Stale theories | theory-artifact + dsl `dsl_spec_version` MAJOR mismatch → reject |
| Catalog vs search | transforms roster ↔ search-loop family map; generators gaps documented |
| Matrix / Hill | z29 matrices + transforms `hill_*` + dsl `z29_matmul`/`z29_det` |

## Artifacts

```text
docs/spec/README.md
docs/spec/z29.md
docs/spec/tokens.md
docs/spec/interrupts.md
docs/spec/transforms.md
docs/spec/scores.md
docs/spec/fixtures.md
docs/spec/tools.md
docs/spec/parity.md
docs/spec/dsl.md
docs/spec/dsl-ast-json.md
docs/spec/theory-artifact.md
docs/spec/checklist.md
```

## Implementation priorities implied by these specs

1. `Index29` / Z29 + unit tests from [z29.md](z29.md)
2. Tokenizer + masks from [tokens.md](tokens.md)
3. Transforms + interrupt policy
4. Fixture manifests as `draft` → recompute → `locked`
5. CLIs matching [tools.md](tools.md)

## Explicitly out of this folder

- CMake / C++ code
- Catch2 binaries
- Committed ciphertext/plaintext fixture files
- CUDA sources
