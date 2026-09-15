# Spec checklist

Specifications are complete when the normative docs below exist and stay consistent
with [`docs/research/`](../research/README.md) (including confidence demotions).

## Checklist

### Arithmetic & tokens

- [x] [z29.md](z29.md) — Index29, ops, inv, error policy, test vectors
- [x] [tokens.md](tokens.md) — kinds, consumable indices, masks, literal kinds

### Interrupts & transforms

- [x] [interrupts.md](interrupts.md) — explicit skip policy; forbidden all-F skip as oracle
- [x] [transforms.md](transforms.md) — required families, JSON envelopes, generators, fixture map

### Scores & fixtures

- [x] [scores.md](scores.md) — Tier A required scores; Tier C non-normative
- [x] [fixtures.md](fixtures.md) — `fixture_manifest_v0`, verification gate, **`non_rune_literal_regions`**

### Tools & parity

- [x] [tools.md](tools.md) — library + CLI contracts; agent-facing five primitives
- [x] [parity.md](parity.md) — CPU obligations for later CUDA bit-identity

### Index / process

- [x] [README.md](README.md) — map of specs + MUST/SHOULD language

## Consistency

| Topic | Spec alignment |
|-------|----------------|
| Skip-index risk | interrupts + fixtures `verification.recomputed_ok` |
| artwaste stats | scores Tier C optional only; not required |
| An End hash | fixtures `non_rune_literal_regions` |
| Why 29 runes | research; transforms assume frozen profile id |
| No Python | tools/parity C++-first |

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
