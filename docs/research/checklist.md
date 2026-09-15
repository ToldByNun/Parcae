# Research checklist

Research docs are complete when the freeze below holds. No code is required here.

## Checklist

### Q1 — Known Liber Primus / Gematria properties

- [x] 29-rune alphabet table frozen with indices `0..28` and first-29 primes  
- [x] Arithmetic domain stated as \(\mathbb{Z}_{29}\) on **indices**, not primes  
- [x] Latin aliases / orthography notes recorded  
- [x] Separator grammar documented (ASCII default + Unicode map)  
- [x] Document: [gematria-primus.md](gematria-primus.md), [separators.md](separators.md)

### Q2 — Plausible transforms

- [x] Tier A confirmed families listed (identity, Atbash, Caesar, compose, Vigenère, totient, interrupts)  
- [x] Affine / Beaufort noted as algebraic / shared-implementation targets  
- [x] Document: [hypotheses.md](hypotheses.md), [solved-methods.md](solved-methods.md)

### Q3 — Modulo-29 hypotheses

- [x] Confirmed vs plausible vs unverified tiers written  
- [x] Cleartext-F vs ciphertext-F interrupt distinction recorded (wiki-primary)  
- [x] artwaste essay **fetched and demoted** to unverified single-source (not frozen fact)  
- [x] Folklore true/false checklist included with confidence labels  
- [x] Document: [hypotheses.md](hypotheses.md)

### Q4 — Test material

- [x] Nine solved fixture ids selected  
- [x] Synthetic micro-vectors listed  
- [x] Negative controls defined  
- [x] Unsolved corpus explicitly deferred  
- [x] Skip-index **recompute gate** documented before golden hash lock  
- [x] `non_rune_literal_regions` covered by fixture schema (`an-end` hash)  
- [x] Document: [test-material.md](test-material.md)

### Process / sources

- [x] Research outline and source list published — [README.md](README.md)  
- [x] Confidence labels + why-29-runes rationale in README  
- [x] Non-goals stated (no solves, no CUDA, no agents in research docs)

## Open gates (block golden fixture locks, not specs)

1. **Skip-index verification:** Welcome + Koan 2 — recompute `ciphertext + key + skips → plaintext` on a named transcript before freezing SHA digests.  
2. **Tier C reproduction (optional):** only if we later want artwaste numbers as first-class scores.  
3. **Fixture schema usage:** `non_rune_literal_regions` for `an-end` and number grids when committing data.

## Artifacts

```text
docs/research/README.md
docs/research/gematria-primus.md
docs/research/separators.md
docs/research/solved-methods.md
docs/research/hypotheses.md
docs/research/test-material.md
docs/research/checklist.md
```

## Next

Normative contracts: [`docs/spec/`](../spec/README.md).  
The CPU reference implementation follows those specs.
