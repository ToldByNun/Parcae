# Test Material and Fixture IDs

**Status:** Selection freeze.  
**Scope:** Solved-page fixtures only (no full LP2 `0`–`55` corpus in-repo for the
CPU reference).

## Selection criteria

A section is good oracle / regression material if it:

1. Has a **community-accepted** plaintext and method  
2. Exercises a **distinct** transform or policy (identity, Atbash, compose, Vigenère+skips, totient+skips)  
3. Can be stored as UTF-8 transcript + JSON manifest without original JPG blobs  
4. Provides at least one **negative control** opportunity where useful  

## Canonical fixture IDs

Use these exact ids in manifests, Catch2 tags, and CLI filters.

| Fixture id | Method family | Why it is in the minimum set |
|------------|---------------|------------------------------|
| `a-warning` | Atbash | Pure reflection; no key state |
| `some-wisdom` | Identity | Orthography + embedded number grid |
| `welcome` | Vigenère + skips | Long keyed page; 11 interrupt indices |
| `koan-1` | Atbash ∘ Caesar(+3) | Composition path |
| `loss-of-divinity` | Identity | Longer identity prose |
| `koan-2` | Vigenère + skips | Short key with C→F orthography; 2 skips |
| `an-instruction` | Identity | Identity + symmetric number block |
| `an-end` | Totient stream + F pass-through | Aperiodic stream; hardest interrupt semantics |
| `lp2-57-identity` | Identity | Tiny LP2 control |

## Planned on-disk layout

```text
data/fixtures/solved/
  ATTRIBUTION.md
  a-warning/
    manifest.json
    ciphertext.txt
    plaintext.txt
  some-wisdom/
    ...
  welcome/
    ...
  koan-1/
    ...
  loss-of-divinity/
    ...
  koan-2/
    ...
  an-instruction/
    ...
  an-end/
    ...
  lp2-57-identity/
    ...
```

### Manifest fields

Formal schema: [`docs/spec/fixtures.md`](../spec/fixtures.md)
(`parcae.fixture_manifest.v0`).

Summary:

- `id` — fixture id above  
- `method` — enum / string matching transform registry  
- `key` — optional Latin or index array  
- `skip_indices` — optional list into consumable rune stream (0-based); **must be recomputed before hash lock** (see [solved-methods.md](solved-methods.md) risk note)  
- `rune_index_base` — `0`  
- `ciphertext_sha256` / `plaintext_sha256` — locked digests after normalization  
- `source_attribution` — wiki / rtkd / etc.  
- `non_rune_literal_regions` — required for pages like `an-end` (hex hash) and number grids  

Until fixtures are committed, treat `an-end`’s hash announcement as covered by that
schema — not as a silent default without a manifest.

## Synthetic unit vectors (in addition to full pages)

Before full-page fixtures, tests should include **hand-computed** micro strings:

| Vector id | Purpose |
|-----------|---------|
| `synth-atbash-01` | Few runes, known `28 - x` |
| `synth-caesar-b3` | Shift ±3 |
| `synth-affine-a2b5` | Invertible affine round-trip |
| `synth-vig-no-skip` | Key `DI` or short key, no F |
| `synth-vig-with-skip` | One forced plaintext-F interrupt |
| `synth-totient-prefix` | First shifts `1,2,4,6,10,…` |

These live under `tests/data/synthetic/` (or Catch2 embedded literals), not necessarily under `data/fixtures/solved/`.

## Negative controls (required)

| Control id | Setup | Expected |
|------------|-------|----------|
| `welcome-ciphertext-f-all-skip` | Same Welcome ciphertext + `DIVINITY`, but skip **every** ciphertext F | Must **fail** plaintext match / hash |
| `an-end-no-f-skip` | Totient decrypt without pass-through on known interrupt | Must desync vs known plaintext |

These prove the interrupt policy is the documented plaintext-F rule.

## Explicitly deferred (not CPU-reference fixtures)

| Material | Reason |
|----------|--------|
| Full LP2 `0.jpg`–`55.jpg` transcripts | Unsolved; large; no oracle plaintext |
| Raw JPG / PNG page images | Not needed for Index29 CPU reference |
| Full LP1 page set beyond the nine ids | Nice-to-have; methods already covered |
| OutGuess / stego payloads from earlier Cicada years | Different problem domain |

## Attribution requirement

Every committed transcript must name its community source in
`data/fixtures/solved/ATTRIBUTION.md`. Cicada / Liber Primus imagery remains
third-party; Parcae stores **research transcripts** for reproducibility, not as
an assertion of copyright ownership over the puzzle art.

## Traceability

| Fixture id | Method doc section |
|------------|--------------------|
| `a-warning` | [solved-methods.md](solved-methods.md) §1 |
| `some-wisdom` | §2 |
| `welcome` | §3 |
| `koan-1` | §4 |
| `loss-of-divinity` | §5 |
| `koan-2` | §6 |
| `an-instruction` | §7 |
| `an-end` | §8 |
| `lp2-57-identity` | §9 |
