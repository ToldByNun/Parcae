# Modulo-29 Hypotheses — Plausible vs Constrained

**Status:** Phase 0 research freeze.  
**Purpose:** Decide which transform families Phase 2 must implement, which Phase 1
should specify as generators, and which claims to avoid wasting cycles on.

All discussion assumes Gematria **indices** in \(\mathbb{Z}_{29}\).

---

## Tier A — Confirmed on solved Liber Primus material

These **must** exist as CPU transforms in Phase 2 and match published plaintexts.

| Family | Definition (decrypt-oriented) | Attested on |
|--------|-------------------------------|-------------|
| Identity | `p = c` | Some Wisdom; Loss of Divinity; An Instruction; LP2 57 |
| Atbash | `p = 28 - c` | A Warning |
| Caesar | `p = (c ± b) mod 29` | Component of Koan 1 |
| Compose | `T2 ∘ T1` | Koan 1 = Caesar(+3) ∘ Atbash |
| Explicit-key Vigenère | `p = (c - k[j]) mod 29`, `j` advances on consumed runes | Welcome (`DIVINITY`); Koan 2 (`FIRFUMFERENFE`) |
| Cleartext-F interrupt | On listed indices: emit F / pass through; do not advance key/stream | Welcome; Koan 2; An End |
| Totient / prime−1 stream | `s = (prime_j - 1) mod 29`; `p = (c - s) mod 29` | An End (`56.jpg`) |

### Affine (confirmed as algebra; not required for a named solved page)

Because 29 is prime, every `a ∈ 1..28` is invertible:

\[
p = a^{-1}(c - b) \pmod{29}
\]

Caesar is affine with `a = 1`. Atbash is affine with `a = 28 ≡ -1`, `b = 28`
under suitable encoding, or simply the dedicated map `28 - x`. Phase 2 should
implement affine as a **generator primitive** even if no single famous page is
“affine-only.”

### Beaufort

\[
p = (k - c) \bmod 29
\]

Atbash equals Beaufort with a constant key in the appropriate convention. Useful
shared implementation; avoids the historical sign-error class of bugs.

---

## Tier B — Plausible for generators / later search (not Phase 2 solve targets)

Safe to **specify** in Phase 1 and optionally stub-enumerate in Phase 2, but not
to claim as solutions for LP2 `0`–`55`:

| Hypothesis | Why it is plausible |
|------------|---------------------|
| Progressive / accumulating shifts | Natural extension of Caesar; common in classical crypto |
| Running key (text-autokey) | Fits Cicada’s love of structured streams |
| Key = Gematria label stream / self-key | Book already encodes its alphabet as numbers |
| Emirp or reversed-prime streams | Prime mystique appears in Cicada lore |
| Totient of composites / other φ schedules | Generalizes page 56 |
| Nested compositions (Atbash then Vigenère, etc.) | Already seen Atbash∘Caesar |
| Variable interrupt policies | F-skip is proven; other interrupters are conceivable |
| Fixed unknown permutation alphabets | Speculative; sometimes motivated by unverified anti-repeat essays (Tier C) |

Phase 2 stance: implement **applicators** and small **bounded generators**
(Caesar 29, affine 28×29, explicit key from caller). Do **not** build a full
dictionary attack engine yet.

---

## Tier C — Unverified single-source constraints (LP2 0–55)

**Confidence: unverified single-source — not a Phase 0 fact freeze.**

Source checked 2026-03-15: [One Bit of Structure: the Liber Primus, measured](https://artwaste.land/strata/liber-primus-measured/).

| Check | Result |
|-------|--------|
| Domain / URL reachable? | **Yes** — page returned a long essay (~39 KB text extract) |
| Claims independently recomputed by Parcae? | **No** |
| Peer-reviewed / multi-source corroboration? | **Not established** in this research pass |
| Allowed to block Phase 2 architecture? | **No** |
| Allowed as optional later heuristic / score idea? | **Yes**, if labeled |

The essay asserts, among other things, measurements on an unsolved rune body
(~12,956 runes):

| Claim on the page | How Parcae should treat it |
|-------------------|----------------------------|
| Self-follow (diagonal) rate ≈ **0.664%** vs chance **1/29 ≈ 3.45%** | Unverified hypothesis — may implement as a *score*, not as a hard reject rule until reproduced |
| Rune unigrams ≈ uniform; weak short-key IC | Unverified hypothesis |
| Additive Vigenère-class ciphers cannot reach that repeat floor under English-like plaintext | Unverified hypothesis — **do not** treat as proof that 0–55 cannot be keyed like Welcome |
| Natural GP algebra sits near ~1.8–2.8% repeats | Unverified hypothesis |
| Engineered anti-bigram alphabets as surviving speculative class | Speculative |
| OTP / running-key statistical neighborhood | Unverified hypothesis |

### Practical consequence for Parcae

1. Phase 2 validates **Tier A** oracles only — unaffected by Tier C.  
2. Optional: later add a self-repeat / diagonal-rate **score** for exploration.  
3. Do **not** discard Vigenère-style generators for unsolved pages solely because of
   this essay.  
4. Separately (wiki-attested): “skip all ciphertext F” is still a bad interrupt
   model on **solved** Welcome/Koan 2 — that stands without artwaste.  
5. Before elevating any Tier C number to “frozen fact,” reproduce it on a named
   Parcae transcript with a committed measurement script/test.

---

## Modulo-29 “folklore” checklist

| Claim | Verdict |
|-------|---------|
| “Everything is mod 29 because there are 29 runes” | **True** for index arithmetic (Tier A alphabet) |
| “Shift by the prime values themselves” | **False** as default; page 56 uses `(p-1) mod 29` |
| “Skip all F runes in ciphertext” | **False** for known solved keyed pages; use explicit plaintext-F skips |
| “Gematria prime sums solve pages” | **Not supported** by solved history |
| “Unsolved pages cannot be global additive Vigenère (artwaste wall)” | **Unverified single-source** — interesting, not frozen |

---

## Generator priority for Phase 1 / early Phase 2

1. Identity, Atbash, Caesar, Affine  
2. InterruptPolicy + explicit-key Vigenère / Beaufort  
3. Prime−1 / totient stream + interrupts  
4. Bounded compose  
5. (Later) progressive / running-key / permutation-alphabet families  

## References

- Solved method catalog — [solved-methods.md](solved-methods.md)
- Boxentriq / Uncovering Cicada wiki (Tier A provenance)
- artwaste essay — secondary, unverified statistics only
