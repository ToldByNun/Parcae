# Solved Pages and Exact Methods

**Status:** Research freeze. These methods are the only **confirmed** Liber Primus
decrypt paths the CPU reference must reproduce.

Cipher arithmetic uses Gematria **indices** `0..28` modulo 29. See
[gematria-primus.md](gematria-primus.md).

## Interrupt rule (applies to keyed / streamed pages)

Multiple ciphers leave plaintext **F** (`ᚠ`, index 0) as cleartext and **do not**
advance the key or prime stream at that position.

| Rule | Welcome / Koan 2 behavior | Confidence |
|------|---------------------------|------------|
| Skip where **plaintext** is F | Correct — key stays periodic | **Working assumption** for fixtures (wiki wording: skipped F’s are cleartext / do not advance key) |
| Skip every **ciphertext** F | Wrong — destroys periodicity; true key looks false | Negative-control design target (must fail oracle) |
| Skip nothing | Wrong | Same |

Important: every interrupter appears as ciphertext F, but **most** ciphertext F
runes are ordinary encipherments that landed on F. Fixture skip lists are
**explicit index sets**, never “all F”.

**Primary provenance for the rule:** Uncovering Cicada wiki (“How the solved pages
were solved”) — cleartext F skips; key/stream does not advance. Boxentriq totient
docs describe the same pass-through for An End.

**Secondary essay:** [artwaste measured analysis](https://artwaste.land/strata/liber-primus-measured/)
argues the same plaintext-F vs ciphertext-F distinction with a periodicity table.
That page **exists** and was fetched, but it is a **single secondary analysis** —
treat its extra statistics as unverified (see [hypotheses.md](hypotheses.md)).
Do not cite artwaste alone as proof of interrupt semantics.

### Risk: skip index lists are not yet recomputed oracles

The concrete index lists below (Welcome / Koan 2) are copied from the **community
wiki** writeup. That is **not** the same as Parcae having verified:

```text
ciphertext + key + skip_indices  →  exact published plaintext
```

| Risk | Impact |
|------|--------|
| Off-by-one indexing (0-based vs 1-based; separators counted or not) | Entire Vigenère/totient fixtures become false ground truth |
| Transcript differs from the wiki’s source transcription | Same skip numbers apply to the wrong rune stream |
| Wiki list incomplete/outdated relative to another archive | Silent wrong plaintext “pass” if we also copy plaintext from the same page |

**Gate before locking golden hashes:** recompute Welcome and Koan 2 from a named
transcript file + key; derive or confirm `skip_indices` by matching known
plaintext; only then freeze manifests. Until that gate passes, lists are
**unverified inherited constants** (confidence: working draft, not frozen fact).

---

## Method catalog

### 1. A Warning — Atbash

| Field | Value |
|-------|-------|
| Fixture id | `a-warning` |
| Method | Atbash |
| Formula | `p[i] = 28 - c[i]` |
| Key / stream | none |
| Interrupts | none |

Plaintext theme: warning not to trust the book blindly; “believe nothing… except
what you know to be true”; do not edit words or numbers.

---

### 2. Some Wisdom — Identity

| Field | Value |
|-------|-------|
| Fixture id | `some-wisdom` |
| Method | Direct Gematria → Latin |
| Formula | identity on indices |
| Interrupts | none |

Includes the “CNOW THIS” number/word grid (numeric literals + runes). Numbers are
not cipher material.

---

### 3. Welcome — Vigenère + F skips

| Field | Value |
|-------|-------|
| Fixture id | `welcome` |
| Method | Vigenère (decrypt by subtraction) |
| Key (Latin) | `DIVINITY` |
| Key (runes) | `ᛞᛁᚢᛁᚾᛁᛏᚣ` |
| Key indices | `[23, 10, 1, 10, 9, 10, 16, 26]` |
| Formula | `p = (c - k[j]) mod 29` on consuming runes; `j` advances only when not interrupted |
| Skip indices (consumable rune stream, 0-based) | `48, 74, 84, 132, 159, 160, 250, 421, 443, 465, 514` |
| Skip list provenance | Uncovering Cicada wiki — **inherited, not yet recomputed in Parcae** |

Typical source images: `03.jpg` / `04.jpg` style Welcome pages (archive naming varies).

Encrypt direction (for round-trip tests): `c = (p + k[j]) mod 29` with the same
advance/skip rule.

---

### 4. Koan 1 — Atbash then Caesar +3

| Field | Value |
|-------|-------|
| Fixture id | `koan-1` |
| Method | Compose Atbash → Caesar |
| Formula | `t = 28 - c`; `p = (t + 3) mod 29` |
| Interrupts | none |

Equivalent single map: `p = (3 - c) mod 29` after simplifying, but the reference
implementation should implement **composition** explicitly to match the documented
solution path.

---

### 5. The Loss of Divinity — Identity

| Field | Value |
|-------|-------|
| Fixture id | `loss-of-divinity` |
| Method | Direct translation |
| Interrupts | none |

Thematic plaintext about consumption, preservation, adherence.

---

### 6. Koan 2 — Vigenère + F skips

| Field | Value |
|-------|-------|
| Fixture id | `koan-2` |
| Method | Vigenère |
| Key (Latin as used) | `FIRFUMFERENFE` |
| Key (runes) | `ᚠᛁᚱᚠᚢᛗᚠᛖᚱᛖᚾᚠᛖ` |
| Notes | Orthography of `CIRCUMFERENCE` with every **C → F** in the key spelling |
| Formula | `p = (c - k[j]) mod 29` |
| Skip indices | `49, 58` |
| Skip list provenance | Recomputed in Parcae against committed transcript (cicada_tools; wiki listed `49, 56`) |

Dictionary search for ordinary English words does **not** recover this key; the
C→F spelling is structural.

---

### 7. An Instruction — Identity

| Field | Value |
|-------|-------|
| Fixture id | `an-instruction` |
| Method | Direct translation |
| Interrupts | none |

Includes a symmetric decimal number grid under “CNOW THIS”.

---

### 8. An End (LP2 `56.jpg`) — Totient / prime−1 stream

| Field | Value |
|-------|-------|
| Fixture id | `an-end` |
| LP2 name | `56.jpg` (sometimes complete-set `73.jpg` in scream314 numbering) |
| Method | Running shift by \(\varphi(p) = p - 1\) for consecutive primes |
| Stream | primes `2, 3, 5, 7, 11, …` |
| Shift | `s[j] = (p_j - 1) mod 29` |
| Decrypt | `plain = (cipher - s[j]) mod 29` |
| Encrypt | `cipher = (plain + s[j]) mod 29` |
| Interrupts | plaintext-F pass-through: copy F unchanged; **do not** consume next prime |

Community note: skipping the wrong F desynchronizes the stream (classic example
around the 57th rune / prime 269 if mishandled). Fixture must encode the exact
interrupt positions used by the known solution.

Plaintext includes the deep-web hash announcement (long hex string treated as
literal, not runes).

---

### 9. LP2 `57.jpg` — Identity

| Field | Value |
|-------|-------|
| Fixture id | `lp2-57-identity` |
| Method | Direct translation |
| Interrupts | none |

Short control page; solved immediately after release via orthographic
transliteration.

---

## LP1 other solved pages

LP1 is widely reported as **17 solved pages**. Beyond the high-traffic methods
above, remaining LP1 material is generally the same families: identity, Atbash
(± shifts), and Vigenère-with-interrupts. The **minimum** oracle set is the nine
fixture ids in this document. Additional LP1 pages may be added later without
changing the method taxonomy.

## Unsolved material (not oracle fixtures)

| Range | Status |
|-------|--------|
| LP2 `0.jpg` … `55.jpg` | Unsolved (community) |

Do not treat experimental decrypts of these pages as validation fixtures.

## Sign / variant pitfalls (document for implementers)

| Pitfall | Effect |
|---------|--------|
| Vigenère as `k = p - c` instead of `k = c - p` | Loses Atbash-as-Beaufort-constant and related pages |
| Treating Atbash as unrelated to Beaufort | Harder to share code paths |
| Ciphertext-F skip on Welcome | ~13% rune accuracy with the true key |
| Advancing totient stream on escaped F | Garbles suffix of An End |

## References

- Uncovering Cicada wiki — solved page methods and skip lists (**primary** for Tier A; skip numbers still need Parcae recompute gate)
- CicadaSolvers — 56.jpg / 57.jpg summary
- Boxentriq totient tool — stream definition and F-skip
- artwaste essay — secondary only; exists but unverified for statistics / not sole interrupt proof
