# Gematria Primus — Frozen Profile Narrative

**Profile id (planned data file):** `gematria-primus-v0`  
**Status:** Research freeze. Implementation lands as JSON under `data/profiles/`.

## What it is

Gematria Primus is the 29-symbol rune alphabet used throughout Liber Primus. Each
symbol has:

1. A **decimal index** in `0..28` — the arithmetic identity used by every mod-29 cipher
2. A **prime value** — the first 29 primes; metadata and keystream material, **not** the modulus
3. One or more **Latin labels** — display / orthography aliases

### Why 29, and why this order

This is **not** a generic Elder Futhark (24) / Younger Futhark / Anglo-Saxon Futhorc
inventory. Cicada’s Gematria Primus is a puzzle-specific 29-glyph encoding whose
order and primes are fixed by the published table. We freeze **this** table because:

- Solved Liber Primus methods (Atbash, Vigenère, totient stream) only reproduce
  known plaintext when indices match this ordering
- The modulus 29 is the size of **this** alphabet, not a historical rune count
- Reviewers sometimes ask “why not standard runes?” — answer: wrong problem;
  historical sets would break every oracle fixture

Confidence: **frozen fact** (multi-source: wiki, Boxentriq, scream314, community tools).

## Arithmetic domain

All cipher arithmetic in Parcae is over \(\mathbb{Z}_{29}\) on **indices**:

\[
x \in \{0,1,\ldots,28\}, \qquad \text{operations wrap modulo } 29
\]

**Do not** reduce prime values modulo 29 and treat that as the rune index. Primes
appear in streams such as page 56 (`shift = (p - 1) \bmod 29`), but the symbol
being shifted is always the index.

Community consensus (and solved pages) repeatedly confirm: wrap under/overflow by
mod 29 so values stay in `0..28`.

## Frozen 29-entry table

| Index | Prime | Rune | Preferred Latin | Aliases |
|------:|------:|------|-----------------|--------|
| 0 | 2 | ᚠ | F | |
| 1 | 3 | ᚢ | U | V |
| 2 | 5 | ᚦ | TH | |
| 3 | 7 | ᚩ | O | |
| 4 | 11 | ᚱ | R | |
| 5 | 13 | ᚳ | C | K |
| 6 | 17 | ᚷ | G | |
| 7 | 19 | ᚹ | W | |
| 8 | 23 | ᚻ | H | |
| 9 | 29 | ᚾ | N | |
| 10 | 31 | ᛁ | I | |
| 11 | 37 | ᛄ | J | |
| 12 | 41 | ᛇ | EO | |
| 13 | 43 | ᛈ | P | |
| 14 | 47 | ᛉ | X | |
| 15 | 53 | ᛋ | S | Z |
| 16 | 59 | ᛏ | T | |
| 17 | 61 | ᛒ | B | |
| 18 | 67 | ᛖ | E | |
| 19 | 71 | ᛗ | M | |
| 20 | 73 | ᛚ | L | |
| 21 | 79 | ᛝ | ING | NG |
| 22 | 83 | ᛟ | OE | |
| 23 | 89 | ᛞ | D | |
| 24 | 97 | ᚪ | A | |
| 25 | 101 | ᚫ | AE | |
| 26 | 103 | ᚣ | Y | |
| 27 | 107 | ᛡ | IA | IO |
| 28 | 109 | ᛠ | EA | |

## Latin orthography notes

Cicada solved plaintext is English written through this alphabet:

- **U / V:** rune ᚢ is often shown as `V` in plaintext dumps (`YOV`, `TRVE`)
- **C / K:** rune ᚳ covers both; dumps show `CNOW`, `BOOC`, etc.
- **S / Z:** rune ᛋ
- **Digraphs:** `TH`, `EO`, `OE`, `AE`, `ING`/`NG`, `IA`/`IO`, `EA` are single runes
- Multi-letter preferred labels are still **one** index in `0..28`

For validation hashing, Parcae documents a single **preferred label** fold (this
table’s Preferred column) so fixture digests are stable.

## Validation rules for the profile loader

A valid `gematria-primus-v0` profile must have:

- Exactly 29 entries
- Contiguous indices `0..28`
- Prime values equal to the first 29 primes (2 … 109)
- Unique rune codepoints
- Unique prime values
- Non-empty preferred Latin labels
- Bijective rune ↔ index and prime ↔ index lookups

## Glyph variants

Some fonts or transcriptions use variant forms (e.g. alternate J-rune glyphs).
The table above is canonical. Unknown glyphs are **errors in strict mode**, not
silent aliases, until a separate glyph-variant profile exists.

## What primes are *not*

Historical community note: raw gematria prime *sums* of words have not been the
method that opened solved Liber Primus pages. Parcae still stores primes because:

- The totient / prime−1 stream on `56.jpg` needs them
- Future generators may use prime schedules as keystreams
- Profile integrity checks use the prime list as a checksum of ordering

## Reversed Gematria / Atbash

Atbash on this alphabet is index reflection:

\[
x \mapsto 28 - x \pmod{29}
\]

That is equivalent to reading the alphabet backwards (EA↔F, IA↔U, …). Several
solved pages use Atbash alone or composed with a Caesar shift.

## References

- Boxentriq Gematria Primus translator table
- Uncovering Cicada wiki Gematria values table
- LiberPrimus-GPU `gematria-profile-v0` (prior art; independently re-stated here)
