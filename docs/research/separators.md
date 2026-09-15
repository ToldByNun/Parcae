# Separator Grammar and Transcript Conventions

**Grammar id (planned data file):** `rtkd-separator-grammar-v0`  
**Status:** Phase 0 research freeze. Phase 2 fixtures use the **ASCII rtkd-style** convention.

## Why separators matter

Liber Primus pages are not raw rune streams. Word breaks, clause ends, line wraps,
and section markers appear in every serious transcription. Cipher arithmetic applies
only to **rune tokens**. Separators must be preserved as first-class tokens so that:

- Source text can round-trip
- Consumable masks exclude non-runes
- Interrupt indices refer to positions in the **consumable rune stream**, not raw bytes

## Canonical ASCII classes (Parcae Phase 2 default)

| Class | Token | Role |
|-------|-------|------|
| `word_separator` | `-` | Word break |
| `clause_separator` | `.` | Sentence / clause end |
| `paragraph_separator` | `&` | Paragraph / block break |
| `line_separator` | `/` | Logical line wrap as transcribed |
| `page_separator_or_marker` | `%` | Page/segment marker in some dumps |
| `chapter_separator` | `§` | Chapter mark when present |
| `whitespace` | space, tab | Rare in rune bodies; preserve if present |
| `physical_newline` | `\n` / `\r\n` | File formatting; may be normalized on load |
| `numeric_literal` | decimal digits | e.g. number grids on “Some Wisdom” / “An Instruction” |
| `hex_literal_candidate` | hex-looking runs | e.g. hash blocks on “An End” plaintext side |
| `unknown_symbol` | other non-rune | Preserve + warn; never silently drop |

### Preservation rules

1. Separators are **tokens**, not discarded noise.
2. `%` does **not** by itself prove a canonical page boundary policy for unsolved work.
3. `/` supports logical-line views; it is not proof of JPG page edges.
4. Unknown symbols produce warnings and remain in the token stream.

## Unicode punctuation map (alternate corpora)

Some transcriptions (e.g. LiberPrayground) use typographic separators:

| Unicode | Typical meaning | ASCII class map |
|---------|-----------------|-----------------|
| `•` (U+2022) | Word / space | `word_separator` |
| `⁘` (U+2058) | Period | `clause_separator` |
| `⁚` (U+205A) | Comma | treat as clause-adjacent separator (map policy in Phase 2 loader) |
| `⁖` (U+205D) | Semicolon | clause-adjacent separator |
| `⁜` (U+205C) | Chapter mark | `chapter_separator` |

Phase 0 decision: **solved fixtures committed in Phase 2 use ASCII `-` `.` `/` `%` `&`**.
A Unicode→ASCII normalizer may be added later; it is not required to start CPU work.

## Example (ASCII)

```text
ᚱ-ᛝᚱᚪᛗᚹ.ᛄᛁᚻᛖᛁᛡᛁ-ᛗᚫᚣᚹ-ᛠᚪᚫᚾ-/
```

Token sketch:

1. rune ᚱ  
2. word `-`  
3. runes ᛝ ᚱ ᚪ ᛗ ᚹ  
4. clause `.`  
5. …  
6. line `/`

Only runes enter Index29 streams for transforms.

## Interrupt indexing convention

Skip lists published for Welcome / Koan 2 / An End count **consumable runes** in
reading order (typically 0-based in modern tooling). Separators do **not** advance
that index. Fixture manifests in Phase 2 must state the indexing base explicitly
(`rune_index_base: 0`).

## Numbers and mixed pages

Several solved pages interleave runes with decimal tables (e.g. 272, 138, …). Those
digits are **not** Gematria indices. Tokenizer class: `numeric_literal`. They pass
through unchanged under identity pages and never consume Vigenère / totient state.

## References

- rtkd `liber-primus__transcription--master.txt`
- LiberPrimus-GPU `separator-grammar-v0` (prior art)
- relikd/LiberPrayground Unicode separator notes
