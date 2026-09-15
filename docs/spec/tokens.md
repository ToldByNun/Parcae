# Spec: Token Model and Consumable Masks

**Status:** Normative  
**Headers (planned):** `parcae/corpus/token.hpp`, `parcae/corpus/tokenizer.hpp`  
**Research:** [separators.md](../research/separators.md)

## Goals

1. Preserve transcript structure (separators, numbers, literals).
2. Expose a **consumable rune stream** for cipher arithmetic.
3. Keep stable indices for interrupt lists (`skip_indices`).

## `TokenKind`

| Kind | Meaning | Consumable? |
|------|---------|-------------|
| `Rune` | One Gematria glyph → `Index29` | **Yes** |
| `WordSep` | `-` (ASCII grammar) | No |
| `ClauseSep` | `.` | No |
| `ParaSep` | `&` | No |
| `LineSep` | `/` | No |
| `PageMark` | `%` | No |
| `ChapterSep` | `§` | No |
| `Whitespace` | space / tab | No |
| `Newline` | `\n` / `\r\n` | No |
| `Number` | decimal digit run | No |
| `HexLiteral` | hex-looking run (policy below) | No |
| `Unknown` | other non-rune | No |

Unicode separator aliases (`•` `⁘` …) **MAY** be accepted by mapping to the ASCII
kinds above. Solved fixtures **MUST** use ASCII separators; Unicode support is
optional sugar with tests if enabled.

## `Token` record

Each token **MUST** carry at least:

| Field | Type / meaning |
|-------|----------------|
| `kind` | `TokenKind` |
| `index29` | present iff `kind == Rune` |
| `byte_begin` / `byte_end` | UTF-8 byte offsets into source |
| `source_slice` | view or copy of original bytes (for round-trip) |
| `consumable_index` | `optional<size_t>` — rank among consumable runes only |

### Consumable index

- Assigned in reading order over tokens with `kind == Rune` only.
- First consumable rune has `consumable_index == 0` when `rune_index_base == 0`.
- Separators and literals **MUST NOT** receive a consumable index.
- Interrupt `skip_indices` refer to **consumable_index**, never byte offsets.

## `TokenStream`

- Ordered sequence of `Token`.
- **MUST** round-trip to original text under the ASCII grammar for fixture
  transcripts (byte-identical if no Unicode normalization was applied).

## `ConsumableMask` / rune view

Implementations **MUST** provide a view equivalent to:

```text
indices:  span<Index29>   // length = number of Rune tokens
mask:     span<bool>      // same length; true = participates in transform
```

Default: all runes masked `true`. Interrupt handling may temporarily treat a
position as pass-through without removing it from the stream (see
[interrupts.md](interrupts.md)).

Transforms **MUST** iterate only positions where the effective policy says
“consume,” and **MUST** copy non-rune tokens unchanged when rebuilding text.

## Number and hex policy

| Input | Kind | Notes |
|-------|------|-------|
| `[0-9]+` | `Number` | e.g. wisdom/instruction grids |
| Long `[0-9a-fA-F]+` meeting fixture/hex rules | `HexLiteral` | e.g. An End hash in plaintext views |

Exact hex detection heuristics **SHOULD** be conservative: prefer fixture-declared
`non_rune_literal_regions` ([fixtures.md](fixtures.md)) over guessing.

## Strict mode

Tokenizer **MUST** support a strict mode that rejects unknown runes / unexpected
glyphs with a hard error. Fixture validation runs in strict mode.

## Out of scope

- JPG OCR
- Page-boundary inference from `%` alone
- Mutating separator semantics per archive without a grammar id
