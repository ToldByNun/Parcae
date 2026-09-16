# An End (`56.jpg`) — interrupt notes

**Fixture:** `an-end`  
**Method:** `totient_prime_stream` (`φ(p) = p - 1` mod 29)  
**Status:** draft — skip list inherited, not yet recomputed in Parcae

## Policy

Solved An End uses **plaintext-F pass-through**:

1. At each listed consumable index, emit/copy **F** (`ᚠ`, index 0) unchanged.
2. **Do not** advance the prime / totient stream at that position.
3. Skip lists are **explicit index sets**, never “every ciphertext F”.

Indexing is **0-based into the consumable rune stream** (separators and the
embedded hex hash do not advance the interrupt index).

## Skip list (draft)

| Source | `skip_indices` |
|--------|----------------|
| cicada_tools `016_an_end` | `[56]` |
| Community note (wiki / Boxentriq) | Wrong skip around the “57th rune” / prime **269** desyncs the suffix |

Parcae draft manifest uses **`[56]`** (0-based ≡ the 57th consumable rune).

## Failure mode

Advancing the totient stream on the interrupter (or skipping all ciphertext F)
garbles everything after the interrupt — classic An End regression.

## Hex literal

The deep-web hash block is **not** cipher material. It is declared as
`non_rune_literal_regions` → `literals/deep-web-hash.txt` so tokenizer /
validators keep it as a hex literal, not `Index29` runes.
