# Spec: Interrupt / Cleartext-F Policy

**Status:** Normative  
**Headers (planned):** `parcae/interrupt/policy.hpp`  
**Research:** [solved-methods.md](../research/solved-methods.md)

## Definition

An **interrupt** is a consumable-rune position where the cipher **pass-through**
applies:

1. Output index equals the ciphertext index at that position (typically `F` = 0
   when plaintext-F was escaped), and
2. Key position / stream position **MUST NOT** advance.

This matches the documented Liber Primus behavior for Welcome, Koan 2, and An End.

## Policy object

```json
{
  "policy_id": "explicit_skip_indices_v0",
  "rune_index_base": 0,
  "skip_indices": [48, 74, 84]
}
```

| Field | Requirement |
|-------|-------------|
| `policy_id` | MUST be `explicit_skip_indices_v0` for solved fixtures |
| `rune_index_base` | MUST be `0` |
| `skip_indices` | Sorted unique non-negative integers; each MUST be `< consumable_rune_count` |

### Validation rules (conforming loaders MUST enforce)

1. All `skip_indices` in range.
2. No duplicates.
3. **SHOULD** check each skip position’s ciphertext rune is `F` (index 0). If not,
   fail fixture load (or warn+fail in strict validation) — wiki interrupters are
   ciphertext F.
4. **MUST NOT** accept a policy of “skip every ciphertext F” as equivalent to an
   explicit list.

## Forbidden shorthand policies (oracle fixtures)

| Policy | Status |
|--------|--------|
| `skip_all_ciphertext_f` | Allowed **only** as a **negative-control** method id, never as a solved fixture |
| `skip_none` | Valid policy (empty list), not an interrupter model |
| Inferred skips from plaintext without listing | Forbidden for golden fixtures |

## Interaction with transforms

For Vigenère / Beaufort / totient-stream:

```text
j = 0  // key or stream cursor
for each consumable position i:
  if i in skip_indices:
    out[i] = in[i]          // pass-through
    // j unchanged
  else:
    out[i] = decrypt(in[i], key_or_stream[j])
    j += 1
```

Non-consumable tokens never affect `j`.

## Encryption vs decryption

- **Encrypt (known plaintext):** interrupters are positions where plaintext is F;
  emit F and do not advance key/stream. Deterministic if plaintext is known.
- **Decrypt (ciphertext only):** a ciphertext F might be either an interrupt or a
  normal symbol that decrypts to something else. Golden fixtures **MUST** supply
  explicit `skip_indices` so decryption is deterministic.
- Search modes that enumerate interrupt hypotheses are **out of scope** for oracle
  fixtures (future search tooling may add them separately).

## Skip-list provenance gate

Inherited wiki lists are **draft** until the implementation proves:

```text
tokenize(ciphertext)
  → apply(method, key, skip_indices)
  → normalize_latin
  → equals(expected_plaintext)
```

Manifests **MUST NOT** lock `plaintext_sha256` until this gate passes for that
fixture. See [fixtures.md](fixtures.md) `verification.status`.

## Negative controls

| Control | Policy | Expected |
|---------|--------|----------|
| `welcome-ciphertext-f-all-skip` | skip every ciphertext F | MUST fail plaintext match |
| `an-end-no-f-skip` | empty skips while solution needs passes | MUST fail / desync |

## Relationship to artwaste essay

Statistical arguments from artwaste are **not** part of this normative interrupt
spec. Interrupt behavior here is defined by wiki/Boxentriq method descriptions and
fixture verification, not by unsolved-page diagonal rates.
