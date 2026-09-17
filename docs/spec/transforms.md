# Spec: Transform Families and Parameter Schemas

**Status:** Normative  
**Headers (planned):** `parcae/transform/*.hpp`  
**Research:** [hypotheses.md](../research/hypotheses.md), [solved-methods.md](../research/solved-methods.md)

## Common contract

Every transform **MUST**:

1. Accept a consumable `span<const Index29>` (and optional interrupt policy).
2. Write results into a caller-provided `span<Index29>` of the **same length**
   via `apply_into` (CUDA-ready; no hidden output allocation on the hot path).
3. Be a pure function of `(input, params, policy)` — same bits in ⇒ same bits out.
4. Serialize params to/from JSON without loss for all fields defined below.
5. Expose `direction`: `encrypt` | `decrypt` where both are defined.

Convenience `apply(...) → vector` MAY allocate once and MUST delegate to
`apply_into`. Static `kernel(...)` helpers take POD / `span` args only (no JSON)
and MUST NOT allocate. JSON key / keystream materialization is setup, not the
elementwise loop. In-place (`out.data() == in.data()`) is allowed for single-pass
families. `compose` uses at most two length-N scratch buffers (ping-pong).

Non-rune tokens are handled by the corpus layer, not inside Index29 kernels.

### Envelope

```json
{
  "transform_id": "vigenere_key",
  "direction": "decrypt",
  "params": { },
  "interrupt": {
    "policy_id": "explicit_skip_indices_v0",
    "rune_index_base": 0,
    "skip_indices": []
  }
}
```

`interrupt` **MAY** be omitted when equivalent to empty `skip_indices`.

---

## Family catalog (CPU reference required)

### `identity`

```json
{ "transform_id": "identity", "direction": "decrypt", "params": {} }
```

`out[i] = in[i]`.

### `atbash`

```json
{ "transform_id": "atbash", "direction": "decrypt", "params": {} }
```

`out[i] = 28 - in[i]`. Direction ignored (involution); still accepted for uniformity.

### `caesar`

```json
{
  "transform_id": "caesar",
  "direction": "decrypt",
  "params": { "shift": 3 }
}
```

| Direction | Formula |
|-----------|---------|
| `decrypt` | `out = sub(in, shift)` |
| `encrypt` | `out = add(in, shift)` |

`shift` MUST be in `0..28`.

### `affine`

```json
{
  "transform_id": "affine",
  "direction": "decrypt",
  "params": { "a": 2, "b": 5 }
}
```

| Direction | Formula |
|-----------|---------|
| `encrypt` | `out = add(mul(a, in), b)` |
| `decrypt` | `out = mul(inv(a), sub(in, b))` |

`a` MUST be in `1..28`; `b` in `0..28`.

### `compose`

```json
{
  "transform_id": "compose",
  "direction": "decrypt",
  "params": {
    "stages": [
      { "transform_id": "atbash", "params": {} },
      { "transform_id": "caesar", "direction": "encrypt", "params": { "shift": 3 } }
    ]
  }
}
```

Semantics: stages describe the **decrypt** pipeline and are applied in array order.
Optional per-stage `direction` overrides the default (`decrypt`); Koan 1 marks the
Caesar stage `"direction": "encrypt"` so decrypt performs Atbash then **+3**. Outer
`encrypt` reverses stage order and inverts each stage direction.

Constraints:

- Nested `compose` MUST be allowed to a small depth (implementation MUST document
  max depth ≥ 4).
- Interrupt policy applies to the **outer** keyed/streamed stage only unless a
  stage carries its own `interrupt` object (first cut: only outer policy required).

### `vigenere_key`

```json
{
  "transform_id": "vigenere_key",
  "direction": "decrypt",
  "params": {
    "key_indices": [23, 10, 1, 10, 9, 10, 16, 26],
    "key_latin": "DIVINITY"
  },
  "interrupt": {
    "policy_id": "explicit_skip_indices_v0",
    "rune_index_base": 0,
    "skip_indices": [48, 74, 84, 132, 159, 160, 250, 421, 443, 465, 514]
  }
}
```

| Direction | On consumed (non-skip) position with key cursor `j` |
|-----------|------------------------------------------------------|
| `decrypt` | `out = sub(in, key[j mod L])` |
| `encrypt` | `out = add(in, key[j mod L])` |

Rules:

- Prefer `key_indices` if both present; if only `key_latin`, map through Gematria
  preferred labels (multi-letter greedy left-to-right as defined + tested in code).
- Empty key MUST hard-error.
- Interrupt rules per [interrupts.md](interrupts.md).

### `beaufort_key`

```json
{
  "transform_id": "beaufort_key",
  "direction": "decrypt",
  "params": { "key_indices": [1] }
}
```

On consumed positions: `out = sub(key[j], in)` (both directions are the same map
for classic Beaufort — document and test involution carefully). Required in the
CPU reference; not required for a named solved fixture.

**Sign pitfalls (wiki / community):** Liber Primus Vigenère decrypt is
`p = (c - k) mod 29`. Using `(k - c)` instead is Beaufort and fails Welcome /
Koan-2. Conversely, Atbash `28 - x` **is** Beaufort with constant key index
`28` — keep that identity so Atbash and Beaufort stay one algebraic family.

### `totient_prime_stream`

```json
{
  "transform_id": "totient_prime_stream",
  "direction": "decrypt",
  "params": {
    "prime_start_index": 0,
    "shift_mode": "prime_minus_one_mod_29"
  },
  "interrupt": {
    "policy_id": "explicit_skip_indices_v0",
    "rune_index_base": 0,
    "skip_indices": []
  }
}
```

Stream:

```text
primes p0=2, p1=3, p2=5, ...
s_j = (p_j - 1) mod 29
```

| Direction | Consumed step |
|-----------|---------------|
| `decrypt` | `out = sub(in, s_j)` then `j++` |
| `encrypt` | `out = add(in, s_j)` then `j++` |

`prime_start_index` MUST default to `0` (first prime = 2). Skips do not consume.

---

## Candidate generators (CPU, bounded)

Generators emit deterministic sequences of transform envelopes.

| Generator id | Enumeration | Cost |
|--------------|-------------|------|
| `gen_caesar` | all `shift` in `0..28` | 29 |
| `gen_atbash` | single candidate | 1 |
| `gen_atbash_caesar` | Atbash ∘ Caesar(+shift) for all shifts (Koan-1 family) | 29 |
| `gen_affine` | all `a∈1..28`, `b∈0..28` (nested `a` then `b`) | **28×29 = 812** |
| `gen_vigenere_explicit_keys` | caller-supplied key list only (no dictionary expansion) | \|keys\| |
| `gen_totient_offsets` | optional small range of `prime_start_index` (bounded) | small |

`gen_affine` is the largest Tier-A monoalphabetic sweep in the CPU reference.
Callers MUST treat 812 as an explicit budget (score/batch), not an unbounded
search. `gen_vigenere_explicit_keys` is an **applicator** for keys the caller
already enumerated — it MUST NOT become a dictionary search engine.

Generators **MUST NOT** silently run unbounded dictionary search.

Output record:

```json
{
  "candidate_id": "caesar:shift=3",
  "envelope": { "transform_id": "caesar", "direction": "decrypt", "params": { "shift": 3 } },
  "output_indices": [ /* ... */ ]
}
```

---

## Fixture method → transform map

| Fixture id | Envelope |
|------------|----------|
| `a-warning` | `atbash` |
| `some-wisdom` | `identity` |
| `loss-of-divinity` | `identity` |
| `an-instruction` | `identity` |
| `lp2-57-identity` | `identity` |
| `koan-1` | `compose[atbash, caesar shift=3]` |
| `welcome` | `vigenere_key` + skips |
| `koan-2` | `vigenere_key` + skips |
| `an-end` | `totient_prime_stream` + skips |

## Explicit non-targets (CPU reference)

- CUDA kernels
- LLM-proposed transforms without a schema id
- Mutating global alphabet order mid-run
