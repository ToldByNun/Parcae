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

### `hill_2`

```json
{
  "transform_id": "hill_2",
  "direction": "decrypt",
  "params": { "matrix": [2, 3, 5, 7] }
}
```

\(2 \times 2\) Hill cipher over \(\mathbb{Z}_{29}\). `matrix` is row-major
`[a, b, c, d]` for \(\begin{pmatrix}a&b\\c&d\end{pmatrix}\).

| Direction | Formula (blocks of 2) |
|-----------|------------------------|
| `encrypt` | \(\mathbf{c} = A\cdot\mathbf{p}\) |
| `decrypt` | \(\mathbf{p} = A^{-1}\cdot\mathbf{c}\) |

Rules:

- Each entry MUST be in `0..28`.
- \(\det(A) \not\equiv 0 \pmod{29}\) (singular keys MUST hard-error).
- Input length MUST be even; odd length MUST hard-error (no implicit pad in v0).
- Interrupt policy is ignored (block cipher; same stance as `affine`).
- Params MUST contain only `matrix`.

### `hill_3`

```json
{
  "transform_id": "hill_3",
  "direction": "decrypt",
  "params": { "matrix": [1, 2, 3, 0, 1, 4, 5, 6, 0] }
}
```

\(3 \times 3\) Hill cipher over \(\mathbb{Z}_{29}\). `matrix` is row-major nine
entries for \(\begin{pmatrix}a&b&c\\d&e&f\\g&h&i\end{pmatrix}\).

| Direction | Formula (blocks of 3) |
|-----------|------------------------|
| `encrypt` | \(\mathbf{c} = A\cdot\mathbf{p}\) |
| `decrypt` | \(\mathbf{p} = A^{-1}\cdot\mathbf{c}\) |

Rules:

- Each entry MUST be in `0..28`.
- \(\det(A) \not\equiv 0 \pmod{29}\) (singular keys MUST hard-error).
- Input length MUST be a multiple of 3; other lengths MUST hard-error (no implicit pad in v0).
- Interrupt policy is ignored (block cipher; same stance as `hill_2` / `affine`).
- Params MUST contain only `matrix`.

### `ciphertext_autokey`

```json
{
  "transform_id": "ciphertext_autokey",
  "direction": "decrypt",
  "params": {
    "key_indices": [3, 5],
    "key_latin": optional
  },
  "interrupt": {
    "policy_id": "explicit_skip_indices_v0",
    "rune_index_base": 0,
    "skip_indices": []
  }
}
```

Ciphertext autokey (CTAK) over \(\mathbb{Z}_{29}\) with primer length \(L\).

| Direction | Key at consumed position \(j\) | Mix |
|-----------|--------------------------------|-----|
| `encrypt` | \(j < L\) → `key[j]`; else prior **ciphertext** at lag \(L\) | `out = add(in, key)` |
| `decrypt` | \(j < L\) → `key[j]`; else prior **ciphertext** at lag \(L\) | `out = sub(in, key)` |

Dense (empty skips) matches the CUDA `DeepScoreBatch` autokey hist kernel and
`CiphertextAutokeyKernel` dense decrypt (`AutokeyCtakDevice`): primer while absolute
index \(i < L\), then key = `ciphertext[i-L]`.

Rules:

- `key_indices` MUST be a non-empty array of integers in `0..28`.
- Optional `key_latin` is metadata only when `key_indices` is present.
- Interrupt skips pass through and do **not** consume primer/feedback (same cursor
  rule as `vigenere_key`).
- Empty primer MUST hard-error.

### `plaintext_autokey`

```json
{
  "transform_id": "plaintext_autokey",
  "direction": "decrypt",
  "params": {
    "key_indices": [3, 5],
    "key_latin": optional
  },
  "interrupt": {
    "policy_id": "explicit_skip_indices_v0",
    "rune_index_base": 0,
    "skip_indices": []
  }
}
```

Plaintext autokey (PTAK) over \(\mathbb{Z}_{29}\) with primer length \(L\).

| Direction | Key at consumed position \(j\) | Mix |
|-----------|--------------------------------|-----|
| `encrypt` | \(j < L\) → `key[j]`; else prior **plaintext** at lag \(L\) | `out = add(in, key)` |
| `decrypt` | \(j < L\) → `key[j]`; else prior **plaintext** at lag \(L\) | `out = sub(in, key)` |

Dense (empty skips): encrypt uses `input[i-L]` as feedback; decrypt uses already
written `output[i-L]`.

Rules: same as `ciphertext_autokey` for `key_indices` / `key_latin` / interrupts /
empty primer.

### `variable_delay_autokey`

```json
{
  "transform_id": "variable_delay_autokey",
  "direction": "decrypt",
  "params": {
    "key_indices": [3, 5],
    "key_latin": optional,
    "lag": 3,
    "lag_prime_index": optional,
    "mode": "ciphertext"
  },
  "interrupt": {
    "policy_id": "explicit_skip_indices_v0",
    "rune_index_base": 0,
    "skip_indices": []
  }
}
```

Variable-delay autokey over \(\mathbb{Z}_{29}\) with prime lag \(p \ge 2\).

Exactly one of `lag` or `lag_prime_index` MUST be set. `lag` MUST be prime.
`lag_prime_index` is 0-based (`0` → \(p=2\), `1` → \(p=3\), …) via `Primes::nth`.
`mode` defaults to `"ciphertext"`; allowed values: `"ciphertext"` | `"plaintext"`.

| Direction | Key at consumed position \(j\) | Mix |
|-----------|--------------------------------|-----|
| `encrypt` | \(j < p\) → `key[j \bmod L]`; else prior stream at lag \(p\) | `out = add(in, key)` |
| `decrypt` | same key rule | `out = sub(in, key)` |

Feedback stream by `mode`:

| `mode` | Encrypt feedback | Decrypt feedback |
|--------|------------------|------------------|
| `ciphertext` | prior **ciphertext** | input **ciphertext** |
| `plaintext` | prior **plaintext** | recovered **plaintext** |

When \(L = p\), mode `ciphertext` matches `ciphertext_autokey`; mode `plaintext`
matches `plaintext_autokey`.

Rules:

- `key_indices` MUST be a non-empty array of integers in `0..28`.
- Optional `key_latin` is metadata only when `key_indices` is present.
- Interrupt skips pass through and do **not** consume primer/feedback.
- Composite / non-prime `lag`, missing lag selector, or empty primer MUST hard-error.

### `spiral_read`

```json
{
  "transform_id": "spiral_read",
  "direction": "decrypt",
  "params": {
    "rows": 3,
    "cols": 4,
    "spiral": "inward"
  }
}
```

Pure index permutation on a `rows × cols` row-major grid (`N = rows·cols`).

| Direction | Semantics |
|-----------|-----------|
| `decrypt` | Read spiral order from row-major input (`out[i] = in[order[i]]`) |
| `encrypt` | Inverse (scatter spiral stream back to row-major) |

Spiral is clockwise starting at top-left. `spiral` defaults to `"inward"`; `"outward"`
is the reverse visit order. Interrupt policy is ignored.

Rules:

- `rows` / `cols` MUST be integers `≥ 1`.
- Input length MUST equal `rows * cols`.
- Params MUST contain only `rows`, `cols`, and optional `spiral`.

### `boustrophedon_read`

```json
{
  "transform_id": "boustrophedon_read",
  "direction": "decrypt",
  "params": {
    "rows": 3,
    "cols": 4,
    "first_row": "ltr"
  }
}
```

Pure index permutation: alternating row directions on a `rows × cols` row-major grid.

| Direction | Semantics |
|-----------|-----------|
| `decrypt` | Read boustrophedon order (`out[i] = in[order[i]]`) |
| `encrypt` | Inverse scatter |

`first_row` defaults to `"ltr"` (row 0 left→right, row 1 right→left, …). `"rtl"` flips
the parity. Interrupt policy is ignored.

Rules: same length / dimension constraints as `spiral_read`. Params MUST contain only
`rows`, `cols`, and optional `first_row`.

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
| `gen_beaufort_explicit_keys` | caller-supplied key list only (Beaufort; no dictionary) | \|keys\| |
| `gen_totient_offsets` | caller-supplied / bounded `prime_start_index` list | small |
| `gen_compose_recipes` | empty → Atbash∘Caesar 29; or `recipes` / `stages` / `template` | \|recipes\| or 29 |

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
