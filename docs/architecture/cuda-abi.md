# CUDA device ABI (v0)

**Status:** Frozen for CUDA implementation  
**Home for `.cu` / CUDA headers:** [`Parcae/Parcae/`](../../Parcae/Parcae/) (Visual Studio project), not a separate CMake-only tree.

CPU contracts remain in [`docs/spec/parity.md`](../spec/parity.md) and
[`cuda-handoff.md`](cuda-handoff.md). This file locks the **on-device buffer
shapes** so kernels and host upload code share one vocabulary.

## Source layout (locked)

```text
Parcae/Parcae/                 # VS: Parcae.vcxproj / Parcae.slnx
  main.cpp                     # host entry (today: smoke)
  cuda/                        # CUDA twins land here
    README.md
    …kernels, device helpers…
include/parcae/                # CPU reference (header-only) — unchanged
```

Public CPU headers stay under `include/parcae/`. Device code and CUDA-facing
host glue live in the VS project so local CUDA builds use the existing
`Parcae/Parcae` toolchain. CMake may later *also* compile those same sources
when `PARCAE_BUILD_CUDA=ON`; the **canonical tree** is `Parcae/Parcae/cuda/`.

## Element types

| Logical | Device storage | Notes |
|---------|----------------|-------|
| `Index29` | `uint8_t` in `{0…28}` | Same as CPU `Index29::value()` |
| Direction | `uint8_t` enum | `0 = decrypt`, `1 = encrypt` (match host) |
| Skip index | `uint32_t` | Consumable rune index; sorted ascending, unique |
| Score | `double` (IEEE-754 binary64) | No fast-math; bit-identical to CPU |

Alignment: Index29 streams are tightly packed `uint8_t` with no padding.
SoA param fields use natural alignment for their types (`uint8_t` / `uint32_t` /
`double`).

## Single-stream buffers

For one transform apply (parity goldens, fixture streams):

| Buffer | Shape | Owner |
|--------|-------|-------|
| `in[T]` | `uint8_t` | host upload → device |
| `out[T]` | `uint8_t` | device → host download |
| `key[K]` | `uint8_t` | keyed families only |
| `skips[S]` | `uint32_t` sorted | or bitmask when `T ≤ 4096` |
| `shifts[J]` | `uint8_t` | totient: **host-materialized** via `TotientKeystream::shifts_into` |

`T` = consumable length. Interrupt policy never inferred on device.

### Interrupt representation

1. **Preferred for fixture-scale** (`T ≤ 4096`): bitmask `uint32_t` words,
   bit `i` set ⇒ skip consumable index `i` (no key/stream advance).
2. **Fallback:** sorted unique `uint32_t skips[S]` + binary search (same as CPU
   `InterruptPolicy::should_skip`).

Host converts from `InterruptPolicy` before H2D.

## Candidate Batch ABI v0 (SoA)

Candidate-major structure-of-arrays (see also `parity.md`):

| Buffer | Shape | Meaning |
|--------|-------|---------|
| `token_index29[C][T]` **or** shared `token_index29[T]` + per-candidate params | indices | |
| `consume_mask[C][T]` | `uint8` boolean | optional if all tokens consumable |
| `params_*` | SoA fields | shift / a,b / key schedule ids |
| `out_index29[C][T]` | `uint8` | results |
| `scores[C]` | `double` | optional after score pass |

- `C` = candidate count, `T` = tokens (or consumable length).
- v0 MAY use shared ciphertext tokens + per-candidate param lanes (caesar 29,
  affine 812) instead of replicating `token_index29` per candidate.
- Top-k reduction in v0 MAY run on host after D2H of `scores[C]`, using
  `BatchOrdering` (CPU serial remains source of truth).

## Direction enum (device)

```text
PARCAE_DIR_DECRYPT = 0
PARCAE_DIR_ENCRYPT = 1
```

Must match `TransformDirection` string mapping used in parity records
(`decrypt` / `encrypt` on the host JSON side).

## Transform family → device entry (twins)

| Family | Device inputs beyond `in`/`out` |
|--------|----------------------------------|
| identity | — |
| atbash | — |
| caesar | `shift`, direction |
| affine | `a`, `b`, direction |
| vigenere_key | `key[K]`, skips/bitmask, direction |
| beaufort_key | `key[K]`, skips/bitmask |
| totient_prime_stream | `shifts[J]`, skips/bitmask, direction |
| compose | host launches stage sequence + ping-pong buffers (no JSON on device) |

## Out of ABI scope

- UTF-8 / tokenizer / gematria JSON
- Fixture I/O and `ValidationReport`
- CLIs (they call host façade → CPU or CUDA backend)
