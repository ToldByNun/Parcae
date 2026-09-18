# Spec: CPU ↔ CUDA Parity Contract

**Status:** Normative (CPU side now; CUDA later)  
**Related:** [transforms.md](transforms.md), [z29.md](z29.md)

## Goal

When CUDA lands, for every supported kernel:

```text
CPU_reference(input, params)  ==  CUDA_kernel(input, params)   (bit-identical Index29 outputs)
```

Score reductions used in batch mode MUST also match under a documented
associativity / ordering policy
([`cuda-score-reduction.md`](../architecture/cuda-score-reduction.md)).

## CPU obligations (parity-ready)

1. Hot transforms operate on `std::span<const Index29>` → `std::span<Index29>`
   (`apply_into` / static `kernel`) without hidden **output** allocations or
   global state. JSON parsing and keystream sieve setup may allocate; the
   Index29 write loop must not. CUDA twins live under `Parcae/Parcae/cuda/`
   (Visual Studio); see [`docs/architecture/cuda-abi.md`](../architecture/cuda-abi.md).
2. Interrupt application is explicit (mask or skip-set), not buried in I/O.
3. Params are POD / trivially serializable (JSON envelope ↔ struct).
4. Each transform run can emit a **parity record** (`ParityRecord` in
   `parcae/parity/parity_record.hpp`):

```json
{
  "parity_schema": "parcae.parity_record.v0",
  "transform_id": "caesar",
  "params_hash_sha256": "...",
  "input_sha256": "...",
  "output_sha256": "...",
  "interrupt_sha256": "...",
  "backend": "cpu"
}
```

Digest inputs (normative):

| Field | Bytes hashed |
|-------|----------------|
| `params_hash_sha256` | Compact `params.dump()` (no whitespace) |
| `input_sha256` / `output_sha256` | Raw `uint8_t` Index29 values in order |
| `interrupt_sha256` | Compact `interrupt.to_json().dump()` |

`ParityRecord::apply_and_capture` runs `ApplyTransform` then fills the record.
`backend` is `"cpu"` for the reference; CUDA ports MUST use `"cuda"` (or another
documented id) with identical remaining fields on a matching run.

5. Single-threaded CPU is the source of truth. Multi-thread CPU batches MUST
   document deterministic reduction (e.g. stable candidate_id order) before CUDA
   claims speedups.

## Buffer ABI sketch (Candidate Batch ABI v0 — contract only)

Future batches SHOULD use candidate-major structure-of-arrays:

| Buffer | Contents |
|--------|----------|
| `token_index29[C][T]` | indices per candidate (or shared tokens + params) |
| `consume_mask[C][T]` | uint8 boolean |
| `params_*` | SoA fields for shift/a/b/key schedules |
| `out_index29[C][T]` | results |
| `scores[C]` | optional |

Exact layout is finalized with the CUDA port; the CPU reference MUST NOT paint
itself into a corner with pointer-chasing tree transforms on the hot path.

## Equality definition

| Layer | Equality |
|-------|----------|
| Index streams | byte-identical `uint8_t` sequences |
| Scores | identical IEEE-754 binary64 after documented algorithm (no fast-math) |
| Latin text | identical UTF-8 after the same `to_latin` profile |

**MUST NOT** enable compiler flags that break determinism (`-ffast-math`,
unsafe FP contractions on score paths).

## CUDA port requirements (preview)

1. Parity tests: CPU vs CUDA on synthetic + solved-fixture index streams.
2. Shared golden `output_sha256` vectors committed under `data/parity/`.
3. No solve claims from CUDA alone — CUDA is an accelerator of the same functions.

## Explicit non-goals for the CPU reference

- No `.cu` files required to ship the CPU toolkit
- No GPU benchmarks as exit criteria for the CPU reference
- No Python reference plane (Parcae is C++-first)
