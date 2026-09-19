# CPU reference exit & CUDA handoff

Checklist for declaring the CPU reference complete, and for landing CUDA twins
that match it bit-for-bit.

**CUDA status:** twin catalog + batch + tools + fused search-run landed —
architecture in [`cuda-reference.md`](cuda-reference.md), device ABI in
[`cuda-abi.md`](cuda-abi.md). Exit tag `v0.3.0-cuda-parity` cut when twins +
property suite are green (see exit checklist below).  
**CUDA source home:** [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (Visual Studio).

See also: [`cpu-reference.md`](cpu-reference.md), [`docs/spec/parity.md`](../spec/parity.md).

---

## CPU reference exit criteria

All of the following SHOULD be true before tagging `v0.2.0-cpu-reference`
(or any successor CPU-reference tag):

### Oracles

- [x] Locked solved fixtures reproduce under Catch2 `[solved]`
- [x] Manifest SHA-256 locks present for ciphertext / plaintext / normalized Latin
- [x] Welcome negative control (ciphertext-F all-skip) fails as designed
- [x] `parcae-validate --all --require-locked` exits 0 locally

### Library surface

- [x] Transforms: identity, atbash, caesar, affine, compose, vigenere_key,
      beaufort_key, totient_prime_stream — with `apply_into` / `kernel`
- [x] Scores: exact_match, hamming_agreement, ic_mod29, chi2_english_gp_v0,
      self_repeat_rate via `ScoreRegistry`
- [x] Generators + `BatchRunner` (serial + ordered parallel)
- [x] `parcae::tool` — tokenize / apply / to_latin / score / validate_fixture
- [x] `ParityRecord` for params + I/O + interrupt digests

### Tools & docs

- [x] CLIs: `parcae-tokenize`, `parcae-decode`, `parcae-score`, `parcae-validate`
- [x] CLI smoke ctests
- [x] Architecture guide ([`cpu-reference.md`](cpu-reference.md))
- [x] This handoff checklist

### CI gate

- [x] Full `ctest` green on Ubuntu + Windows (`.github/workflows/ci.yml`)
- [x] `[solved]` green (subset of the above)

---

## CPU entry points that MUST gain CUDA twins

These are the **only** compute surfaces CUDA is required to mirror for parity.
Host-side UTF-8 / fixture I/O stays on CPU.

### Transform kernels (required)

| CPU entry | Header / symbol | Notes |
|-----------|-----------------|-------|
| `IdentityTransform::apply_into` / copy | `identity_transform.hpp` | Trivial |
| `AtbashTransform::kernel` | `atbash_transform.hpp` | Involution |
| `CaesarTransform::kernel` | `caesar_transform.hpp` | |
| `AffineTransform::kernel` | `affine_transform.hpp` | Needs `inv(a)` |
| `VigenereKeyTransform::kernel` | `vigenere_key_transform.hpp` | Key + skips |
| `BeaufortKeyTransform::kernel` | `beaufort_key_transform.hpp` | Key + skips |
| `TotientPrimeStreamTransform::kernel` | `totient_prime_stream_transform.hpp` | Precomputed shifts span |
| `ComposeTransform::apply_into` | `compose_transform.hpp` | Ping-pong stages; may call twins |
| `ApplyTransform::apply_into` | `apply_transform.hpp` | CPU dispatch |
| `CudaBackend::apply_into` | `Parcae/Parcae/cuda/backend.hpp` | Full catalog CUDA twin entry + `apply_and_capture` |

Keystream setup (may stay host-side if device receives a shifts buffer):

| CPU entry | Header |
|-----------|--------|
| `TotientKeystream::shifts_into` | `math/totient_keystream.hpp` |
| `Primes::first` / sieve | `math/primes.hpp` |

### Scores (required for batch search)

| CPU entry | Header |
|-----------|--------|
| `IcMod29::score` | `score/ic_mod29.hpp` → CUDA `IcMod29Score` |
| `Chi2EnglishGp::score` | `score/chi2_english_gp.hpp` → CUDA `Chi2EnglishGpScore` |
| `SelfRepeatRate::score` | `score/self_repeat_rate.hpp` → CUDA `SelfRepeatRateScore` |
| `ExactMatch::score` | `score/exact_match.hpp` → CUDA `ExactMatchScore` |
| `HammingAgreement::score` | `score/hamming_agreement.hpp` → CUDA `HammingAgreementScore` |
| `ScoreRegistry::score` | `score/score_registry.hpp` → CUDA `CudaScore` (dispatch) |

Document reduction associativity before claiming parallel CUDA score speedups
([`cuda-score-reduction.md`](cuda-score-reduction.md); formulas in
[`scores.md`](../spec/scores.md), equality in [`parity.md`](../spec/parity.md)).

### Batch / generators (required for search throughput)

| CPU entry | Header | Notes |
|-----------|--------|-------|
| Candidate expand (caesar / atbash / affine / …) | `generate/*.hpp` | Emit params + envelopes |
| `BatchRunner::run` | `batch/batch_runner.hpp` | CPU top-k; CUDA batches feed `scores[C]` |
| `BatchOrdering` | `batch/batch_ordering.hpp` | Deterministic top-k |
| `CudaBatchScore` | `Parcae/Parcae/cuda/cuda_batch_score.hpp` | Lane `CudaScore` + host `BatchOrdering` |
| `CandidateBatchBuffers` | `Parcae/Parcae/cuda/candidate_batch_buffers.hpp` | Host SoA ABI v0 (`kMaxC`/`kMaxT`) |
| `CaesarBatchKernel` | `Parcae/Parcae/cuda/caesar_batch_kernel.hpp` | Shared ciphertext + 29 shift lanes |
| `AtbashBatchKernel` | `Parcae/Parcae/cuda/atbash_batch_kernel.hpp` | Shared ciphertext → atbash lanes |
| `AtbashCaesarBatchKernel` | `Parcae/Parcae/cuda/atbash_caesar_batch_kernel.hpp` | Koan-1 compose 29 shifts |
| `AffineBatchKernel` | `Parcae/Parcae/cuda/affine_batch_kernel.hpp` | Shared ciphertext + 812 `(a,b)` lanes |
| `VigenereBatchKernel` | `Parcae/Parcae/cuda/vigenere_batch_kernel.hpp` | Explicit key-list SoA + shared interrupts |

### Explicitly **not** CUDA twins

- Tokenizer / UTF-8 / gematria profile load
- Fixture loader / validator / plaintext normalizer
- CLIs and `parcae::tool` wrappers (they **call** backends)
- JSON param parsing (cold); device gets POD / SoA

### Where twins are implemented

| Layer | Path |
|-------|------|
| CPU reference kernels | `include/parcae/transform/*.hpp` (and score/batch headers) |
| CUDA twins | **`Parcae/Parcae/cuda/`** inside the VS project |
| Shared ABI doc | [`cuda-abi.md`](cuda-abi.md) |

---

## Parity hash procedure

Goal: for a fixed `(transform_id, params, interrupt, direction, input)` prove

```text
CPU output bytes  ==  CUDA output bytes
```

and that both sides emit the same `ParityRecord` digests (except `backend`).

### 1. Materialize a golden on CPU

1. Choose a stream: synthetic Index29 vector **or** a locked fixture’s consumable
   ciphertext indices after tokenize.
2. Run:

   ```text
   ParityRecord::apply_and_capture(id, input, params, direction, interrupt, "cpu")
   ```

3. Persist under `data/parity/` (committed; regenerate with `parcae-parity-gen`), e.g.:

   ```text
   data/parity/<name>.json          # full ParityRecord JSON
   data/parity/<name>.out.idx29     # optional raw uint8 output bytes
   ```

   Replay / CUDA compare: `parcae-parity check [--compare-cuda]`.
   Dump a live record: `parcae-parity dump --name <golden> [--backend cuda]`.

4. Record `output_sha256` (and input/params/interrupt hashes) in the JSON.
   **Do not** rewrite locked solved-fixture hashes for CUDA experiments.

### 2. Replay on CUDA

1. Load the same input bytes, params POD, and skip indices onto the device.
2. Launch the twin kernel; copy output Index29 bytes back.
3. Build:

   ```text
   ParityRecord::capture(id, params, interrupt, input, cuda_output, "cuda")
   ```

4. Assert field equality:

   | Field | Must match CPU golden? |
   |-------|-------------------------|
   | `parity_schema` | yes |
   | `transform_id` | yes |
   | `params_hash_sha256` | yes |
   | `input_sha256` | yes |
   | `output_sha256` | **yes — this is the gate** |
   | `interrupt_sha256` | yes |
   | `backend` | differ (`cpu` vs `cuda`) |

### 3. Digest recipes (normative)

Copied from [`parity.md`](../spec/parity.md):

| Field | Bytes hashed (SHA-256, lowercase hex) |
|-------|----------------------------------------|
| `params_hash_sha256` | compact `params.dump()` (no whitespace) |
| `input_sha256` / `output_sha256` | consecutive `uint8_t` Index29 values |
| `interrupt_sha256` | compact `interrupt.to_json().dump()` |

### 4. Minimum golden set for first CUDA PR

| Case | Why |
|------|-----|
| Caesar shift=3, length 64 random | Elementwise |
| Atbash, length 64 | Involution |
| Vigenère key len 8 + 2 skips | Keyed + interrupt |
| Totient stream, start=0, 1 skip | Keystream |
| Affine `(a,b)=(2,5)` | Mul/inv |
| Compose atbash→caesar+3 | Multi-stage |
| One locked fixture consumable stream (e.g. `a-warning`) | Real length / orthography |

### 5. CI expectation (CUDA era)

- Job: build CPU + CUDA, run parity Catch2 tag (e.g. `[parity][cuda]`)
- Fail if any golden `output_sha256` mismatches (`[cuda][parity][golden]`)
- Fail if locked solved consumable streams diverge CPU↔CUDA (`[cuda][parity][solved]`)
- CPU-only runners still lock goldens/solved digests via the same tags (CUDA apply skipped)
- Hosted CI stays CPU-default; GPU runners exercise the CUDA branches

---

## Tagging

CPU reference milestone (when cut):

```text
v0.2.0-cpu-reference
```

CUDA parity milestone (CUDA exit):

```text
v0.3.0-cuda-parity
```

Prerequisites for the CPU tag: exit criteria above, including `[solved]` green on CI.  
Prerequisites for the CUDA tag: see **CUDA exit checklist** below and
[`cuda-reference.md`](cuda-reference.md).

---

## CUDA exit checklist

Declare CUDA parity complete (then cut `v0.3.0-cuda-parity`) when all of the
following hold. Architecture narrative: [`cuda-reference.md`](cuda-reference.md).

### Twins & tools

- [x] Full transform catalog through `CudaBackend`
- [x] Tier-A score twins + `CudaScore` dispatch
- [x] Candidate Batch ABI v0 + batch apply (caesar / atbash / atbash∘caesar /
      affine / vigenère) + `CudaBatchScore` top-k
- [x] `parcae-decode` / `parcae-score` `--backend cuda`
- [x] `parcae-parity` check/dump (+ `--compare-cuda`)
- [x] Device-resident fused χ² search-run (`parcae-search-run`, families above)

### Gates

- [x] Committed `data/parity/` goldens; `parcae-parity check --all` green
- [x] `[cuda][parity][golden]` + `[cuda][parity][solved]` green with Toolkit+GPU
- [x] Hosted CI CPU-default; `[cuda]` host stubs pass without Toolkit
- [x] CUDA property round-trips (roadmap commit 40)
- [x] Annotated tag `v0.3.0-cuda-parity` (commits 41–42)

### Docs

- [x] [`cuda-abi.md`](cuda-abi.md), [`cuda-score-reduction.md`](cuda-score-reduction.md),
      [`cuda-build.md`](cuda-build.md), [`cuda-roadmap.md`](cuda-roadmap.md)
- [x] [`cuda-reference.md`](cuda-reference.md) (this era’s architecture guide)
