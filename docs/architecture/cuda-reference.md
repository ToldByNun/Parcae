# CUDA reference architecture

How CUDA twins attach to the CPU Liber Primus toolkit: layout under
[`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/), device-resident batch search,
and the parity gate before tagging `v0.3.0-cuda-parity`.

CPU remains the source of truth ([`cpu-reference.md`](cpu-reference.md)).
Normative ABI: [`cuda-abi.md`](cuda-abi.md). Reduction rules:
[`cuda-score-reduction.md`](cuda-score-reduction.md). Twin checklist + hash
procedure: [`cuda-handoff.md`](cuda-handoff.md). Build notes:
[`cuda-build.md`](cuda-build.md).

**Status:** twin catalog + batch SoA + tool backends + fused search-run path
landed; exit tag not yet cut (see [Exit checklist](#cuda-exit-checklist)).

## Design rules

1. **Same Index29 bits** — for a fixed `(id, params, interrupt, direction, input)`,
   CUDA output bytes MUST equal CPU `apply_into` / `kernel` bytes.
2. **Same score IEEE bits** — Tier-A scores via integer histograms / counts then
   fixed-order FP finalize ([`cuda-score-reduction.md`](cuda-score-reduction.md)).
3. **Host owns UTF-8 / fixtures** — tokenize, gematria, validators stay on CPU;
   device sees `uint8_t` Index29 streams and POD params.
4. **Interrupts are uploaded, never inferred** — `InterruptDeviceView` bitmask or
   sorted skips; keyed families advance the key cursor the same way as CPU.
5. **CI stays CPU-default** — `PARCAE_BUILD_CUDA` off on hosted runners; GPU is
   local / optional self-hosted (`cuda-build.md`).
6. **Hot search path minimizes PCIe** — setup H2D once; timed work stays on device;
   D2H only `scores[C]` (or small histograms), not full `out[C·T]`, when using
   fused family χ² batches.

## Module map

```text
Parcae/Parcae/cuda/
  device_buffer.hpp          RAII cudaMalloc / H2D / D2H → Status
  cuda_error.hpp             cudaError_t → Status
  params.hpp / params_json   POD families mirroring JSON
  interrupt_device_view.hpp  bitmask (T≤4096) | sorted skips
  z29_device.hpp             Z29 add/sub/mul/inv on device
  backend.hpp                CudaBackend::apply_into (+ capture)
  identity_copy / atbash / caesar / affine / …
  *_key_kernel / totient / compose_driver
  candidate_batch_buffers.hpp   Host SoA ABI v0 (C×T)
  *_batch_kernel.*              Batch apply twins
  cuda_score.hpp / *_score.*    Per-stream score twins
  cuda_batch_score.hpp          Lane scores + host BatchOrdering
  chi2_batch_score.*            Multi-block hist from out[C·T] + finalize
  caesar_chi2_batch.*           Fused Caesar decrypt→χ² (no out materialize)
  family_chi2_batch.*           Fused atbash / atbash∘caesar / affine / vigenère

include/parcae/run/
  search_run*.hpp            Throughput + sweep + fixture-eval dashboard
  search_run_cuda.hpp        CUDA family sweeps (device-resident)

tools/
  parcae-decode | parcae-score     --backend cpu|cuda
  parcae-parity                    check / dump (+ --compare-cuda)
  parcae-search-run                AI-style throughput / score / eval console
```

| Layer | Role | Parity? |
|-------|------|---------|
| Single-stream kernels + `CudaBackend` | Catalog twins | Byte-identical Index29 |
| Score twins + `CudaScore` | Tier A | Bit-identical `double` |
| Batch SoA apply + `CudaBatchScore` | Search expand | Index29 + scores vs CPU |
| Fused `*Chi2Batch` / `SearchRun` | Throughput dashboard | χ² vs `ScoreRegistry` |

## Data flow

### Single-stream parity (oracle / `parcae-parity`)

```text
ParityRecord golden (data/parity/*.json)
        │
        ├─► CPU ApplyTransform / ParityRecord::apply_and_capture("cpu")
        │
        └─► CudaBackend::apply_into ── D2H ── capture("cuda")
                    │
                    ▼
            output_sha256 MUST match (backend field may differ)
```

### Batch SoA (ABI v0)

```text
shared ciphertext[T] + param lanes[C]   (CandidateBatchBuffers, host)
        │
        ▼ H2D
 device_in / device_params / device_out[C·T]
        │
        ▼ batch kernel (candidate-major)
 device_out lanes
        │
        ├─► CudaBatchScore::score_lanes ──► scores[C] ──► host top-k
        │
        └─► (optional) Chi2BatchScore::score_from_out_async
```

### Fused search-run (device-resident χ²)

```text
setup (untimed): H2D cipher + params + expected probs; alloc counts/scores
        │
        ▼ timed loop (async queue, one sync at end)
 FamilyChi2Batch / CaesarChi2Batch
   decrypt on-the-fly → shared hist → counts[C·29] → finalize scores[C]
        │
        ▼ D2H scores[C] only (~8·C bytes)
 SearchRunConsole / metrics (tok/s, sweep bars, fixture eval, CPU↔CUDA)
```

CLI: `parcae-search-run --backend cuda --family caesar|atbash|atbash_caesar|affine|vigenere`.

## Backend switch

| Surface | CPU | CUDA |
|---------|-----|------|
| `parcae::tool::Backend` | default | requires `PARCAE_HAS_CUDA` |
| `parcae-decode` / `parcae-score` | `--backend cpu` | `--backend cuda` (exit 2 if not built) |
| `parcae-parity check` | golden digests | `--compare-cuda` |
| `parcae-search-run` | `family=caesar` only | all fused families |

## Tests (CUDA era)

| Tag | Role |
|-----|------|
| `[cuda][parity][*]` | Per-family / golden / solved consumable digests |
| `[cuda][score][*]` | Tier-A twins + noise separation |
| `[cuda][batch][*]` | SoA apply, batch score, rank-1 recovery |
| `[cuda][batch][chi2][fuse]` | Fused Caesar χ² vs CPU |
| `[run][search]` | SearchRun CPU + CUDA (+ family parity) |
| `cli_*_cuda` ctests | decode / score / parity / search-run smoke |

## CUDA exit checklist

Prerequisites for annotated tag `v0.3.0-cuda-parity` (also mirrored in
[`cuda-handoff.md`](cuda-handoff.md)):

### Twins

- [x] Transform catalog via `CudaBackend` (identity, atbash, caesar, affine,
      vigenère, beaufort, totient, compose)
- [x] Score twins + `CudaScore` dispatch (exact, hamming, ic, self_repeat, chi²)
- [x] Batch SoA buffers + caesar / atbash / atbash∘caesar / affine / vigenère batch
- [x] `CudaBatchScore` + deterministic host top-k
- [x] Tool `--backend cuda` + `parcae-parity` dump/compare
- [x] Device-resident fused χ² search-run (`parcae-search-run`)

### Parity gates

- [x] `data/parity/` goldens + `parcae-parity check [--compare-cuda]`
- [x] `[cuda][parity][golden]` / `[cuda][parity][solved]` green with Toolkit+GPU
- [x] Hosted CI remains CPU-default; `[cuda]` host stubs still pass

### Docs & release

- [x] This reference + ABI / reduction / build / handoff
- [ ] Property round-trips on CUDA (roadmap commit 40)
- [ ] Version bump + annotated tag `v0.3.0-cuda-parity` (commits 41–42)

## Related docs

- [`cpu-reference.md`](cpu-reference.md) — CPU module map (source of truth)
- [`cuda-handoff.md`](cuda-handoff.md) — twin list + hash procedure
- [`cuda-abi.md`](cuda-abi.md) — buffer shapes / interrupts
- [`cuda-score-reduction.md`](cuda-score-reduction.md) — FP / histogram rules
- [`cuda-build.md`](cuda-build.md) — local Toolkit / CMake
- [`cuda-roadmap.md`](cuda-roadmap.md) — frozen commit plan
- [`docs/spec/tools.md`](../spec/tools.md) — CLI contracts incl. search-run
