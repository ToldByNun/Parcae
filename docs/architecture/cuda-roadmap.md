# CUDA plan freeze

**Status:** Frozen start of CUDA work  
**CPU reference:** complete (oracles, span kernels, CLIs, `ParityRecord`)  
**Implementation home:** [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (Visual Studio)

North star:

```text
CPU_reference(input, params, interrupt, direction)
  ==  CUDA_twin(...)     (byte-identical Index29; identical IEEE-754 scores)
```

Authoritative twin list and parity hash procedure:
[`cuda-handoff.md`](cuda-handoff.md).  
Device buffer shapes: [`cuda-abi.md`](cuda-abi.md).

## Locked decisions

| Topic | Decision |
|-------|----------|
| Source tree | CUDA `.cu` / device helpers under **`Parcae/Parcae/cuda/`** (VS project) |
| CPU headers | Stay under `include/parcae/` (header-only reference) |
| Build | VS CUDA project primary for day-to-day; CMake may optionally compile the same sources later (`PARCAE_BUILD_CUDA`) |
| Keystream | Totient shifts materialized **on host**; device receives shifts buffer |
| Compose | Host-orchestrated stage launches + device ping-pong |
| Scores | Single-stream bit-identical first; parallel reduce only with documented order |
| CI | Default GitHub Actions stays **CPU-only**; CUDA tests run where Toolkit/GPU exists |
| Exit tag | `v0.3.0-cuda-parity` when twin parity suite is green |

## Commit roadmap (granular)

Workstreams below; **layout overrides** any earlier sketch that used
`include/parcae/cuda/` as the primary tree.

### A — Scaffold & device ABI

| Commit | Title |
|--------|-------|
| 1 | docs: CUDA plan freeze + architecture update (**this**) |
| 2 | build: enable CUDA in `Parcae/Parcae` VS project (+ optional CMake hook) |
| 3 | feat: device buffer RAII (alloc/copy/free → `Status`) |
| 4 | feat: POD params mirroring JSON families |
| 5 | feat: interrupt device view (bitmask / sorted skips) |
| 6 | feat: CUDA backend façade (`apply_into` dispatch) |
| 7 | test: smoke identity copy kernel |
| 8 | docs: local CUDA build notes; CI remains CPU-default ([cuda-build.md](cuda-build.md)) |

### B — Transform twins

| Commit | Title |
|--------|-------|
| 9–16 | Kernels: atbash, caesar, identity, affine, vigenere, beaufort, totient, compose driver |
| 17 | Full catalog dispatch |
| 18 | data: `data/parity/` goldens |
| 19–20 | tests: golden `output_sha256` + solved consumable CPU↔CUDA |

### C — Score twins

| Commit | Title |
|--------|-------|
| 21 | docs: reduction associativity ([cuda-score-reduction.md](cuda-score-reduction.md)) |
| 22–26 | exact_match, hamming, ic_mod29, self_repeat, chi², dispatch |
| 27 | test: CUDA score suite (noise + plaintext separation via `CudaScore`) |

### D — Batch SoA

| Commit | Title |
|--------|-------|
| 28 | feat: Candidate Batch ABI v0 buffers (host side) |
| 29 | feat: CUDA batch apply — caesar all 29 shifts |
| 30–33 | atbash/affine/vigenere batches, score + top-k |
| 34 | test: rank-1 recovery on CUDA batch |

### E — Tools & exit

| Commit | Title |
|--------|-------|
| 35–37 | tool/CLI `--backend cuda`, `parcae-parity` |
| 38–39 | cuda-reference + exit checklist docs |
| 40 | property round-trips on CUDA |
| 41–42 | version 0.3.0 + annotated tag `v0.3.0-cuda-parity` |

## Non-goals (still)

- Agents / LLM loops (later)
- Replacing CPU `[solved]` as source of truth
- Fast-math or schedule-dependent score heaps
- Requiring CUDA to build the header-only CPU toolkit
