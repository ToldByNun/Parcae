# Export duty cycle (2026-10-03, RTX 5070 Ti)

Prove that **GPU idle on the export path** is host / PCIe / sync gaps, not a
Kernel SLO that is “PCIe-bound.” Fair `cudaEvent` Kernel SLO excludes H2D;
nsys on multi-chunk `GpuCandidateExport` shows where wall time goes.

Captured with
[`scripts/cuda/capture_export_duty.ps1`](../../../scripts/cuda/capture_export_duty.ps1).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Toolkit / tools:** CUDA 13.3, Nsight Systems 2026.1.3  
**Metric (primary Kernel SLO):** `BenchTimer` cudaEvent, setup excluded  
**Playbook:** [`cuda-profile-theory.md`](../../cuda-profile-theory.md) § Export duty cycle

**idle GPU ≠ Kernel SLO PCIe**

## Fair Kernel SLO (diary %-of-896B)

Source: `theory_fair.json` — `parcae-bench --suite theory --allow-cuda`,
`T=1048576`, `repeats=8`. Then-Spec **896B**; Remap Done for Caesar/S1/S2 is
now ~**2.0 TB** ([`../../../hist-alphabet-remap.md`](../../../hist-alphabet-remap.md)).

| Row | C | T | reps | runes/s | % of 896B (diary) | Gate (then) | Notes |
|-----|---|---|------|---------|-------------------|-------------|-------|
| `T.theory.caesar_bytecode` (S0) | 29 | 1048576 | 8 | **67.83B** | **7.57%** | **fail** | Interpreter; `checkpoint_50B=hit` |
| `T.theory.compare_caesar` | 29 | 1048576 | 8 | **390.32B** | **43.56%** | **fail** | Catalog twin; Remap climb |
| `T.theory.s1_lut29` (S1) | 29 | 1048576 | 8 | **395.31B** | **44.12%** | **fail** | LUT-29; ≈ twin; Remap climb |
| `T.theory.progressive` (S2) | 841 | 1048576 | 8 | **257.86B** | **28.78%** | **fail** | Linear keyed; fair C=841 |

Fair Kernel SLO is measured with H2D / host prepare **outside** the timer. S1 ~
44% of then-896B is still compute/hist-bound, not PCIe-limited on the fair grid.

## Export stress config

Catch filter `[search][export][theory][duty][cuda]`
(`theory_export_duty_stress_test.cpp`):

| Param | Value |
|-------|-------|
| T (tokens) | **65536** |
| C (candidates) | **64** |
| chunks | **8** |
| Cache | warm `TheoryExportCache` (host compile once; device `ops`/`imm` once) |
| Path | `GpuCandidateExport::theory_scores_only` (quadratic_polynomial_stream) |

## nsys wall split / duty

Report: `export_duty_stress_timeline.nsys-rep` → `export_duty_stress_stats.txt`.

### Top kernels (`cuda_gpu_kern_sum`)

| Kernel | Time share | Instances | Total time |
|--------|------------|-----------|------------|
| `theory_chi2_hist_kernel` | **94.2%** | 8 | 1.68 ms |
| `chi2_finalize_kernel` | **5.5%** | 8 | 98 µs |
| `theory_chi2_patch_inf_kernel` | **0.3%** | 8 | 4.5 µs |

GPU kernel work across 8 chunks is ~**1.8 ms** total — a thin slice of wall.

### CUDA API (`cuda_api_sum`)

| API | Time share | Notes |
|-----|------------|-------|
| `cudaMalloc` | **93.7%** | Cold process alloc dominate (~70.9 ms; one large call) |
| `cudaDeviceSynchronize` | **2.0%** | ~1.54 ms |
| `cudaMemcpy` | **1.5%** | ~1.11 ms host API time |
| `cudaLaunchKernel` | **1.2%** | — |

Device memcpy time (`cuda_gpu_mem_time_sum`): H2D ~23 µs, D2H ~16 µs total —
PCIe payload for this stress is tiny vs host waits.

### NVTX ranges (`nvtx_sum`)

| Range | Time share | Instances | Notes |
|-------|------------|-----------|-------|
| `h2d` | **98.9%** | 8 | Host-range wall; med ~98 µs, one cold outlier ~245 ms |
| `hist_kernel` | **0.5%** | 8 | ~1.28 ms total |
| `prepare_theory` | **0.2%** | 8 | — |
| `finalize` | **0.2%** | 8 | — |
| `d2h` | **0.1%** | 8 | — |
| `bind_slots` | **~0%** | 8 | ~120 µs total |

`materialize` / `ingest` absent (scores-only path).

### Duty interpretation

- **Fair Kernel SLO** (table above) already excludes PCIe/setup: S0 ~8%, S1 ~44%
  of 896B — gate failures are kernel/hist arithmetic, not bus bandwidth.
- **Export wall** is dominated by host-side `h2d` NVTX (incl. sync/wait) and
  cold `cudaMalloc`, while hist kernels are under 2 ms for the whole stress.
- Therefore: **idle GPU on export = host / PCIe / sync gaps**. That does **not**
  mean Kernel SLO is PCIe-bound.

**idle GPU ≠ Kernel SLO PCIe**

## Artifacts

| File | Role |
|------|------|
| `theory_fair.json` | Fair Kernel SLO JSON |
| `export_duty_stress_timeline.nsys-rep` | Multi-chunk stress nsys |
| `export_duty_stress_stats.txt` | kern / api / nvtx / mem sums |
| `export_duty_cache_timeline.nsys-rep` | Short cache NVTX smoke |
| `capture_log.txt` | Script transcript |
