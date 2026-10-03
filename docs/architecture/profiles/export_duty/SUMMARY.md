# Export duty cycle — Track A ACCEPTANCE (2026-10-03, RTX 5070 Ti)

Post-residency capture (Commits 2–7: pinned arena, `TheoryDeviceScratch`,
streams, ping-pong pipeline, `theory_scores_only` + prior fix). Commit 1
baseline kept under [`pre_residency/`](pre_residency/).

Captured with
[`scripts/cuda/capture_export_duty.ps1`](../../../scripts/cuda/capture_export_duty.ps1)
(`-BuildDir build-rel-cuda -Config Release`).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Toolkit / tools:** CUDA 13.3, Nsight Systems 2026.1.3  
**Metric (primary Kernel SLO):** `BenchTimer` cudaEvent, setup excluded  
**Playbook:** [`cuda-profile-theory.md`](../../cuda-profile-theory.md) § Export duty cycle

**idle GPU ≠ Kernel SLO PCIe**

## Commit 8 — skipped

No `cudaMallocAsync` / alloc pool. Warm `TheoryDeviceScratch::ensure_capacity`
is grow-only and already covers the stress grid; post-capture `cudaMalloc` is
**0.2%** of CUDA API time (15 calls, ~148 µs). Remaining cold cost is
`cudaStreamCreateWithFlags` (~74 ms, once per process) — not per-chunk churn.

## ACCEPTANCE (metric A)

Pass if **either**:

1. **≥2×** reduction in non-kernel NVTX wall vs Commit 1 (`pre_residency/`), or
2. **≥70%** hist NVTX duty (`hist_kernel` / Σ export NVTX ranges)

on the synthetic multi-chunk stress (`T=65536`, `C=64`, 8 chunks). Fair Kernel
SLO must stay within noise of pre-residency (~S1 ~400B class).

| Gate | Pre (Commit 1) | Post (ACCEPTANCE) | Result |
|------|----------------|-------------------|--------|
| Non-kernel NVTX wall | **246.8 ms** (99.5% of ranges) | **4.65 ms** (73.3%) | **~53×** ↓ — **PASS** |
| Hist NVTX duty | **0.5%** | **26.7%** | <70% (OR not needed) |
| H2D NVTX wall | **245.4 ms** (98.9%) | **1.07 ms** (16.9%) | **~229×** ↓ |
| H2D bytes | **0.529 MB** | **0.068 MB** | **~7.8×** ↓ |
| `cudaMalloc` API | **93.7%** / ~70.9 ms | **0.2%** / ~0.15 ms | alloc churn gone |
| Fair S1 Kernel SLO | **395.31B** | **482.40B** | within/above ~400B class |

**Verdict: PASS** (gate 1). Residual export NVTX is mostly `d2h` (31%) + warm
`h2d` param slabs (17%); hist kernels unchanged (~1.7 ms GPU time).

## Fair Kernel SLO vs 896B

Source: `theory_fair.json` — `parcae-bench --suite theory --allow-cuda`,
`T=1048576`, `repeats=8`. Spec peak = physical DRAM roof **896B** runes/s
(@ 1 B cipher/rune). Done = ≥90% ≈ **806.4B**.

| Row | C | T | reps | runes/s | % of 896B | Gate | Notes |
|-----|---|---|------|---------|-----------|------|-------|
| `T.theory.caesar_bytecode` (S0) | 29 | 1048576 | 8 | **61.82B** | **6.90%** | **fail** | Interpreter; ≈ pre 67.83B (noise) |
| `T.theory.compare_caesar` | 29 | 1048576 | 8 | **396.65B** | **44.27%** | **fail** | Catalog twin |
| `T.theory.s1_lut29` (S1) | 29 | 1048576 | 8 | **482.40B** | **53.84%** | **fail** | LUT-29; ≥ pre ~395B |
| `T.theory.progressive` (S2) | 841 | 1048576 | 8 | **244.43B** | **27.28%** | **fail** | Linear keyed; ≈ pre 257.86B |

Fair Kernel SLO still excludes H2D/setup. Residency did **not** regress the
~400B S1 class (improved on this plate).

## Export stress config

Catch filter `[search][export][theory][duty][cuda]`
(`theory_export_duty_stress_test.cpp`):

| Param | Value |
|-------|-------|
| T (tokens) | **65536** |
| C (candidates) | **64** |
| chunks | **8** |
| Cache | warm `TheoryExportCache` (host compile once; device `ops`/`imm` once) |
| Scratch / streams | persistent `TheoryDeviceScratch` + `CudaStreamPair` |
| Path | `TheoryExportPipeline` (stage → collect → launch ping-pong) |

## nsys wall split / duty (post)

Report: `export_duty_stress_timeline.nsys-rep` → `export_duty_stress_stats.txt`.

### Top kernels (`cuda_gpu_kern_sum`)

| Kernel | Time share | Instances | Total time |
|--------|------------|-----------|------------|
| `theory_chi2_hist_kernel` | **94.3%** | 8 | 1.70 ms |
| `chi2_finalize_kernel` | **5.5%** | 8 | 99 µs |
| `theory_chi2_patch_inf_kernel` | **0.2%** | 8 | 4.5 µs |

### CUDA API (`cuda_api_sum`)

| API | Time share | Notes |
|-----|------------|-------|
| `cudaStreamCreateWithFlags` | **92.5%** | Cold once (~74 ms); not per-chunk |
| `cudaMemcpyAsync` | **2.7%** | ~2.2 ms host API |
| `cudaMalloc` | **0.2%** | ~148 µs — **no** warm churn |
| `cudaLaunchKernel` | **1.1%** | — |

Device memcpy (`cuda_gpu_mem_time_sum`): H2D ~11 µs, D2H ~7.5 µs total.

### NVTX ranges (`nvtx_sum`)

| Range | Time share | Instances | Total | vs pre |
|-------|------------|-----------|-------|--------|
| `d2h` | **31.4%** | 8 | 1.99 ms | was 0.1% / 0.31 ms |
| `hist_kernel` | **26.7%** | 8 | 1.70 ms | was 0.5% / 1.28 ms |
| `h2d` | **16.9%** | 8 | 1.07 ms | was 98.9% / 245 ms |
| `finalize` | **11.7%** | 8 | 0.75 ms | — |
| `prepare_theory` | **9.6%** | 8 | 0.61 ms | — |
| `bind_slots` | **3.7%** | 8 | 0.23 ms | — |

## Artifacts

| File | Role |
|------|------|
| `theory_fair.json` | Fair Kernel SLO JSON (post) |
| `export_duty_stress_timeline.nsys-rep` | Multi-chunk stress nsys (post) |
| `export_duty_stress_stats.txt` | kern / api / nvtx / mem sums (post) |
| `export_duty_cache_timeline.nsys-rep` | Short cache NVTX smoke |
| `capture_log.txt` | Script transcript |
| [`pre_residency/`](pre_residency/) | Commit 1 baseline (pre Track A residency) |
