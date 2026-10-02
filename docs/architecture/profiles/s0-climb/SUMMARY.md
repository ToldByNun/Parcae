# S0 climb baseline (2026-10-02, RTX 5070 Ti)

**TheoryChi2Batch** (`theory_chi2_hist_kernel`) vs **Caesar twin**
(`caesar_chi2_histogram_decrypt_kernel`) — anchor for S0 Kernel SLO climb toward
≥90% of **75B** (`T.theory.caesar_bytecode`).

Captured with
[`scripts/cuda/capture_s0_climb_baseline.ps1`](../../../scripts/cuda/capture_s0_climb_baseline.ps1).

**Prior reference:** [`../baseline/SUMMARY.md`](../baseline/SUMMARY.md) (2026-10-01).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Toolkit / tools:** CUDA 13.3, Nsight Systems 2026.1.3, Nsight Compute 2026.2.1  
**Metric (primary):** `BenchTimer` cudaEvent median-of-3, setup excluded  

## cudaEvent Kernel SLO

Source: `theory_fair.json` (this directory).

| Row | C | T | reps | runes/s | % peak | Gate |
|-----|---|---|------|---------|--------|------|
| `T.theory.caesar_bytecode` (S0) | 29 | 1048576 | 8 | **59.44B** | **79.3%** of 75B | **fail** (need ≥67.5B) |
| `T.theory.compare_caesar` (twin) | 29 | 1048576 | 8 | **406.62B** | 103.7% of 392B | pass (reference) |

**Gap (fair T):** twin ≈ **6.8×** S0 cudaEvent (406.6 / 59.4). Climb must close this
without breaking Caesar-as-bytecode semantics.

S1/S2 on the same capture (context only — not this milestone): S1 **393.9B** (94%),
S2 progressive **187.2B** (94%).

## ncu (profiled launch)

Workload: `parcae-bench --suite theory --allow-cuda --tokens 262144 --repeats 2`.
S0: `--no-compare-catalog`; Caesar: default theory suite (compare row). Skip 4,
count 1. Reports: `s0_hist.ncu-rep`, `caesar_hist.ncu-rep`.

| Kernel | Duration | SM % peak | DRAM % peak | Warps active % | Implied runes/s |
|--------|----------|-----------|-------------|----------------|-----------------|
| `theory_chi2_hist_kernel` | **157.38 µs** | 58.1 | 0.94 | 89.0 | ~48.3B |
| `caesar_chi2_histogram_decrypt_kernel` | **19.55 µs** | 65.9 | 2.56 | 80.4 | ~389B |

**Ratio:** S0 duration ≈ **8.0×** Caesar at same `C×T` (grid tiling differs:
theory `(29,1024)` vs Caesar `(29,256)`).

Warp stall metrics (`smsp__warp_issue_stalled_*`) still `n/a` on sm_120 with this
metric set — duration + SM/DRAM throughput are the usable counters (same as baseline).

## nsys (`s0_climb_timeline.nsys-rep`)

Short theory suite + `--campaign-grid`. Digest: `nsys_kern_sum.txt`.

`cuda_gpu_kern_sum` (avg):

| Kernel | Instances | Avg (ns) | Share |
|--------|-----------|----------|-------|
| `theory_chi2_hist_kernel` | 20 | **135645** | 71% |
| `chi2_finalize_kernel` | 50 | 13778 | 18% |
| `caesar_chi2_histogram_decrypt_kernel` | 10 | **14451** | 3% |
| `theory_hist_chi2_s1_lut_kernel` | 10 | 14973 | 3% |

NVTX (same capture): `:hist_kernel` ~51% of NVTX time; `:compare_caesar` ~13%.
Cold-process `cudaMalloc` still dominates CUDA API wall in this micro-run.

## Climb checklist (next code changes)

1. Re-run this script after each S0 kernel change; append
   [`cuda-profile-theory.md`](../../cuda-profile-theory.md) progress row.
2. Target fair cudaEvent **≥67.5B** (90% of 75B); twin row stays sanity reference.
3. Do not treat short-`T` ncu/nsys underfill rows as the SLO gate.

## Artifacts (local / gitignored binaries)

| File | Role |
|------|------|
| `theory_fair.json` | Theory suite cudaEvent JSON |
| `s0_hist.ncu-rep` | ncu TheoryChi2Batch |
| `caesar_hist.ncu-rep` | ncu Caesar twin |
| `s0_climb_timeline.nsys-rep` | nsys timeline |
| `nsys_kern_sum.txt` | Kernel summary export |
| `capture_log.txt` | Capture transcript (may be partial if nsys stats failed mid-script) |
