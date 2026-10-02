# S0 climb baseline (2026-10-02, RTX 5070 Ti)

**TheoryChi2Batch** (`theory_chi2_hist_kernel`) vs **Caesar twin**
(`caesar_chi2_histogram_decrypt_kernel`) — anchor for S0 Kernel SLO.

Captured with
[`scripts/cuda/capture_s0_climb_baseline.ps1`](../../../scripts/cuda/capture_s0_climb_baseline.ps1).

**Prior reference:** [`../baseline/SUMMARY.md`](../baseline/SUMMARY.md) (2026-10-01).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Toolkit / tools:** CUDA 13.3, Nsight Systems 2026.1.3, Nsight Compute 2026.2.1  
**Metric (primary):** `BenchTimer` cudaEvent median-of-3, setup excluded  

## Spec peak (physical DRAM roofline)

`estimated_peak` = **physical DRAM roofline**, not a measured quiet max:

```text
896e9 B/s GDDR7 / 1 B cipher/rune = 896B runes/s
Done (≥90%) ≈ 806.4B
```

Canonical: [`BenchTierSpec`](../../../../include/parcae/bench/bench_tier_spec.hpp) /
[`cuda-throughput.md`](../../cuda-throughput.md). Historical quiet S0 medians
(~69–75B) are **progress**, not the Spec ceiling.

| Spec id | Peak | 90% Done gate | Notes |
|---------|------|---------------|-------|
| `T.theory.caesar_bytecode` | **896B** | **806.4B** | Same roof as catalog fused hist |

## Climb progress

Measured cudaEvent values unchanged; **% of 896B** reinterpreted.

| Tag | Fair S0 cudaEvent | % of 896B | Notes |
|-----|-------------------|-----------|-------|
| s0-climb baseline | **59.44B** | **~6.6%** | pre-pass-1 |
| **s0-climb-p1** | **~69–75B** | **~7.7–8.4%** | tiles/trusted/residency — interpreter plateau |
| **s0-climb-p2** | no net win | — | uchar4 / 32-tok / launch_bounds regressed |
| **dram-roof** | Spec → **896B** | — | peak = physics; S0 still **not Done** |

Caesar twin on the same capture: **406.62B** ≈ **45.4%** of 896B (reference only).

## Artifacts (local / gitignored binaries)

| File | Role |
|------|------|
| `theory_fair.json` | Theory suite cudaEvent JSON |
| `s0_hist.ncu-rep` / `caesar_hist.ncu-rep` | ncu (gitignored) |
| `s0_climb_timeline.nsys-rep` | nsys (gitignored) |
| `nsys_kern_sum.txt` | Kernel summary export |
