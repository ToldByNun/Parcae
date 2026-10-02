# S0 climb baseline (2026-10-02, RTX 5070 Ti)

**TheoryChi2Batch** (`theory_chi2_hist_kernel`) vs **Caesar twin**
(`caesar_chi2_histogram_decrypt_kernel`) — anchor for S0 Kernel SLO.

Captured with
[`scripts/cuda/capture_s0_climb_baseline.ps1`](../../../scripts/cuda/capture_s0_climb_baseline.ps1).

**Prior reference:** [`../baseline/SUMMARY.md`](../baseline/SUMMARY.md) (2026-10-01).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Toolkit / tools:** CUDA 13.3, Nsight Systems 2026.1.3, Nsight Compute 2026.2.1  
**Metric (primary):** `BenchTimer` cudaEvent median-of-3, setup excluded  

## Spec peak (practical ceiling)

`estimated_peak` is the **best fair quiet-GPU** cudaEvent for this shape — the
limit we aim at. `%peak` **must stay ≤100**; if a quiet run goes above, **raise**
the Spec peak. Do not lower it to make noisy runs look like 90%.

| Spec id | Peak | 90% Done gate | Notes |
|---------|------|---------------|-------|
| `T.theory.caesar_bytecode` | **75B** | **67.5B** | Quiet climb max ~75B |

## Climb progress

| Tag | Fair S0 cudaEvent | Notes |
|-----|-------------------|-------|
| s0-climb baseline | **59.44B** | pre-pass-1 |
| **s0-climb-p1** | **~69–75B** | tiles/trusted/residency — hits ceiling band |
| **s0-climb-p2** | no net win | uchar4 / 32-tok / launch_bounds regressed |
| **s0-peak** | keep **75B** | >100% would mean stale/low Spec — raise, don’t cut for noise |

## Artifacts (local / gitignored binaries)

| File | Role |
|------|------|
| `theory_fair.json` | Theory suite cudaEvent JSON |
| `s0_hist.ncu-rep` / `caesar_hist.ncu-rep` | ncu (gitignored) |
| `s0_climb_timeline.nsys-rep` | nsys (gitignored) |
| `nsys_kern_sum.txt` | Kernel summary export |
