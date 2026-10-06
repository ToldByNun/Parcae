# Cache-bound plate — L2 / DRAM evidence for fused hist

**Status:** harness + quiet plate filled (2026-10-06)  
**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120), GDDR7 **896 GB/s**  
**Date:** 2026-10-06  
**Capture:** [`scripts/cuda/capture_cache_bound.ps1`](../../../../scripts/cuda/capture_cache_bound.ps1)  
**Wrapper:** [`scripts/cuda/profile_theory_hist.ps1`](../../../../scripts/cuda/profile_theory_hist.ps1) `-MetricsPreset cache_bound -ExportCsv`  
**Playbook:** [`cuda-profile-theory.md`](../../cuda-profile-theory.md)  
**Contract peaks:** [`BenchTierSpec`](../../../../include/parcae/bench/bench_tier_spec.hpp)

## Why this plate exists

Prior captures recorded SM% / DRAM% of peak and duration, but **not** absolute
`dram__bytes.sum` or L1/L2 sector / hit-rate on every twin. Playbook already
asked for L1/L2 sectors; scripts omitted them. Operator observation of
**~16.9 GB/s** absolute DRAM throughput (~**1.9%** of 896 GB/s) matches this
plate’s Affine row (**17.08 GB/s**, DRAM SoL **1.82%**) — compute / shared-hist
atomics with cipher largely L2-served, not GDDR7-bound.

## How to run

```powershell
cmake -S . -B build-cuda -DPARCAE_BUILD_CUDA=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build-cuda --config Release --target parcae-bench

.\scripts\cuda\capture_cache_bound.ps1 -BuildDir build-rel-cuda
# cudaEvent only:
.\scripts\cuda\capture_cache_bound.ps1 -BuildDir build-rel-cuda -SkipNcu
# re-derive tables from CSVs:
python .\scripts\cuda\derive_cache_bound_metrics.py
```

**ERR_NVGPUCTRPERM:** enable GPU performance counters or elevate; cudaEvent JSON
still valid. **Metric fallback:** if an L1/L2 name is rejected on sm_120, capture
retries core metrics (`dram__bytes.sum` + SM/DRAM % + duration) and sets
`ncu_fallback=true` in `capture_digest.json`. This plate used the **full**
`cache_bound` preset (no fallback).

## Kernels (ncu tags)

| Tag | Kernel filter | Launcher |
|-----|---------------|----------|
| `caesar_hist` | `caesar_chi2_histogram_decrypt_kernel` | `parcae-bench --suite theory` |
| `atbash_hist` | `atbash_chi2_hist_kernel` | `parcae-bench --suite slo --extended` |
| `s1_lut` | `theory_hist_chi2_s1_lut_kernel` | theory `--no-compare-catalog` |
| `s2_linear` | `theory_hist_chi2_s2_linear_kernel` | theory `--no-compare-catalog` |
| `affine_hist` | `affine_chi2_hist_kernel` | slo `--extended` |

Fair cudaEvent: `T=1048576` → `theory_fair.json` + `catalog_slo_extended.json`.  
ncu: `T=262144`, `--launch-skip 4 --launch-count 1`, `grid.y=64` (fat-tile).

## MetricsPreset `cache_bound`

| Metric | Use |
|--------|-----|
| `gpu__time_duration.sum` | Duration → runes/s = `C×T / duration_s` |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute SoL |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM SoL % |
| `dram__bytes.sum` | Absolute DRAM traffic → **GB/s** and bytes/rune |
| `lts__t_sector_hit_rate.pct` | L2 hit rate |
| `lts__t_sectors_srcunit_tex_lookup_{hit,miss}.sum` | L2 sector evidence |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld*.sum` | L1TEX global-load sectors |
| stall / occupancy (baseline subset) | Often **n/a on sm_120** |

### Derived fields

```text
dram_gbs     = dram__bytes.sum / duration_s / 1e9
bytes_rune   = dram__bytes.sum / (C × T)
pct_of_896   = 100 * dram_gbs / 896
runes_per_s  = C × T / duration_s
```

## Quiet plate results (2026-10-06)

ncu derived from `*_metrics.csv` → `ncu_derived.json` (assume `T=262144` from
capture args; `C` / `grid.y` from launch metrics).

| Kernel | `(C, tiles)` | Duration | SM % | DRAM % | `dram__bytes` | DRAM GB/s | L2 hit % | bytes/rune | ncu runes/s |
|--------|--------------|----------|------|--------|---------------|-----------|----------|------------|-------------|
| Caesar twin | (29, 64) | 10.91 µs | 41.7 | 3.13 | 312.2 KB | **28.61** | 48.0 | 0.0411 | 696.7B |
| Atbash | (512, 64) | 449.09 µs | 57.1 | 0.51 | 2.17 MB | **4.82** | **98.5** | 0.0161 | 298.9B |
| S1 LUT | (29, 64) | 12.16 µs | 59.8 | 3.00 | 333.2 KB | **27.40** | 92.2 | 0.0438 | 625.2B |
| S2 linear | (841, 64) | 548.32 µs | 73.5 | 0.82 | 4.23 MB | **7.72** | 22.6 | 0.0192 | 402.1B |
| Affine | (812, 64) | 237.47 µs | 88.8 | 1.82 | 4.06 MB | **17.08** | **97.6** | 0.0191 | 896.4B |

Notes:

- Absolute DRAM is **tens of GB/s** (Affine **17.08 ≈ operator 16.9**), never near 896.
- **bytes/rune ≪ 1** for every row: shared cipher buffer across `C` candidates → L2
  absorbs most loads. Unique-key Spec still uses **1 B/rune** as the *logical*
  roof for Done; physical traffic is the occupancy/L2 story.
- Atbash bytes/rune **0.016** ≈ traffic_model **0.01111** (same class; small plate
  noise / counter scope).
- Affine L2 hit **97.6%** + DRAM **17 GB/s** while fair cudaEvent can print high
  absolute RPS — traffic model fixed in Spec (`kHistBytesPerRuneAffineSharedCipher
  = 0.01906` → ~47.0 TB); see [`../traffic_model/SUMMARY.md`](../traffic_model/SUMMARY.md).

Fair cudaEvent (`T=1048576`, H2D excluded):

| Spec / row | runes/s | % of Spec peak | Source |
|------------|---------|----------------|--------|
| `T.theory.compare_caesar` | **749.3B** | **83.6%** | `theory_fair.json` |
| `T.theory.caesar_bytecode` (ShapeInline) | **655.7B** | **73.2%** | `theory_fair.json` |
| `T.theory.s1_lut29` | **418.0B** | **46.6%** | `theory_fair.json` |
| `T.theory.progressive` (S2) | **352.9B** | **39.4%** | `theory_fair.json` |
| `F.atbash` | **1344.4B** | **1.67%** of 80.7 TB | `catalog_slo_extended.json` |
| `F.affine` | **885.2B** | **98.8%** | `catalog_slo_extended.json` |
| `T1` Caesar | **555.2B** | **62.0%** | `catalog_slo_extended.json` |

(Fair plate was a single quiet pass during harness bring-up; Kernel SLO Done still
needs multi-run medians on a dedicated quiet plate.)

## Diagnosis checklist

- [x] Absolute DRAM GB/s ≪ 896 for all five kernels
- [x] Atbash bytes/rune ≪ 1 (L2 / shared-cipher occupancy; L2 hit 98.5%)
- [x] Caesar / S1 / S2 / Affine physical bytes/rune also ≪ 1 at this grid (shared cipher)
- [x] High L2 hit when DRAM SoL low (Atbash / Affine / S1); S2 lower hit (22.6%) with still tiny DRAM SoL
- [x] Stall metrics: not relied on (often n/a on sm_120); SM/DRAM/L2/bytes used instead

## Artifacts

| File | Role | Git |
|------|------|-----|
| `theory_fair.json` | Fair theory cudaEvent | allow-listed |
| `catalog_slo_extended.json` | Catalog slo extended | allow-listed |
| `capture_digest.json` | Which ncu/CSV landed | allow-listed |
| `ncu_derived.json` / `fair_derived.json` | Parsed tables | allow-listed |
| `*_metrics.csv` | ncu raw CSV export | allow-listed |
| `*.ncu-rep` | Binary ncu report | gitignored |
| `capture_log.txt` | Transcript | gitignored |
| `SUMMARY.md` | This write-up | allow-listed |

## Follow-up (not this commit)

Roof climb / Spec fixes use this plate as before/after evidence. Do **not** mark
Kernel SLO Done from campaign wall or from DRAM GB/s alone — Done remains
≥90% of shape `estimated_peak` at fair `T≥2^20` (see contracts).
