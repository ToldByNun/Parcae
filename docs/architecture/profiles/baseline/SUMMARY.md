# Theory CUDA baseline (2026-10-01, RTX 5070 Ti)

Captured with [`scripts/cuda/capture_theory_baseline.ps1`](../../../scripts/cuda/capture_theory_baseline.ps1)
and [`scripts/cuda/profile_theory_hist.ps1`](../../../scripts/cuda/profile_theory_hist.ps1).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Toolkit / tools:** CUDA 13.3, Nsight Systems 2026.1.3, Nsight Compute 2026.2.1  
**Metric (primary):** `BenchTimer` cudaEvent median-of-3, setup excluded  

**Spec peak model (normative, reinterpreted):** physical DRAM roofline **896B**
runes/s (@ 1 B cipher/rune). Done = ≥90% ≈ **806.4B**. Historical catalog %
figures below that cited 392B/473B/550B peaks are **obsolete** — same measured
runes/s, new denominator.

## cudaEvent Kernel SLO

Source: `theory_fair.json`, `catalog_slo_extended.json` (this directory).

| Row | C | T | reps | runes/s | % of 896B | Notes |
|-----|---|---|------|---------|-----------|-------|
| `T.theory.caesar_bytecode` | 29 | 1048576 | 8 | **65.87B** | **7.4%** | TheoryChi2Batch interpreter |
| `T.theory.compare_caesar` | 29 | 1048576 | 8 | **304.97B** | **34.0%** | CaesarChi2Batch twin, same cipher |
| `T.theory.progressive` | 9 | 1048576 | 8 | **29.71B** | **3.3%** | keyed stream bytecode |
| `T.theory.caesar_campaign` | 16384 | 262 | 8 | **24.14B** | — | underfill / not SLO gate |
| Catalog `T1` Caesar | 29 | 1048576 | 64 | **347.08B** | **38.7%** | was “88.5% of 392B” under old Spec |
| Catalog `F.affine` | 812 | (tier) | | **422.00B** | **47.1%** | was “89.2% of 473B” |
| Catalog `F.atbash` | | | | **478.44B** | **53.4%** | was “87.0% of 550B”; closest specialized |

**Gap:** fair Theory Caesar bytecode ≈ **4.6×** slower than Caesar twin at same `(C,T)`
(65.87B / 304.97B). Twin is still far below the DRAM roof on this quiet-GPU run.

## nsys timeline (`theory_export_timeline.nsys-rep`)

Workload: `parcae-bench --suite theory --tokens 262144 --repeats 2 --campaign-grid`
(`-t cuda,nvtx`; Windows nsys has no `osrt` trace).

CUDA GPU kernel summary (`cuda_gpu_kern_sum`):

| Kernel | Instances | Avg (ns) | Share |
|--------|-----------|----------|-------|
| `theory_chi2_hist_kernel` | 30 | **122581** | 83% |
| `chi2_finalize_kernel` | 40 | 14320 | 13% |
| `caesar_chi2_histogram_decrypt_kernel` | 10 | **14470** | 3% |
| `theory_chi2_patch_inf_kernel` | 30 | 569 | ~0% |

Host API (selected): `cudaMalloc` dominates wall in cold process (~90% of traced
API time); once warm, kernel + `cudaEventSynchronize` matter. H2D memcpy total
~0.82 MB across 18 copies in this short run.

NVTX: `nvtx_sum` skipped on this capture (no NVTX payload in report — microbench
path; export-path ranges live under `GpuCandidateExport` / scheduler). Re-run
with `search_cycle` + shared `TheoryExportCache` after host-amortize to confirm
`ops`/`imm` upload once per URI (slots still per chunk).

## ncu (re-captured 2026-10-01 after GPU counter permission)

Counters enabled for all users in NVIDIA Control Panel. Reports:
`theory_hist.ncu-rep`, `caesar_hist.ncu-rep`, `affine_hist.ncu-rep`.

Workload: theory/caesar at `C=29 T=262144`; affine at catalog F.affine
`C=812 T=262144`. Profiled launch after 4 skips, count=1.

| Kernel | Duration | SM % peak | DRAM % peak | Warps active % | Implied runes/s |
|--------|----------|-----------|-------------|----------------|-----------------|
| `theory_chi2_hist_kernel` | **155.65 µs** | 74.2 | 1.02 | 90.4 | ~48.8B (`29×262144/t`) |
| `caesar_chi2_histogram_decrypt_kernel` | **19.14 µs** | 67.6 | 2.42 | 80.0 | ~397B |
| `affine_chi2_hist_kernel` | **478.11 µs** | 73.7 | 0.15 | 80.4 | ~445B (`812×262144/t`) |

**Ratio:** theory hist duration ≈ **8.1×** Caesar hist at same `C×T` (grid tiles
differ: theory `(29,1024)` vs Caesar `(29,256)` — scalar bytecode tiling).

Warp stall metrics (`smsp__warp_issue_stalled_*`) returned `n/a` on this
Nsight Compute / sm_120 combo with the requested names; SoL + duration are the
usable baseline counters until a Blackwell-valid stall set is known.

Affine ncu process exit code was 1 because `parcae-bench --suite slo --extended`
failed `C.koan1_fused` on that run — the `.ncu-rep` for affine still wrote OK.

## Acceptance reminder

Done gate = **≥90% of physical DRAM roofline (896B)**. Current bytecode ~66B
(~7.4%) and catalog Atbash ~478B (~53%) are interim datapoints — compute-bound
(DRAM SoL ~0–3%), **not Done**.

## Artifacts (local / gitignored binaries)

| File | Role |
|------|------|
| `theory_fair.json` | Theory suite cudaEvent JSON |
| `catalog_slo_extended.json` | Catalog SLO extended JSON |
| `theory_export_timeline.nsys-rep` | nsys report |
| `theory_hist.ncu-rep` | ncu TheoryChi2Batch hist |
| `caesar_hist.ncu-rep` | ncu Caesar twin hist |
| `affine_hist.ncu-rep` | ncu Affine hist |
