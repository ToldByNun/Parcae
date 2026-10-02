# Theory CUDA post–specialized emit (2026-10-02, RTX 5070 Ti)

Captured with [`scripts/cuda/capture_theory_specialized.ps1`](../../../scripts/cuda/capture_theory_specialized.ps1)
after S1 LUT-29 / S2 linear emit + `BenchTierSpec` theory rows (commit 13).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Toolkit / tools:** CUDA 13.x, Nsight Systems / Compute (same plate as baseline)  
**Metric (primary):** `BenchTimer` cudaEvent median-of-3, setup excluded  
**Compare-to:** [`profiles/baseline/SUMMARY.md`](../baseline/SUMMARY.md) (2026-10-01)

## cudaEvent Kernel SLO (fair T)

Source: `theory_fair.json` — `T=1048576`, reps=8, `--campaign-grid`.

| Row | C | T | reps | runes/s | % peak (Spec) | Gate | Notes |
|-----|---|---|------|---------|---------------|------|-------|
| `T.theory.caesar_bytecode` (S0) | 29 | 1048576 | 8 | **60.53B** | 80.7% of 75B | **fail** | Interpreter; ≈baseline 65.87B (GPU noise) |
| `T.theory.compare_caesar` | 29 | 1048576 | 8 | **383.21B** | 97.8% of 392B | **pass** | Catalog twin; `checkpoint_50B=hit` |
| `T.theory.s1_lut29` (S1) | 29 | 1048576 | 8 | **395.58B** | 83.6% of 473B | **fail** | Specialized LUT; `checkpoint_50B=hit` |
| `T.theory.progressive` (S2) | 9 | 1048576 | 8 | **187.93B** | 53.7% of 350B | **fail** | Linear uchar4; `checkpoint_50B=hit` |
| `T.theory.caesar_campaign` | 16384 | 262 | 8 | **27.67B** | — | pass | `underfill_not_slo_gate` |

**vs baseline (same fair grid):**

| Shape | Baseline (2026-10-01) | Specialized (2026-10-02) | Delta |
|-------|----------------------|--------------------------|-------|
| S0 Caesar bytecode | 65.87B | 60.53B | ~noise / quiet-GPU variance |
| Caesar twin | 304.97B | 383.21B | twin variance; not a theory change |
| Progressive / S2 | **29.71B** (bytecode) | **187.93B** (S2 linear) | **~6.3×** kernel SLO |
| S1 LUT (new row) | — | **395.58B** | ≈ Caesar twin; ~**6.5×** vs baseline S0 |

Provisional Spec peaks (S1=473B Affine-class, S2=350B plan) are **not yet remesaured** —
90% gate fails on S1/S2 until commit-16 remesaure recalibrates `estimated_peak`.
S1 already clears the optional **50B** interim checkpoint.

## nsys timeline (`theory_specialized_timeline.nsys-rep`)

Workload: `parcae-bench --suite theory --tokens 262144 --repeats 2 --campaign-grid`.

`cuda_gpu_kern_sum` (avg):

| Kernel | Instances | Avg (ns) |
|--------|-----------|----------|
| `theory_chi2_hist_kernel` | 20 | **130551** |
| `theory_hist_chi2_s1_lut_kernel` | 10 | **15168** |
| `caesar_chi2_histogram_decrypt_kernel` | 10 | **14605** |
| `theory_hist_chi2_s2_linear_kernel` | 10 | **8435** |

S1 avg duration ≈ Caesar twin; S0 still ~**8.6×** S1 at this short-T grid.

## ncu (T=262144, launch skip=4, count=1)

Reports: `s0_hist.ncu-rep`, `s1_lut.ncu-rep`, `s2_linear.ncu-rep`, `caesar_hist.ncu-rep`.

| Kernel | Duration | SM % peak | DRAM % peak | Implied runes/s @ C×T |
|--------|----------|-----------|-------------|------------------------|
| `theory_chi2_hist_kernel` | **160.86 µs** | 56.5 | 2.59 | ~47.3B (`29×262144`) |
| `theory_hist_chi2_s1_lut_kernel` | **21.54 µs** | 60.1 | 2.83 | ~353B |
| `theory_hist_chi2_s2_linear_kernel` | **11.97 µs** | 62.8 | 4.93 | ~197B (`9×262144`) |
| `caesar_chi2_histogram_decrypt_kernel` | **20.51 µs** | 63.1 | 2.79 | ~370B |

**Ratio (ncu duration @ same C=29, T=262k):** S0 / S1 ≈ **7.5×**; S1 / Caesar ≈ **1.05×**
(specialized LUT ≈ catalog Caesar hist).

Baseline S0 ncu was **155.7 µs** @ same T — specialized capture S0 **160.9 µs** (noise).

## Acceptance

| Shape | Spec peak | 90% gate | Measured fair | Status |
|-------|-----------|----------|---------------|--------|
| S0 | 75B | 67.5B | 60.53B | **below** (interpreter ceiling ~69B interim) |
| S1 | 473B provisional | 425.7B | 395.58B | **below** Spec; remesaure peak downward or tune further |
| S2 | 350B provisional | 315B | 187.93B | **below** Spec; remesaure / C-grid caveat (C=9 vs Spec 841) |

Done gate remains **≥90% estimated_peak** after peaks are calibrated on this plate
(commit 16 remesaure). This snapshot is the required post-emit compare to baseline.

## Artifacts (local / gitignored binaries)

| File | Role |
|------|------|
| `theory_fair.json` | Theory suite cudaEvent JSON (fair + campaign) |
| `theory_fair.txt` | Human stdout (separate quiet run) |
| `s0_hist.ncu-rep` | ncu S0 bytecode hist |
| `s1_lut.ncu-rep` | ncu S1 LUT hist |
| `s2_linear.ncu-rep` | ncu S2 linear hist |
| `caesar_hist.ncu-rep` | ncu Caesar twin hist |
| `theory_specialized_timeline.nsys-rep` | nsys report |
