# Traffic model pass — shared-cipher occupancy (2026-10-06)

ncu-backed fix for Atbash / totient `%peak>100` under the 1 B/rune Spec.

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120), GDDR7 **896 GB/s**  
**Tool:** Nsight Compute 2026.2.1 + `parcae-bench --suite slo --extended`  
**Contract:** peak = `DRAM_BW / bytes_per_rune` — never quiet max  
([`cuda-throughput.md`](../../cuda-throughput.md), [`BenchTierSpec`](../../../../include/parcae/bench/bench_tier_spec.hpp))

## Problem

Quiet Kernel SLO printed **`%peak>100`** for:

| Row | Quiet RPS (prior plates) | Old Spec peak | Symptom |
|-----|--------------------------|---------------|---------|
| `F.atbash` (C=512 pad) | ~1340–1525B | 896B | 150–170% |
| `F.totient` (C=512) | ~878–1105B | 896B | 98–123% |
| `T.dsl_smart.*_atbash` | ~1332–1515B | 896B | same class |

Cause: occupancy-padded **shared cipher** (Atbash identical lanes; totient shared
`in[]`) hits L2 hard. Effective DRAM traffic ≪ 1 B cipher per `(C,T)` rune, so
the 1 B/rune roof understates the physical ceiling for that traffic class.

## ncu (this plate)

Workload: `parcae-bench --suite slo --extended --allow-cuda --tokens 262144 --repeats 2`  
Profile: `--launch-skip 4 --launch-count 1` after warmups.

| Kernel | Grid | `dram__bytes.sum` | Duration | DRAM SoL | SM SoL |
|--------|------|-------------------|----------|----------|--------|
| `atbash_chi2_hist_kernel` | (512,64)×256 | **1.490688 MB** | 447.17 µs | **0.37%** | 57.2% |
| `totient_chi2_hist_kernel` | (512,64)×256 | **1.47 MB** | 142.53 µs | **1.15%** | 63.8% |

```text
bytes_per_rune = dram_bytes / (C × T)
               = 1.490688e6 / (512 × 262144)  ≈ 0.01111   (Atbash)
               ≈ 1.47e6     / (512 × 262144)  ≈ 0.0110    (Totient)

peak_runes/s   = 896e9 / 0.01111  ≈ 80.7 TB runes/s
```

Spec constant: `BenchTierSpec::kHistBytesPerRuneSharedCipherOccupancy = 0.01111`  
→ `kDramRooflineSharedCipherOccupancyPeak` ≈ **80.7e12** runes/s.

Applied to: `F.atbash`, `F.totient`, `T.dsl_smart.custom_atbash`,
`T.dsl_smart.compare_Fatbash`. Unique-key Vigenère/Beaufort/Caesar stay on
**896B** @ 1 B/rune. Affine is a **separate** shared-cipher class (below).

## Sanity after Atbash/totient model fix

With the shared-cipher **DRAM diary**, quiet Atbash ~1350B → **`%peak ≈ 1.7%`**
of 80.7 TB (≤100). Totient ~880B → **`%peak ≈ 1.1%`**. Matches DRAM SoL
(compute/L2-bound).

**Spec was not lowered to quiet max** — bytes/rune came from ncu DRAM counters.

### Compute roof Done gate (2026-10-06)

90% of ≈80.7 TB is not a reachable Done while hist is atomic-bound and cipher
is L2-resident. Spec therefore freezes a separate **compute roof**:

| Constant | Value | Role |
|----------|-------|------|
| `kDramRooflineSharedCipherOccupancyPeak` | ≈**80.7e12** | Traffic **diary** (ncu bytes/rune) |
| `kSharedCipherComputeRoofRps` | **2.0e12** | **Done / Stretch** gate (`estimated_peak`) |

Calibration: `HistOccupancyRoof` identity hist+finalize @ C=512 T=2^20
(`[cuda][hist][compute_roof]`). Quiet identity plate ~1.2–2.2 TB (median ~1.6 TB);
Spec freezes **2.0 TB** with plate-noise headroom. Applied to `F.atbash`,
`F.totient`, `T.dsl_smart.custom_atbash`, `T.dsl_smart.compare_Fatbash`.

**DRAM-bound is not required** for Done on this traffic class — `pass_tier`
uses the compute roof only. Quiet Atbash ~1.1–1.7 TB → **~55–85%** of 2.0 TB
(stretch climb; absolute climb remains Commit 8).

## Affine shared-cipher (2026-10-06)

Quiet Kernel SLO intermittently printed **`%peak>100`** for `F.affine` /
`T.dsl_smart.*_affine` under the unique-key **896B** Spec (e.g. catalog median
**1142B** → 127%). Cause: C=812 unique `(a,b)` lanes still share one cipher
buffer → L2 residency; physical DRAM ≪ 1 B/rune.

ncu from [`../cache_bound/`](../cache_bound/) (`affine_hist_metrics.csv`):

| Kernel | Grid | `dram__bytes.sum` | Duration | DRAM GB/s | DRAM SoL | L2 hit | bytes/rune |
|--------|------|-------------------|----------|-----------|----------|--------|------------|
| `affine_chi2_hist_kernel` | (812,64)×256 @ T=262144 | **≈4.0566 MB** | 237.47 µs | **17.08** | **1.82%** | **97.6%** | **0.01906** |

```text
bytes_per_rune = 4056596.611 / (812 · 262144) ≈ 0.01906
peak_runes/s   = 896e9 / 0.01906 ≈ 47.0 TB
```

Spec: `BenchTierSpec::kHistBytesPerRuneAffineSharedCipher = 0.01906`  
→ `kDramRooflineAffineSharedCipherPeak` ≈ **47.0e12** runes/s.

Applied to: `F.affine`, `T.dsl_smart.custom_affine`, `T.dsl_smart.compare_Faffine`.

Under the Affine roof, prior quiet medians stay ≤100:

| Row | Quiet RPS (dsl_smart E2E med) | % of 47.0 TB |
|-----|-------------------------------|--------------|
| `custom_affine` | 964B | **≈2.05%** |
| `compare_Faffine` | 1142B | **≈2.43%** |

Absolute DRAM ~**17 GB/s** matches the operator **~16.9 GB/s** observation.

## Artifacts (local / gitignored binaries)

| File | Role |
|------|------|
| `atbash_hist.ncu-rep` | ncu Atbash hist |
| `totient_hist.ncu-rep` | ncu Totient hist |
| `../cache_bound/affine_hist.ncu-rep` | ncu Affine hist (cache_bound plate) |
| `../cache_bound/affine_hist_metrics.csv` | Affine CSV (bytes + L2) |

## Follow-up

Re-quiet done 2026-10-06:

- [`../kernel_slo/SUMMARY.md`](../kernel_slo/SUMMARY.md) — Atbash/totient model **PASS**
- [`../dsl_smart/SUMMARY.md`](../dsl_smart/SUMMARY.md) — custom Atbash model **PASS**; Affine model fixed under 0.01906 B/rune (re-quiet `%peak` below)

Affine Done still deferred (compute/L2-bound; ≪90% of 47 TB roof).
