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
`T.dsl_smart.compare_Fatbash`. Unique-key families (`F.affine`, Caesar, …)
stay on **896B** @ 1 B/rune.

## Sanity after model fix

With the shared-cipher roof, quiet Atbash ~1350B → **`%peak ≈ 1.7%`** (≤100).
Totient ~880B → **`%peak ≈ 1.1%`**. Matches DRAM SoL (compute-bound; Done
deferred until memory-bound).

**Spec was not lowered to quiet max** — bytes/rune came from ncu DRAM counters.

## Artifacts (local / gitignored binaries)

| File | Role |
|------|------|
| `atbash_hist.ncu-rep` | ncu Atbash hist |
| `totient_hist.ncu-rep` | ncu Totient hist |

## Follow-up

Re-quiet done 2026-10-06:

- [`../kernel_slo/SUMMARY.md`](../kernel_slo/SUMMARY.md) — Atbash/totient model **PASS**
- [`../dsl_smart/SUMMARY.md`](../dsl_smart/SUMMARY.md) — custom Atbash model **PASS**

Affine unique-key intermittent `%peak>100` is a separate traffic class (still
1 B/rune Spec).
