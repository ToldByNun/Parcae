# Kernel SLO ACCEPTANCE — metric B (requiet 2026-10-06)

Quiet fair Kernel SLO vs shape `estimated_peak`. Default fused hist @ **1 B/rune →
896B**; Atbash/totient Done uses shared-cipher **compute roof → 2.0 TB**
(`kSharedCipherComputeRoofRps`). DRAM occupancy diary ≈**80.7 TB** (ncu 0.01111
B/rune) — [`../traffic_model/SUMMARY.md`](../traffic_model/SUMMARY.md).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Tool:** `parcae-bench --suite theory --allow-cuda --json` and
`parcae-bench --suite slo --extended --allow-cuda --json`  
**Protocol:** `BenchTimer` cudaEvent (4 warmups + median-of-3), setup/H2D excluded  
**Playbook:** [`cuda-profile-theory.md`](../../cuda-profile-theory.md)

**idle GPU ≠ Kernel SLO PCIe** — metric B is kernel rates only.

## Gates

| Gate | Rule | Applies when |
|------|------|--------------|
| **Stretch** | ≥**80%** of shape `estimated_peak` | Caesar / S1 / Atbash-class quiet fair |
| **Done** | ≥**90%** of shape `estimated_peak` | Unique-key: prefer DRAM-bound; **shared-cipher Atbash/totient: compute roof only (DRAM-bound N/A)** |
| **Model** | `%peak` must stay ≤100 | Always — if >, fix roof model, do **not** lower Spec |

Prior ncu on hist kernels: DRAM SoL ~**1–5%** (Atbash/totient ~0.4–1%). Unique-key
Done still wants DRAM-bound aspiration. Shared-cipher Atbash/totient gate on the
**2.0 TB** compute roof (quiet plate ~55–80% — stretch climb open).

## Theory fair (3 quiet runs → median) — plate 2026-10-03

Sources: `theory_fair_run{1,2,3}.json` (`T=1048576`). Unchanged by traffic-model
Spec (still 896B @ 1 B/rune).

| Row | run1 | run2 | run3 | **median** | % of 896B | Stretch | Done | Notes |
|-----|------|------|------|------------|-----------|---------|------|-------|
| `T.theory.compare_caesar` | 849.3B | 782.9B | 780.1B | **782.9B** | **87.4%** | **PASS** | 1/3 pass (run1 94.8%) | fat-64 catalog twin |
| `T.theory.caesar_bytecode` | 679.9B | 693.4B | 680.8B | **680.8B** | **76.0%** | **fail** | fail | `specialize_S1`; stable ~76–77% |
| `T.theory.s1_lut29` | 303.4B | 487.0B | 692.2B | **487.0B** | **54.4%** | **fail** | fail | high run-to-run noise (bake inside timer) |

### S1 LUT residency (2026-10-07)

Fair `T.theory.s1_lut29` bakes the device LUT **once outside** `BenchTimer`
(`detail=S1_lut_resident`). Export (`TheoryDeviceScratch`) skips rebake when
URI + ops/imm + slot layout + bound slots match the launch slab.

Quiet theory ×6: **670–733B** (~75–82% of 896B). Median of the six ~**705B**.
Stretch **716.8B** is inside the band but not the median. Domain `+inf` and
C=1 / empty-T rejects stay green.
| `T.theory.progressive` (S2 @841) | 401.6B | 394.0B | 403.5B | **401.6B** | **44.8%** | fail | fail | ks29 plate (pre-residue) |

### S2 running residue requiet (2026-10-07)

Hot-loop `% 29` → running residue; `HistTileCap` S2 default tile **32**
(sweep: 32 > 64/128). Catch2 `[cuda][hist][s2][residue]` + `parcae-bench --suite theory`.

| Row | run1 | run2 | run3 | **median** | % of 896B | Stretch | Done | Notes |
|-----|------|------|------|------------|-----------|---------|------|-------|
| `T.theory.progressive` (S2 @841) | 763.6B | 683.4B | 657.2B | **683.4B** | **76.3%** | fail (1/3 pass) | fail | +~70% vs ks29; stretch 716.8B not median-stable |

## Catalog SLO extended (requiet 2026-10-06, 2 quiet runs)

Sources: `slo_extended.json`, `slo_extended_run2.json`.

| Row | run1 | run2 | Stretch (≥80%) | Done (≥90%) | Model |
|-----|------|------|----------------|-------------|-------|
| T1 Caesar | 759.3B (85%) | 600.2B (67%) | borderline | fail | OK |
| F.atbash | 1600B (**80.0%** of 2.0 TB) | 1437B (**71.9%**) | borderline / fail | fail | **OK** (`%peak≤100`) |
| F.affine | 819B (91%) | 771B (86%) | stretch / stretch | Done* / fail | OK this plate |
| F.vigenere | 835B (93%) | 857B (96%) | **PASS** | **PASS** | OK under 1 B/rune |
| F.beaufort | 854B (95%) | 872B (97%) | **PASS** | **PASS** | OK |
| F.totient | 1109B (**55.5%** of 2.0 TB) | 1091B (**54.6%**) | fail | fail | **OK** (`%peak≤100`) |
| C.koan1_fused | 720B (80%) | 575B (64%) | borderline | fail | OK |
| T2 / T3 | — | — | fail | fail | T3 roof 448B; not hist focus |

\*Unique-key Done claims prefer DRAM-bound (ncu). Affine still on ≈47.0 TB DRAM
shared-cipher class. Atbash/totient Done = ≥90% of **2.0 TB** compute roof
(DRAM-bound N/A).

## ACCEPTANCE verdict (metric B)

| Claim | Result |
|-------|--------|
| Stretch ≥80% — Caesar-class (`compare_caesar` / T1) | **PASS** (median twin **87%**; T1 often ~80%) |
| Stretch ≥80% — S1 / fair specialize | **FAIL** (median specialize **76%**; S1 noisy) |
| Model — Atbash / totient `%peak≤100` | **PASS** — under compute roof **2.0 TB** (DRAM diary kept) |
| Stretch / Done — Atbash-class vs compute roof | **FAIL** — ~72–80% / ~55%; absolute climb open |
| Done ≥90% unique-key while DRAM-bound | **N/A** — hist still compute-bound on unique-key |
| Spec lowered to quiet max | **No** — compute roof from identity hist; not Atbash quiet max |

**Overall: PARTIAL PASS**

Landed: Caesar catalog twin stretch; F.vigenere/beaufort ≥90% under 1 B/rune;
Atbash/totient **model gate** under compute roof. Open: S1/specialize stretch
(≥80%), S2 roof climb, Atbash/totient absolute climb toward 2.0 TB Done.

## Artifacts

| File | Role |
|------|------|
| `theory_fair_run{1,2,3}.json` | Quiet theory suite (3×, 2026-10-03) |
| `theory_fair.json` | Copy of run3 (representative hot S1) |
| `theory_fair.txt` | Human table (pre-JSON capture) |
| `slo_extended.json` / `slo_extended_run2.json` | Quiet SLO extended requiet (2×, 2026-10-06) |
| `slo_extended.txt` | Human table |

## Next

1. S1 fair median ≥**716.8B** (stretch) — residency landed (~670–733B); remaining gap is hist vs Caesar twin.
2. S2 median ≥**716.8B** stretch (residue+tile32 at ~76%; remaining atomic/L2).
3. Atbash/totient absolute climb toward **≥1.8 TB** (90% of 2.0 TB compute roof).
4. Combined write-up: [`gpu_full_load/SUMMARY.md`](../gpu_full_load/SUMMARY.md)
   (metric A duty + metric B roof).
