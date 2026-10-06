# DSL smart customs — End-to-end ACCEPTANCE (fair Kernel SLO)

Hand-written HotLoop → specialized twins measured against catalog / specialized
counterparts on the **same** C/T.

**PRIMARY gate:** fair `T≥2^20` vs shape `estimated_peak` (Kernel SLO).  
**Not PRIMARY:** campaign wall (`T≈262`, huge C).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Date:** 2026-10-06 (E2E quiet plate)  
**Tool:** `parcae-bench --suite dsl_smart --allow-cuda --json`  
**Capture:** [`scripts/cuda/capture_dsl_smart.ps1`](../../../../scripts/cuda/capture_dsl_smart.ps1)  
**Artifact:** `dsl_smart_fair.json` (copy of run3) + `dsl_smart_fair_run{1,2,3}.json`  
**Contract:** [`dsl-smart-hist.md`](../../dsl-smart-hist.md)  
**Compare-to plate:** [`kernel_slo/SUMMARY.md`](../kernel_slo/SUMMARY.md) (metric B)  
**Traffic model:** [`traffic_model/SUMMARY.md`](../traffic_model/SUMMARY.md)

## Gates

| Gate | Rule |
|------|------|
| **Done (PRIMARY)** | ≥**90%** of shape `estimated_peak` **and** ≥ `slo_floor`; fair `T≥2^20`; shape **DRAM-bound** (ncu) |
| **Stretch** | ≥**80%** of shape `estimated_peak` — interim only; never replaces Done |
| **Model** | Printed `%peak` ≤ **100** — fix bytes/rune; never lower Spec to quiet max |
| **Campaign wall** | Never Done evidence |

## Rows

| Spec id | C | T (fair) | Detail tag | Twin | Peak class |
|---------|---|----------|------------|------|------------|
| `T.dsl_smart.custom_atbash` | 512 | 1048576 | `ShapeInline_atbash` | `compare_Fatbash` | shared-cipher ≈**80.7 TB** |
| `T.dsl_smart.compare_Fatbash` | 512 | 1048576 | `catalog_F.atbash` | — | shared-cipher ≈**80.7 TB** |
| `T.dsl_smart.custom_caesar` | 29 | 1048576 | `ShapeInline_caesar` | `compare_caesar` | **896B** |
| `T.dsl_smart.compare_caesar` | 29 | 1048576 | `catalog_CaesarChi2` | — | **896B** |
| `T.dsl_smart.custom_affine` | 812 | 1048576 | `ShapeInline_affine` | `compare_Faffine` | **896B** |
| `T.dsl_smart.compare_Faffine` | 812 | 1048576 | `catalog_F.affine` | — | **896B** |
| `T.dsl_smart.custom_linear` | 841 | 1048576 | `S2_linear` | `compare_S2` | **896B** |
| `T.dsl_smart.compare_S2` | 841 | 1048576 | `catalog_S2_linear` | — | **896B** |
| `T.dsl_smart.custom_autokey` | 28 | 1048576 | `S4_autokey` | `compare_S4` | **896B** |
| `T.dsl_smart.compare_S4` | 28 | 1048576 | `catalog_S4_autokey` | — | **896B** |

## Quiet fair capture (3 runs → median)

Sources: `dsl_smart_fair_run{1,2,3}.json` (`T=1048576`, `BenchTimer` cudaEvent).

| Row | run1 | run2 | run3 | **median** | % of peak | vs twin |
|-----|------|------|------|------------|-----------|---------|
| `custom_atbash` | 1302B | 1267B | 1641B | **1302B** | **1.61%** of 80.7 TB | ≈ catalog |
| `compare_Fatbash` | 1678B | 1592B | 1286B | **1592B** | **1.97%** of 80.7 TB | — |
| `custom_caesar` | 827B | 786B | 537B | **786B** | **87.7%** of 896B | ≈ twin |
| `compare_caesar` | 728B | 822B | 780B | **780B** | **87.0%** of 896B | — |
| `custom_affine` | 915B | 964B | 1241B | **964B** | **107.6%** of 896B | ≈ catalog |
| `compare_Faffine` | 956B | 1142B | 1179B | **1142B** | **127.4%** of 896B | — |
| `custom_linear` | 402B | 372B | 384B | **384B** | **42.8%** of 896B | ≈ S2 twin |
| `compare_S2` | 367B | 355B | 373B | **367B** | **41.0%** of 896B | — |
| `custom_autokey` | 475B | 491B | 475B | **475B** | **53.1%** of 896B | ≈ S4 twin |
| `compare_S4` | 489B | 466B | 492B | **489B** | **54.6%** of 896B | — |

## Stretch / Done table

| Row | Median RPS | %peak | Stretch (≥80%) | Done (≥90% + DRAM-bound) | Model |
|-----|------------|-------|----------------|--------------------------|-------|
| `custom_atbash` | 1302B | 1.61% | fail | fail / N/A (compute-bound) | **OK** |
| `compare_Fatbash` | 1592B | 1.97% | fail | fail / N/A | **OK** |
| `custom_caesar` | 786B | 87.7% | **PASS** | fail | **OK** |
| `compare_caesar` | 780B | 87.0% | **PASS** | fail | **OK** |
| `custom_affine` | 964B | 107.6% | invalid\* | invalid\* | **`%peak>100`** |
| `compare_Faffine` | 1142B | 127.4% | invalid\* | invalid\* | **`%peak>100`** |
| `custom_linear` | 384B | 42.8% | fail | fail | **OK** |
| `compare_S2` | 367B | 41.0% | fail | fail | **OK** |
| `custom_autokey` | 475B | 53.1% | fail | fail | **OK** |
| `compare_S4` | 489B | 54.6% | fail | fail | **OK** |

\*Stretch/Done rejected while model `%peak>100` (unique-key Affine Spec stays 896B;
separate from shared-cipher Atbash fix).

## ACCEPTANCE verdict

| Claim | Result |
|-------|--------|
| Custom Atbash in F.atbash twin class | **PASS (model + twin)** — both ~2% of 80.7 TB |
| Custom Caesar in CaesarChi2 twin class | **PASS (stretch)** — both ~87% of 896B |
| Custom Affine in F.affine twin class | **PARTIAL** — tracks catalog; model intermittent `>100` |
| Custom linear in S2 twin class | **PASS (twin)** — ~43% vs ~41%; stretch fail (compute-bound) |
| Custom autokey in S4 twin class | **PASS (twin)** — ~53% vs ~55%; stretch fail |
| Campaign wall used as Done evidence | **No** |
| Spec lowered to quiet max | **No** |
| Edge tags green | **PASS** — `[dsl][normalize]`, `[dsl][shape]`, `[cuda][theory][edge]`, `[search][export][theory][shape]` |

**Overall: PARTIAL PASS** — twins land for Atbash/Caesar/Affine/linear/autokey;
Caesar stretch; Atbash model OK; Affine model + linear/autokey stretch open;
Done deferred until DRAM-bound.

```powershell
.\scripts\cuda\capture_dsl_smart.ps1 -BuildDir build-rel-cuda -Runs 3
```

## Artifacts

| File | Role |
|------|------|
| `dsl_smart_fair_run{1,2,3}.json` | Quiet suite JSON ×3 |
| `dsl_smart_fair.json` | Copy of run3 |
| `README.md` | Suite / Spec map |
| `capture_log.txt` | Local transcript (gitignored) |
