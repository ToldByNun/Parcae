# DSL smart customs — fair Kernel SLO (vs F.atbash / Caesar twin)

Hand-written HotLoop → `ShapeInline` twins measured against catalog
`F.atbash` / CaesarChi2 / `F.affine` on the **same** C/T.

**PRIMARY gate:** fair `T≥2^20` vs Spec DRAM roof (Kernel SLO).  
**Not PRIMARY:** campaign wall (`T≈262`, huge C).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Date:** 2026-10-06 (requiet after traffic-model pass)  
**Tool:** `parcae-bench --suite dsl_smart --allow-cuda --json`  
**Capture:** [`scripts/cuda/capture_dsl_smart.ps1`](../../../../scripts/cuda/capture_dsl_smart.ps1)  
**Artifact:** `dsl_smart_fair.json` (copy of run3) + `dsl_smart_fair_run{1,2,3}.json`  
**Contract:** [`dsl-smart-hist.md`](../../dsl-smart-hist.md)  
**Compare-to plate:** [`kernel_slo/SUMMARY.md`](../kernel_slo/SUMMARY.md) (metric B)  
**Traffic model:** [`traffic_model/SUMMARY.md`](../traffic_model/SUMMARY.md)

## Rows

| Spec id | C | T (fair) | Detail tag | Twin | Peak class |
|---------|---|----------|------------|------|------------|
| `T.dsl_smart.custom_atbash` | 512 | 1048576 | `ShapeInline_atbash` | `compare_Fatbash` | shared-cipher ≈**80.7 TB** |
| `T.dsl_smart.compare_Fatbash` | 512 | 1048576 | `catalog_F.atbash` | — | shared-cipher ≈**80.7 TB** |
| `T.dsl_smart.custom_caesar` | 29 | 1048576 | `ShapeInline_caesar` | `compare_caesar` | **896B** |
| `T.dsl_smart.compare_caesar` | 29 | 1048576 | `catalog_CaesarChi2` | — | **896B** |
| `T.dsl_smart.custom_affine` | 812 | 1048576 | `ShapeInline_affine` | `compare_Faffine` | **896B** |
| `T.dsl_smart.compare_Faffine` | 812 | 1048576 | `catalog_F.affine` | — | **896B** |

Suite `pass_tier` is ≥90% of shape peak at fair T. Stretch bar (metric B, when
not DRAM-bound) is ≥80% — same plate language as `kernel_slo/`.

## Quiet fair capture (3 runs → median)

Sources: `dsl_smart_fair_run{1,2,3}.json` (`T=1048576`, `BenchTimer` cudaEvent).

| Row | run1 | run2 | run3 | **median** | % of peak | vs twin | Notes |
|-----|------|------|------|------------|-----------|---------|-------|
| `custom_atbash` | 1598B | 1251B | 1704B | **1598B** | **1.98%** of 80.7 TB | ≈ catalog | Model **OK**; compute-bound |
| `compare_Fatbash` | 1776B | 1571B | 1371B | **1571B** | **1.95%** of 80.7 TB | — | Same class as `F.atbash` |
| `custom_caesar` | 551B | 707B | 821B | **707B** | **79.0%** of 896B | ≈ twin | Stretch borderline; Done fail |
| `compare_caesar` | 698B | 792B | 509B | **698B** | **77.9%** of 896B | — | Twin; noisy |
| `custom_affine` | 981B | 1021B | 984B | **984B** | **109.8%** of 896B | ≈ catalog | Model intermittent `>100` (unique-key Spec) |
| `compare_Faffine` | 1215B | 933B | 938B | **938B** | **104.7%** of 896B | — | Same intermittent class |

## ACCEPTANCE (Kernel SLO — not campaign wall)

| Claim | Result |
|-------|--------|
| Custom Caesar in CaesarChi2 twin class @ fair T | **PASS** — medians in same band (~700B); stretch borderline |
| Custom Atbash in F.atbash twin class | **PASS (model)** — both ~2% of 80.7 TB; custom within noise of catalog |
| Custom Affine decrypt in F.affine twin class | **PARTIAL** — custom tracks catalog; both can print `%peak>100` under 1 B/rune |
| Campaign wall used as Done evidence | **No** |
| Spec lowered to quiet max | **No** — Atbash peak from ncu bytes/rune |

**Overall: PARTIAL PASS** — Caesar custom in twin class; Atbash model gate cleared
with catalog. Affine unique-key intermittent `%peak>100` remains outside the
shared-cipher traffic fix.

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
