# DSL smart customs — fair Kernel SLO (vs F.atbash / Caesar twin)

Hand-written HotLoop → `ShapeInline` twins measured against catalog
`F.atbash` / CaesarChi2 / `F.affine` on the **same** C/T.

**PRIMARY gate:** fair `T≥2^20` vs Spec DRAM roof **896B** (Kernel SLO).  
**Not PRIMARY:** campaign wall (`T≈262`, huge C).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Date:** 2026-10-04  
**Tool:** `parcae-bench --suite dsl_smart --allow-cuda --json`  
**Capture:** [`scripts/cuda/capture_dsl_smart.ps1`](../../../../scripts/cuda/capture_dsl_smart.ps1)  
**Artifact:** `dsl_smart_fair.json`  
**Contract:** [`dsl-smart-hist.md`](../../dsl-smart-hist.md)  
**Compare-to plate:** [`kernel_slo/SUMMARY.md`](../kernel_slo/SUMMARY.md) (metric B)

## Rows

| Spec id | C | T (fair) | Detail tag | Twin |
|---------|---|----------|------------|------|
| `T.dsl_smart.custom_atbash` | 512 | 1048576 | `ShapeInline_atbash` | `compare_Fatbash` |
| `T.dsl_smart.compare_Fatbash` | 512 | 1048576 | `catalog_F.atbash` | — |
| `T.dsl_smart.custom_caesar` | 29 | 1048576 | `ShapeInline_caesar` | `compare_caesar` |
| `T.dsl_smart.compare_caesar` | 29 | 1048576 | `catalog_CaesarChi2` | — |
| `T.dsl_smart.custom_affine` | 812 | 1048576 | `ShapeInline_affine` | `compare_Faffine` |
| `T.dsl_smart.compare_Faffine` | 812 | 1048576 | `catalog_F.affine` | — |

Peak / floors: `BenchTierSpec` `T.dsl_smart.*` (896B peak, 15B floor). Suite
`pass_tier` is ≥90% peak at fair T. Stretch bar (metric B, when not DRAM-bound)
is ≥80% — same plate language as `kernel_slo/`.

## Quiet fair capture (single plate)

Source: `dsl_smart_fair.json` (`T=1048576`, `BenchTimer` cudaEvent).

| Row | RPS | % of 896B | vs twin | Notes |
|-----|-----|-----------|---------|-------|
| `custom_atbash` | **1332B** | **148.6%** | −12% vs F.atbash | Same model defect class as catalog Atbash (`%peak>100`) |
| `compare_Fatbash` | **1515B** | **169.1%** | — | Catalog twin; matches kernel_slo Atbash-class invalid |
| `custom_caesar` | **772B** | **86.2%** | **+4%** vs CaesarChi2 | Stretch ≥80% **PASS**; Done 90% fail (same class as twin) |
| `compare_caesar` | **743B** | **82.9%** | — | Catalog twin; aligns with `T.theory.compare_caesar` quiet ~83–87% |
| `custom_affine` | **1128B** | **125.9%** | **+5%** vs F.affine | `%peak>100` — traffic-model defect (shared with catalog) |
| `compare_Faffine` | **1070B** | **119.4%** | — | Catalog twin; intermittent `>100` also on kernel_slo plate |

## ACCEPTANCE (Kernel SLO — not campaign wall)

| Claim | Result |
|-------|--------|
| Custom Caesar in CaesarChi2 twin class @ fair T | **PASS** — 772B vs 743B (~same band; both stretch, neither Done) |
| Custom Atbash in F.atbash twin class | **BLOCKED (model)** — both `%peak>100`; custom within noise of catalog |
| Custom Affine decrypt in F.affine twin class | **BLOCKED (model)** — both `%peak>100`; custom slightly above catalog |
| Campaign wall used as Done evidence | **No** |
| Spec lowered to quiet max | **No** — Spec stays **896B** |

**Overall: PARTIAL PASS** — Caesar custom proves catalog-twin Kernel SLO class.
Atbash / Affine customs match their catalog twins but inherit the shared
bytes/rune model defect (`%peak>100`) already flagged under `kernel_slo/`.

```powershell
.\scripts\cuda\capture_dsl_smart.ps1 -BuildDir build-rel-cuda -Runs 3
```

## Artifacts

| File | Role |
|------|------|
| `dsl_smart_fair.json` | Quiet suite JSON (this plate) |
| `README.md` | Suite / Spec map |
| `capture_log.txt` | Local transcript (gitignored) |
