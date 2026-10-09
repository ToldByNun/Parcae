# DSL smart customs — fair Kernel SLO digests

Hand-written HotLoop customs (no catalog `TransformId`) vs specialized twins
on the **same** C/T. PRIMARY gate is fair `T≥2^20` vs shape `estimated_peak`
(`BenchTierSpec`). Campaign wall is never PRIMARY.

**Contract:** [`dsl-smart-hist.md`](../../dsl-smart-hist.md)  
**Remap roofs:** [`../../hist-alphabet-remap.md`](../../hist-alphabet-remap.md)  
**Traffic model:** [`../traffic_model/SUMMARY.md`](../traffic_model/SUMMARY.md)  
**Tool:** `parcae-bench --suite dsl_smart --allow-cuda`  
**Capture:** [`scripts/cuda/capture_dsl_smart.ps1`](../../../../scripts/cuda/capture_dsl_smart.ps1)

| Spec row | Workload | Twin | Peak |
|----------|----------|------|------|
| `T.dsl_smart.custom_atbash` | ShapeInline Atbash (`28−x`) | `compare_Fatbash` (`F.atbash`) | compute roof **2.0 TB** |
| `T.dsl_smart.custom_caesar` | ShapeInline Caesar | `compare_caesar` (CaesarChi2Batch) | alphabet Remap **2.0 TB** |
| `T.dsl_smart.custom_affine` | ShapeInline Affine decrypt | `compare_Faffine` (`F.affine`) | alphabet Remap **2.0 TB** |
| `T.dsl_smart.custom_linear` | S2 linear `x±(b0+b1·i)` | `compare_S2` (TheoryHistChi2S2) | column Remap **2.0 TB** |
| `T.dsl_smart.custom_autokey` | S4 AutokeyRing vigenere_lag | `compare_S4` (TheoryHistChi2S4) | lag Remap **2.0 TB** |

Quiet ACCEPTANCE + stretch/Done table: [`SUMMARY.md`](SUMMARY.md).
