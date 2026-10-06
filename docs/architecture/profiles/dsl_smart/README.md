# DSL smart customs — fair Kernel SLO digests

Hand-written HotLoop customs (no catalog `TransformId`) vs catalog twins
on the **same** C/T. PRIMARY gate is fair `T≥2^20` vs shape `estimated_peak`
(`BenchTierSpec`): Atbash twins use shared-cipher occupancy (~**80.7 TB**);
Caesar/Affine stay on **896B** @ 1 B/rune. Campaign wall is never PRIMARY.

**Contract:** [`dsl-smart-hist.md`](../../dsl-smart-hist.md)  
**Traffic model:** [`../traffic_model/SUMMARY.md`](../traffic_model/SUMMARY.md)  
**Tool:** `parcae-bench --suite dsl_smart --allow-cuda`  
**Capture:** [`scripts/cuda/capture_dsl_smart.ps1`](../../../../scripts/cuda/capture_dsl_smart.ps1)

| Spec row | Workload | Twin | Peak |
|----------|----------|------|------|
| `T.dsl_smart.custom_atbash` | ShapeInline Atbash (`28−x`) | `T.dsl_smart.compare_Fatbash` (`F.atbash`) | ≈80.7 TB |
| `T.dsl_smart.custom_caesar` | ShapeInline Caesar | `T.dsl_smart.compare_caesar` (CaesarChi2Batch) | 896B |
| `T.dsl_smart.custom_affine` | ShapeInline Affine decrypt | `T.dsl_smart.compare_Faffine` (`F.affine`) | 896B |

Quiet ACCEPTANCE numbers live in [`SUMMARY.md`](SUMMARY.md) when captured.
