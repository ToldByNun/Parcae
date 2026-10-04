# DSL smart customs — fair Kernel SLO digests

Hand-written HotLoop customs (no catalog `TransformId`) vs catalog twins
on the **same** C/T. PRIMARY gate is fair `T≥2^20` vs the 896B DRAM roof
(`BenchTierSpec`); campaign wall is never PRIMARY.

**Contract:** [`dsl-smart-hist.md`](../../dsl-smart-hist.md)  
**Tool:** `parcae-bench --suite dsl_smart --allow-cuda`  
**Capture:** [`scripts/cuda/capture_dsl_smart.ps1`](../../../../scripts/cuda/capture_dsl_smart.ps1)

| Spec row | Workload | Twin |
|----------|----------|------|
| `T.dsl_smart.custom_atbash` | ShapeInline Atbash (`28−x`) | `T.dsl_smart.compare_Fatbash` (`F.atbash`) |
| `T.dsl_smart.custom_caesar` | ShapeInline Caesar | `T.dsl_smart.compare_caesar` (CaesarChi2Batch) |
| `T.dsl_smart.custom_affine` | ShapeInline Affine decrypt | `T.dsl_smart.compare_Faffine` (`F.affine`) |

Quiet ACCEPTANCE numbers live in [`SUMMARY.md`](SUMMARY.md) when captured.
