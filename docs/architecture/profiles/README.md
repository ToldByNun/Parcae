# Theory CUDA profile reports (local)

nsys / ncu outputs from
[`cuda-profile-theory.md`](../cuda-profile-theory.md) land here.

Binary reports (`.ncu-rep` / `.nsys-rep`) are **gitignored**. Text summaries and
JSON digests under `baseline/` / `specialized/` / `s0-climb/` / `kernel_slo/` /
`roof_hist/` / `export_duty/` / `dsl_smart/` / `traffic_model/` / `cache_bound/`
may be committed — see
[`baseline/SUMMARY.md`](baseline/SUMMARY.md),
[`specialized/SUMMARY.md`](specialized/SUMMARY.md),
[`s0-climb/SUMMARY.md`](s0-climb/SUMMARY.md),
[`kernel_slo/SUMMARY.md`](kernel_slo/SUMMARY.md),
[`roof_hist/SUMMARY.md`](roof_hist/SUMMARY.md),
[`export_duty/SUMMARY.md`](export_duty/SUMMARY.md),
[`dsl_smart/`](dsl_smart/) (customs without presets),
[`traffic_model/SUMMARY.md`](traffic_model/SUMMARY.md) (Atbash/totient bytes/rune),
and [`cache_bound/SUMMARY.md`](cache_bound/SUMMARY.md) (L2/DRAM absolute GB/s).
Remap traffic + 93.5%/896B reinterpretation:
[`../hist-alphabet-remap.md`](../hist-alphabet-remap.md).
Quiet Remap Spec plate (2026-10-09 roofs): [`quiet_remap/SUMMARY.md`](quiet_remap/SUMMARY.md).

Wrapper: [`scripts/cuda/profile_theory_hist.ps1`](../../../scripts/cuda/profile_theory_hist.ps1)  
Baseline capture: [`scripts/cuda/capture_theory_baseline.ps1`](../../../scripts/cuda/capture_theory_baseline.ps1)  
Post–specialized emit: [`scripts/cuda/capture_theory_specialized.ps1`](../../../scripts/cuda/capture_theory_specialized.ps1)  
S0 climb baseline (S0 vs Caesar twin): [`scripts/cuda/capture_s0_climb_baseline.ps1`](../../../scripts/cuda/capture_s0_climb_baseline.ps1)  
Cache / DRAM evidence (L2 hit + `dram__bytes.sum`): [`scripts/cuda/capture_cache_bound.ps1`](../../../scripts/cuda/capture_cache_bound.ps1) → [`cache_bound/`](cache_bound/).  
Kernel SLO ACCEPTANCE (metric B): quiet `parcae-bench --suite theory` +
`slo --extended` → [`kernel_slo/`](kernel_slo/).  
Smart hist customs: contract [`dsl-smart-hist.md`](../dsl-smart-hist.md) →
[`dsl_smart/`](dsl_smart/) via `parcae-bench --suite dsl_smart` +
[`scripts/cuda/capture_dsl_smart.ps1`](../../../scripts/cuda/capture_dsl_smart.ps1).
