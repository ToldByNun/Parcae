# Theory CUDA profile reports (local)

nsys / ncu outputs from
[`cuda-profile-theory.md`](../cuda-profile-theory.md) land here.

Binary reports (`.ncu-rep` / `.nsys-rep`) are **gitignored**. Text summaries and
JSON digests under `baseline/` / `specialized/` / `s0-climb/` may be committed — see
[`baseline/SUMMARY.md`](baseline/SUMMARY.md),
[`specialized/SUMMARY.md`](specialized/SUMMARY.md), and
[`s0-climb/SUMMARY.md`](s0-climb/SUMMARY.md).

Wrapper: [`scripts/cuda/profile_theory_hist.ps1`](../../../scripts/cuda/profile_theory_hist.ps1)  
Baseline capture: [`scripts/cuda/capture_theory_baseline.ps1`](../../../scripts/cuda/capture_theory_baseline.ps1)  
Post–specialized emit: [`scripts/cuda/capture_theory_specialized.ps1`](../../../scripts/cuda/capture_theory_specialized.ps1)  
S0 climb baseline (S0 vs Caesar twin): [`scripts/cuda/capture_s0_climb_baseline.ps1`](../../../scripts/cuda/capture_s0_climb_baseline.ps1).
