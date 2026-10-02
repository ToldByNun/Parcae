# CUDA fused throughput reference

**Status:** local reference plateaus (not a CI gate)  
**Canonical constants:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp)  
**Canonical tool:** `parcae-bench --suite slo [--extended] --allow-cuda`  
**Compat tool:** `parcae-throughput-tiers` (thin wrapper → same `BenchSloSuite` with extended on)  
**Hardware:** NVIDIA GeForce RTX 5070 Ti — **896 GB/s** GDDR7 (published)  
**Metric:** `repeats × C × T / median-of-3 cudaEvent` (setup excluded)

## What `estimated_peak` means (physical, not measured)

`BenchTierSpec::estimated_peak` is the **DRAM roofline** for fused decrypt+χ² hist:

```text
peak_runes/s = DRAM_BW / bytes_cipher_per_rune
             = 896e9 B/s / 1 B/rune
             = 896B runes/s
```

That is what the GPU **could** sustain if the kernel were memory-bound at one
cipher byte per `(candidate,token)`. It is **not** “best bench run we saw”.
`%peak = measured / peak` is therefore ≤ **100%** by construction (physics).
If a quiet run ever prints &gt;100%, the traffic model is wrong — fix the model,
do not celebrate “super-linear” silicon.

Today’s specialized kernels sit around **~40–45%** of this roof (DRAM SoL in ncu
is only a few % — they are still compute/latency bound). S0 bytecode is ~**7–8%**.
Done = ≥**90% of 896B** (≈806B) — i.e. near memory-bound ideal.

`ThroughputTiers` / `DslPeakSanity` delegate to `BenchTierSpec`
(Catch2 `[bench][spec]` / `[dsl][peak]`).

**Theory search export** (`family=theory`) uses the **same** physical roof as
catalog fused hist (same traffic class).

### Path (transpiler → search)

```text
TheoryIr decrypt HotLoop
  → TheoryHistChi2Emit (S1 LUT-29 / S2 linear uchar4 / else S0)
  → TheoryExportCache HistPlan
  → GpuCandidateExport prefers specialized launch; soft-fallback S0 bytecode
  → export_backend=cuda
```

| Strategy | Spec id | Kernel | Physical peak (DRAM roof) |
|----------|---------|--------|---------------------------|
| S0 bytecode interpreter | `T.theory.caesar_bytecode` | `theory_chi2_hist_kernel` | **896B** |
| S1 LUT-29 (`f(x)`-only) | `T.theory.s1_lut29` | `theory_hist_chi2_s1_lut_kernel` | **896B** |
| S2 linear uchar4 | `T.theory.s2_linear` / `T.theory.progressive` | `theory_hist_chi2_s2_linear_kernel` | **896B** |

Fair gate: `parcae-bench --suite theory --allow-cuda` → `BenchTierSpec::pass_tier`
(≥90% of **896B** + `slo_min` at `T≥2^20`). Soft-fallback S0 will fail that gate
until specialize-away (or a true memory-bound S0 — unlikely). Short T is
measurement-only (`underfill_not_slo_gate`).

### Kernel SLO vs campaign wall

| Metric | Tool / log | Includes | Gate? |
|--------|------------|----------|-------|
| **Kernel SLO** | `parcae-bench --suite theory` (cudaEvent) | Device hist/finalize only | **Yes** — ≥90% DRAM roof |
| **Campaign wall** | `parcae-search-cycle` → `research/run.log` | Host prepare/bind, H2D, kernel, D2H, materialize, ingest | **No** — ops / ETA only |

Do **not** compare campaign wall at short page `T` to catalog or theory cudaEvent
peaks at `T≈2^20`. Normative Done rules:
[`theory-hist-transpile.md`](theory-hist-transpile.md). Playbook + progress
log: [`cuda-profile-theory.md`](cuda-profile-theory.md).

### Theory plateaus (physical roof)

| Tier | Workload | Ceiling (runes/s) | SLO floor | Notes |
|------|----------|-------------------|-----------|-------|
| T.theory.caesar_bytecode | S0 interpreter | **896B** | ≥15B | DRAM roof; S0 typically ~7–8% today |
| T.theory.s1_lut29 | S1 LUT-29 | **896B** | ≥15B | same roof; specialized ~45% today |
| T.theory.s2_linear | S2 progressive / bitmask | **896B** | ≥15B | same roof @ C=9 microbench |

Operator handbook: [`search-handbook.md`](search-handbook.md) § Theory URI.
Emit API: [`theory_hist_chi2_emit.hpp`](../../include/parcae/dsl/theory_hist_chi2_emit.hpp).
Contract / Done: [`theory-hist-transpile.md`](theory-hist-transpile.md).
Artifact stream vs hist paths: [`theory-artifact.md`](../spec/theory-artifact.md).

## How to measure

```bash
cmake -S . -B build-cuda -DPARCAE_BUILD_CUDA=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build-cuda --config Release --target parcae-bench parcae-throughput-tiers

# Preferred
./build-cuda/tools/Release/parcae-bench --suite slo --extended --allow-cuda

# Compat (same suite; no --allow-cuda flag)
./build-cuda/tools/Release/parcae-throughput-tiers
```

Pass rule: SLO floor **and** ≥ 90% of the **DRAM-roofline** peak for that row
(`BenchTierSpec::pass_tier`).

## Canonical SLO configs (`BenchTierSpec`)

| Tier | Workload | C | T | reps | Target (runes/s) | Ceiling |
|------|----------|---|---|------|------------------|---------|
| T1 | Caesar fused χ² | 29 | 1048576 | 64 | 15.00B–35.00B | **896B** (DRAM roof) |
| T2 | multi-key / autokey / dynamic-shift (worst) | 4096 | 262144 | 8 | 3.00B–10.00B | **896B** |
| T3 | Caesar bigram + dict | 512 | 262144 | 8 | ≥1.00B | **448B** (2 B/rune model) |

Display bands (`slo_max`) are expectations only — faster than the upper bound still
**passes**. Failures are below the floor or below the 90% peak band.

## Reference plateaus (RTX 5070 Ti)

Ceilings = **DRAM roofline** (`BenchTierSpec::dram_roofline_hist_peak` =
**896B** for 1 B cipher/rune; T3 = **448B** for 2 B/rune). Measured runs today
are far below 90% — that is expected until kernels are memory-bound.

### SLO tiers

| Tier | Workload | Ceiling (runes/s) | SLO floor |
|------|----------|-------------------|-----------|
| T1 | Caesar fused χ² | **896B** | ≥15B |
| T2 | multi-key / autokey / dynamic-shift (worst) | **896B** | ≥3B |
| T3 | Caesar bigram + dict | **448B** | ≥1B |

### Transform families

| Tier | Workload | Ceiling (runes/s) | SLO floor |
|------|----------|-------------------|-----------|
| F.atbash | Atbash fused χ² | **896B** | ≥15B |
| F.affine | Affine fused χ² (812) | **896B** | ≥15B |
| F.vigenere | Vigenère fused χ² (key len 8) | **896B** | ≥3B |
| F.beaufort | Beaufort fused χ² (key len 8) | **896B** | ≥3B |
| F.totient | Totient stream fused χ² | **896B** | ≥3B |

### Compose

| Tier | Workload | Ceiling (runes/s) | SLO floor |
|------|----------|-------------------|-----------|
| C.koan1_fused | Atbash→Caesar+shift fused χ² | **896B** | ≥15B |
| C.koan1_stages | Atbash (async) + Caesar χ² | **896B** | ≥15B |

## Recalibration rule

1. Peak = **physical DRAM roofline**, not measured max. Derive from
   `device_peak_dram_bytes_per_s / bytes_per_rune` (RTX 5070 Ti → 896e9 /
   448e9). Do **not** raise or lower the Spec to chase quiet-run medians.
2. Run `parcae-bench --suite slo --extended --allow-cuda` (or compat
   `parcae-throughput-tiers`) on a quiet GPU to measure **progress toward**
   that roof (`%peak`), not to redefine it.
3. `%peak` **must stay ≤100**. If a quiet run prints **&gt;100**, the Spec
   peak or BW assumption is wrong (wrong bytes/rune or wrong DRAM GB/s) —
   fix the roof model, not “raise to absorb” a measured outlier.
4. PRIMARY / climb pass = **≥90% of the DRAM roof**. Below that is unfinished
   specialize work, not a calibration problem.
5. If mins fall under 90% of the roof while the kernel is already memory-bound
   on ncu, check clocks / residency / traffic bytes before touching the Spec.

## Related

- Canonical specs: [`include/parcae/bench/bench_tier_spec.hpp`](../../include/parcae/bench/bench_tier_spec.hpp)
- Suite map / headers: [`include/parcae/bench/README.md`](../../include/parcae/bench/README.md)
- CLI contract: [`docs/spec/tools.md`](../spec/tools.md) § `parcae-bench`
- Agent deny-list: [`docs/spec/agent-tools.md`](../spec/agent-tools.md) (`parcae-bench` / `parcae-throughput-tiers`)
- Implementation (runner): [`include/parcae/run/throughput_tiers.hpp`](../../include/parcae/run/throughput_tiers.hpp)
- DSL compile-time mirror (no CUDA): [`include/parcae/dsl/dsl_peak_sanity.hpp`](../../include/parcae/dsl/dsl_peak_sanity.hpp)
- Theory hist emit: [`include/parcae/dsl/theory_hist_chi2_emit.hpp`](../../include/parcae/dsl/theory_hist_chi2_emit.hpp)
- Kernels: [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (`hist_fast.hpp`, `*_chi2_batch.cu`, `theory_hist_chi2_s{1,2}.*`)
- Build notes: [cuda-build.md](cuda-build.md)
- Theory profiling (nsys/ncu): [cuda-profile-theory.md](cuda-profile-theory.md)
- Theory hist transpile contract (S0–S5 ≥90% shape peak): [theory-hist-transpile.md](theory-hist-transpile.md)
- Operator handbook (theory URI + dispatch): [search-handbook.md](search-handbook.md)
- Transpiler + HistChi2 strategies: [python-transpiler.md](python-transpiler.md)