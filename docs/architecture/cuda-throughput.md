# CUDA fused throughput reference

**Status:** local reference plateaus (not a CI gate)  
**Canonical constants:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp)  
**Canonical tool:** `parcae-bench --suite slo [--extended] --allow-cuda`  
**Compat tool:** `parcae-throughput-tiers` (thin wrapper → same `BenchSloSuite` with extended on)  
**Hardware:** NVIDIA GeForce RTX 5070 Ti — **896 GB/s** GDDR7 (published)  
**Metric:** `repeats × C × T / median-of-3 cudaEvent` (setup excluded)

## What `estimated_peak` means (physical, not measured)

`BenchTierSpec::estimated_peak` is the shape ceiling used by Done/Stretch:

```text
# Unique-key fused hist (Caesar / S1 / S2 / Vigenère / …)
peak_runes/s = DRAM_BW / bytes_cipher_per_rune
             = 896e9 B/s / 1 B/rune
             = 896B runes/s

# Shared-cipher occupancy (F.atbash / F.totient / dsl_smart Atbash)
peak_runes/s = kSharedCipherComputeRoofRps   # SM/atomic capacity
             = 2.0e12                         # frozen 2026-10-06 identity hist
# DRAM occupancy diary (ncu 0.01111 B/rune → ≈80.7 TB) is NOT the Done gate.
```

Unique-key peaks are what the GPU **could** sustain if memory-bound at the
traffic model. Shared-cipher Atbash/totient peaks are the **compute roof**
(identity occupancy hist calibration) — cipher is L2-resident and hist is
atomic-bound, so 90% of the DRAM diary (~80.7 TB) is not a reachable Done.
`%peak = measured / peak` is therefore ≤ **100%** by construction. If a quiet
run ever prints &gt;100%, the roof model is wrong — fix the model, do not
celebrate “super-linear” silicon.

Quiet ACCEPTANCE ([`profiles/kernel_slo/SUMMARY.md`](profiles/kernel_slo/SUMMARY.md)):
Caesar twin stretch (~**87%** median); fair specialize ~**76%**; S1 noisy;
F.vigenere/beaufort ≥90% under 1 B/rune. Atbash/totient Done uses the
**compute roof** (**2.0 TB**); DRAM occupancy (~**80.7 TB**, ncu 0.01111 B/rune)
is diary only ([`profiles/traffic_model/SUMMARY.md`](profiles/traffic_model/SUMMARY.md)).
Affine uses the **Affine shared-cipher** DRAM class (~**0.01906 B/rune** →
≈**47.0 TB**). Hard-S0 interpreter remains ~**7–8%**.

`ThroughputTiers` / `DslPeakSanity` delegate to `BenchTierSpec`
(Catch2 `[bench][spec]` / `[dsl][peak]`).

**Theory search export** (`family=theory`) uses the **same** physical roof as
catalog fused hist (same traffic class).

### Path (transpiler → search)

```text
TheoryIr decrypt HotLoop
  → Z29ExprNormalize + TheoryShapeMatch   (target; see dsl-smart-hist.md)
  → TheoryHistChi2Emit (shape twin / S1 / S2 / else S0)
  → TheoryExportCache HistPlan
  → GpuCandidateExport prefers specialized launch; soft-fallback S0 bytecode
  → export_backend=cuda
```

Smart customs (self-written math without catalog API): [`dsl-smart-hist.md`](dsl-smart-hist.md).

| Strategy | Spec id | Kernel | Physical peak (DRAM roof) |
|----------|---------|--------|---------------------------|
| Fair Caesar (specialize S1 when eligible; else S0) | `T.theory.caesar_bytecode` | S1 lut / `theory_chi2_hist_kernel` | **896B** |
| S1 LUT-29 (`f(x)`-only) | `T.theory.s1_lut29` | `theory_hist_chi2_s1_lut_kernel` | **896B** |
| S2 linear uchar4 | `T.theory.s2_linear` / `T.theory.progressive` | `theory_hist_chi2_s2_linear_kernel` | **896B** |

Fair gate: `parcae-bench --suite theory --allow-cuda` → `BenchTierSpec::pass_tier`
(≥90% of **896B** + `slo_min` at `T≥2^20`). Soft-fallback / hard-S0 interpreter
still fails that gate (~7–8%); eligible Caesar fair row is stretch (~79%) via
specialize dispatch. Short T is measurement-only (`underfill_not_slo_gate`).

### Kernel SLO vs campaign wall

| Metric | Tool / log | Includes | Gate? |
|--------|------------|----------|-------|
| **Kernel SLO** | `parcae-bench --suite theory` (cudaEvent) | Device hist/finalize only | **Yes** — ≥90% DRAM roof |
| **Campaign wall** | `parcae-search-cycle` → `research/run.log` | Host prepare/bind, H2D, kernel, D2H, materialize, ingest | **No** — ops / ETA only |

Do **not** compare campaign wall at short page `T` to catalog or theory cudaEvent
peaks at `T≈2^20`. Normative Done rules:
[`theory-hist-transpile.md`](theory-hist-transpile.md). Smart ShapeId path:
[`dsl-smart-hist.md`](dsl-smart-hist.md). Playbook + progress log:
[`cuda-profile-theory.md`](cuda-profile-theory.md).

### Theory plateaus (physical roof)

| Tier | Workload | Ceiling (runes/s) | SLO floor | Notes |
|------|----------|-------------------|-----------|-------|
| T.theory.caesar_bytecode | specialize S1 when eligible | **896B** | ≥15B | ~79% fair (`specialize_S1`); hard-S0 ~7–8% |
| T.theory.s1_lut29 | S1 LUT-29 | **896B** | ≥15B | same roof; ~79% fair (fat-64) |
| T.theory.s2_linear | S2 progressive / bitmask | **896B** | ≥15B | ~44% fair @ C=841 (ks29) |

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
| F.atbash | Atbash fused χ² (C=512 occupancy pad) | **2.0 TB** (compute roof; DRAM diary ≈80.7 TB) | ≥15B |
| F.affine | Affine fused χ² (812 shared cipher) | **≈47.0 TB** (0.01906 B/rune ncu) | ≥15B |
| F.vigenere | Vigenère fused χ² (key len 8) | **896B** | ≥3B |
| F.beaufort | Beaufort fused χ² (key len 8) | **896B** | ≥3B |
| F.totient | Totient stream fused χ² (C=512) | **2.0 TB** (same compute roof as Atbash) | ≥3B |

### Compose

| Tier | Workload | Ceiling (runes/s) | SLO floor |
|------|----------|-------------------|-----------|
| C.koan1_fused | Atbash→Caesar+shift fused χ² | **896B** | ≥15B |
| C.koan1_stages | Atbash (async) + Caesar χ² | **896B** | ≥15B |

## Recalibration rule

1. Unique-key peak = **physical DRAM roofline**, not measured max. Derive from
   `device_peak_dram_bytes_per_s / bytes_per_rune` (RTX 5070 Ti → 896e9 /
   448e9). Shared-cipher Atbash/totient peak = **compute roof**
   (`kSharedCipherComputeRoofRps`); recalibrate only from identity occupancy
   hist (`HistOccupancyRoof`), never from Atbash quiet max. Do **not** raise
   or lower Spec to chase production quiet-run medians.
2. Run `parcae-bench --suite slo --extended --allow-cuda` (or compat
   `parcae-throughput-tiers`) on a quiet GPU to measure **progress toward**
   that roof (`%peak`), not to redefine it.
3. `%peak` **must stay ≤100**. If a quiet run prints **&gt;100**, the Spec
   peak or BW assumption is wrong (wrong bytes/rune, wrong DRAM GB/s, or
   compute roof too low) — fix the roof model, not “raise to absorb” a
   measured outlier.
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
- DSL smart hist (customs without presets): [dsl-smart-hist.md](dsl-smart-hist.md)
- Operator handbook (theory URI + dispatch): [search-handbook.md](search-handbook.md)
- Transpiler + HistChi2 strategies: [python-transpiler.md](python-transpiler.md)