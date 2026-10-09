# CUDA fused throughput reference

**Status:** local reference plateaus (not a CI gate)  
**Canonical constants:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp)  
**Canonical tool:** `parcae-bench --suite slo [--extended] --allow-cuda`  
**Compat tool:** `parcae-throughput-tiers` (thin wrapper → same `BenchSloSuite` with extended on)  
**Hardware:** NVIDIA GeForce RTX 5070 Ti — **896 GB/s** GDDR7 (published)  
**Metric (dual):**
- **logical** `runes/s` = `repeats × C × T / median-of-3 cudaEvent` (setup excluded) —
  historical rate; **inflated** under remap (cipher streamed ~once).
- **physical** `cipher_B/s` = `repeats × T × bytes_per_token / elapsed`
  (`BenchMetric::cipher_bytes_per_sec`) — compare to GDDR7 **896 GB/s**.

## What `estimated_peak` means (Done gate, not always DRAM)

`BenchTierSpec::estimated_peak` is the shape ceiling used by Done/Stretch on
**logical** runes/s:

```text
# Alphabet / column / lag / bigram remap (production once-count + bin remap)
peak_runes/s = kAlphabetRemapHistRoofRps | kColumnRemapHistRoofRps |
               kLagRemapHistRoofRps | kBigramRemapHistRoofRps
             ≈ 2.0e12 (alphabet/column/lag interim) / 1.0e12 (bigram)
# Not 896B — cipher traffic is O(T), not O(C·T).

# Shared-cipher occupancy (F.atbash / F.totient / dsl_smart Atbash)
peak_runes/s = kSharedCipherComputeRoofRps   # SM/atomic capacity
             = 2.0e12                         # frozen 2026-10-06 identity hist

# Legacy unique-key decode→hist (hard-S0 / koan stages)
peak_runes/s = DRAM_BW / 1 B/rune = 896B
```

Remap roofs are interim freezes from the identity-hist / CipherHistOnce plate
class — re-calibrate on a quiet remap plate when available. DRAM occupancy
diaries (Atbash ~80.7 TB, Affine decode ~47 TB) are **not** production Done.
`%peak = measured / peak` is ≤ **100%** by construction. If a quiet run ever
prints &gt;100%, the roof model is wrong — fix the model.

Quiet ACCEPTANCE ([`profiles/kernel_slo/SUMMARY.md`](profiles/kernel_slo/SUMMARY.md)):
historical 896B %-of-peak numbers for Caesar/S1/S2 are **superseded** by Remap
roofs after production once-count+remap. Atbash/totient Done stays the
**compute roof** (**2.0 TB**). Hard-S0 interpreter remains ~**7–8%** of 896B.

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

| Strategy | Spec id | Kernel | Done peak (logical) |
|----------|---------|--------|---------------------|
| Fair Caesar (specialize S1 when eligible; else S0) | `T.theory.caesar_bytecode` | S1 mono-LUT remap | **alphabet Remap 2.0 TB** |
| S1 LUT-29 (`f(x)`-only) | `T.theory.s1_lut29` | CipherHistOnce + LUT remap | **alphabet Remap 2.0 TB** |
| S2 linear | `T.theory.s2_linear` / `T.theory.progressive` | ColumnHistOnce(L=29) + remap | **column Remap 2.0 TB** |

Fair gate: `parcae-bench --suite theory --allow-cuda` → `BenchTierSpec::pass_tier`
(≥90% of Remap roof + `slo_min` at `T≥2^20`). Soft-fallback / hard-S0 interpreter
still fails that gate (~7–8% of **896B** diary); eligible Caesar/S1/S2 fair rows
use Remap roofs. Short T is measurement-only (`underfill_not_slo_gate`).

### Kernel SLO vs campaign wall

| Metric | Tool / log | Includes | Gate? |
|--------|------------|----------|-------|
| **Kernel SLO** | `parcae-bench --suite theory` (cudaEvent) | Device hist/finalize only | **Yes** — ≥90% Remap / DRAM roof |
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
| T.theory.s1_lut29 | S1 LUT-29 | **896B** | ≥15B | fair hist+finalize (LUT baked outside timer); ~75–80% |
| T.theory.s2_linear | S2 progressive / bitmask | **896B** | ≥15B | ~76% fair @ C=841 (running residue + tile32) |

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