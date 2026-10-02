# CUDA fused throughput reference

**Status:** local reference plateaus (not a CI gate)  
**Canonical constants:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp)  
**Canonical tool:** `parcae-bench --suite slo [--extended] --allow-cuda`  
**Compat tool:** `parcae-throughput-tiers` (thin wrapper → same `BenchSloSuite` with extended on)  
**Hardware used for the table below:** NVIDIA GeForce RTX 5070 Ti (~896 GB/s DRAM)  
**Metric:** `repeats × C × T / median-of-3 cudaEvent` (setup excluded)

These numbers are **practical ceilings** for the fused decrypt+χ² (and deep-score)
paths after HistFast / vec4 / tiles_for work. They are not marketing FLOPS and not
portable across GPUs — re-run the tool on your card and recalibrate
`BenchTierSpec` peaks if `%peak` goes above 100. `ThroughputTiers` and
`DslPeakSanity` delegate to that header (Catch2 `[bench][spec]` / `[dsl][peak]`).

**Theory search export** (`family=theory`) is a **separate** fused-χ² path from
catalog T1/F.* rows — same `BenchTimer` metric, different kernels and Spec ids.

### Path (transpiler → search)

```text
TheoryIr decrypt HotLoop
  → TheoryHistChi2Emit (S1 LUT-29 / S2 linear uchar4 / else S0)
  → TheoryExportCache HistPlan
  → GpuCandidateExport prefers specialized launch; soft-fallback S0 bytecode
  → export_backend=cuda
```

| Strategy | Spec id | Kernel | Provisional peak (5070 Ti) |
|----------|---------|--------|----------------------------|
| S0 bytecode interpreter | `T.theory.caesar_bytecode` | `theory_chi2_hist_kernel` | 75B |
| S1 LUT-29 (`f(x)`-only) | `T.theory.s1_lut29` | `theory_hist_chi2_s1_lut_kernel` | 473B |
| S2 linear uchar4 | `T.theory.s2_linear` / `T.theory.progressive` | `theory_hist_chi2_s2_linear_kernel` | 350B |

Fair gate: `parcae-bench --suite theory --allow-cuda` → `BenchTierSpec::pass_tier`
(≥90% peak + `slo_min` at `T≥2^20`). Short T is measurement-only
(`underfill_not_slo_gate`).

### Kernel SLO vs campaign wall

| Metric | Tool / log | Includes | Gate? |
|--------|------------|----------|-------|
| **Kernel SLO** | `parcae-bench --suite theory` (cudaEvent) | Device hist/finalize only | **Yes** — ≥90% shape peak |
| **Campaign wall** | `parcae-search-cycle` → `research/run.log` | Host prepare/bind, H2D, kernel, D2H, materialize, ingest | **No** — ops / ETA only |

Do **not** compare campaign wall at short page `T` to catalog or theory cudaEvent
peaks at `T≈2^20`. Playbook + progress log:
[`cuda-profile-theory.md`](cuda-profile-theory.md). Post-emit snapshot:
[`profiles/specialized/SUMMARY.md`](profiles/specialized/SUMMARY.md).

### Theory plateaus (provisional)

| Tier | Workload | Ceiling (runes/s) | SLO floor | Notes |
|------|----------|-------------------|-----------|-------|
| T.theory.caesar_bytecode | S0 interpreter (Caesar-as-bytecode) | 75B | ≥15B | Soft-fallback for unmatched shapes |
| T.theory.s1_lut29 | S1 LUT-29 | 473B | ≥15B | Affine-class until remesaure |
| T.theory.s2_linear | S2 progressive / bitmask linear | 350B | ≥15B | Plan provisional; suite often uses C=9 |

Operator handbook: [`search-handbook.md`](search-handbook.md) § Theory URI.
Emit API: [`theory_hist_chi2_emit.hpp`](../../include/parcae/dsl/theory_hist_chi2_emit.hpp).

## How to measure

```bash
cmake -S . -B build-cuda -DPARCAE_BUILD_CUDA=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build-cuda --config Release --target parcae-bench parcae-throughput-tiers

# Preferred
./build-cuda/tools/Release/parcae-bench --suite slo --extended --allow-cuda

# Compat (same suite; no --allow-cuda flag)
./build-cuda/tools/Release/parcae-throughput-tiers
```

Pass rule: SLO floor **and** ≥ 90% of the practical ceiling for that row
(`BenchTierSpec::pass_tier`).

## Canonical SLO configs (`BenchTierSpec`)

| Tier | Workload | C | T | reps | Target (runes/s) | Ceiling |
|------|----------|---|---|------|------------------|---------|
| T1 | Caesar fused χ² | 29 | 1048576 | 64 | 15.00B–35.00B | 392B |
| T2 | multi-key / autokey / dynamic-shift (worst) | 4096 | 262144 | 8 | 3.00B–10.00B | 402B |
| T3 | Caesar bigram + dict | 512 | 262144 | 8 | ≥1.00B | 55B |

Display bands (`slo_max`) are expectations only — faster than the upper bound still
**passes**. Failures are below the floor or below the 90% peak band.

## Reference plateaus (RTX 5070 Ti)

Ceilings match `BenchTierSpec::estimated_peak`. Typical healthy runs sit around
90–99% of these values.

### SLO tiers

| Tier | Workload | Ceiling (runes/s) | SLO floor |
|------|----------|-------------------|-----------|
| T1 | Caesar fused χ² | 392B | ≥15B |
| T2 | multi-key / autokey / dynamic-shift (worst) | 402B | ≥3B |
| T3 | Caesar bigram + dict | 55B | ≥1B |

### Transform families

| Tier | Workload | Ceiling (runes/s) | SLO floor |
|------|----------|-------------------|-----------|
| F.atbash | Atbash fused χ² | 550B | ≥15B |
| F.affine | Affine fused χ² (812) | 473B | ≥15B |
| F.vigenere | Vigenère fused χ² (key len 8) | 398B | ≥3B |
| F.beaufort | Beaufort fused χ² (key len 8) | 402B | ≥3B |
| F.totient | Totient stream fused χ² | 460B | ≥3B |

### Compose

| Tier | Workload | Ceiling (runes/s) | SLO floor |
|------|----------|-------------------|-----------|
| C.koan1_fused | Atbash→Caesar+shift fused χ² | 372B | ≥15B |
| C.koan1_stages | Atbash (async) + Caesar χ² | 322B | ≥15B |

## Recalibration rule

1. Run `parcae-bench --suite slo --extended --allow-cuda` (or compat
   `parcae-throughput-tiers`) several times on a quiet GPU.
2. For each row, take the **max** of the reported medians.
3. Bump `BenchTierSpec` peaks slightly above that max (round up) — runners and
   `DslPeakSanity` pick it up automatically.
4. Confirm subsequent runs print `%peak` in roughly **90–99**, not above 100 —
   if they do, the stored ceiling is stale (too low), not “super-linear hardware”.
5. If mins fall under 90% while maxes stay under 100%, widen the timed window
   (more reps) before lowering the ceiling — variance usually means clocks, not
   a faster kernel.

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
- Operator handbook (theory URI + dispatch): [search-handbook.md](search-handbook.md)
- Transpiler + HistChi2 strategies: [python-transpiler.md](python-transpiler.md)