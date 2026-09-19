# CUDA fused throughput reference

**Status:** local reference plateaus (not a CI gate)  
**Tool:** `parcae-throughput-tiers`  
**Hardware used for the table below:** NVIDIA GeForce RTX 5070 Ti (~896 GB/s DRAM)  
**Metric:** `repeats × C × T / median-of-3 cudaEvent` (setup excluded)

These numbers are **practical ceilings** for the fused decrypt+χ² (and deep-score)
paths after HistFast / vec4 / tiles_for work. They are not marketing FLOPS and not
portable across GPUs — re-run the tool on your card and recalibrate
`ThroughputTiers::estimated_peak` if `%peak` goes above 100.

## How to measure

```bash
cmake -S . -B build-cuda -DPARCAE_BUILD_CUDA=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build-cuda --config Release --target parcae-throughput-tiers
./build-cuda/tools/Release/parcae-throughput-tiers   # adjust path on Windows
```

Pass rule: SLO floor **and** ≥ 90% of the practical ceiling for that row.

## Reference plateaus (RTX 5070 Ti)

Ceilings match `ThroughputTiers::estimated_peak`. Typical healthy runs sit around
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

1. Run `parcae-throughput-tiers` several times on a quiet GPU.
2. For each row, take the **max** of the reported medians.
3. Set `estimated_peak` slightly above that max (round up).
4. Confirm subsequent runs print `%peak` in roughly **90–99**, not above 100 —
   if they do, the stored ceiling is stale (too low), not “super-linear hardware”.
5. If mins fall under 90% while maxes stay under 100%, widen the timed window
   (more reps) before lowering the ceiling — variance usually means clocks, not
   a faster kernel.
## Related

- Implementation: [`include/parcae/run/throughput_tiers.hpp`](../../include/parcae/run/throughput_tiers.hpp)
- Kernels: [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (`hist_fast.hpp`, `*_chi2_batch.cu`)
- Build notes: [cuda-build.md](cuda-build.md)
