# Roof hist — fat-tile + S2 period-29 keystream (2026-10-03, RTX 5070 Ti)

Isolate **tiling** vs hist primitive: keep `HistFast::add_private`, fatten
`grid.y` so each block does more grid-stride work. Then cut S2 per-rune mul/add
via shared period-29 keystream.

Captured with [`scripts/cuda/capture_roof_hist.ps1`](../../../scripts/cuda/capture_roof_hist.ps1)
and `parcae-bench --suite theory`.

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Protocol:** `BenchTimer` cudaEvent (4 warmups + median-of-3), fair `T=1048576`  
**Peak:** physical DRAM roof **896B** runes/s (@ 1 B cipher/rune model)

## Fat-tile sweep (Caesar twin)

| Tile cap (`grid.y`) | runes/s | % of 896B | Notes |
|---------------------|---------|-----------|-------|
| **1024** (legacy) | **502.8B** | **56.1%** | Uncapped work tiles @ T=1M |
| 256 | 724.2B | 80.8% | Stretch (≥80%) |
| 128 | 802.8B | 89.6% | Near Done |
| **64** | **837.3B** | **93.5%** | **Best — ≥90% Done** |
| 32 | 743.6B | 83.0% | Past peak |
| 16 | 698.0B | 77.9% | — |
| 8 | 691.5B | 77.2% | — |

Scores identical across caps. **Shipped:** `HistFast::production_tile_cap = 64`.

## Fair theory after fat-64 wire

Source: `theory_fair_fat64_wired.json` (S1/S2/F.* on production clamp):

| Row | runes/s | % of 896B |
|-----|---------|-----------|
| `T.theory.compare_caesar` | **838.1B** | **93.5%** Done |
| `T.theory.s1_lut29` | **690.2B** | **77.0%** stretch |
| `T.theory.progressive` (S2 @841) | **342.9B** | **38.3%** |
| `T.theory.caesar_bytecode` (S0) | 61.3B | 6.8% |

SLO extended (`slo_extended_fat64.json`): T1 **746B**, F.atbash **1525B**,
F.affine **838B**, F.vigenere **883B**.

## S2 period-29 keystream precompute

`theory_hist_chi2_s2_linear_kernel`: per block fill
`__shared__ uint8_t ks[29]` with `ks[i] = b0 + b1·i (mod 29)`, then hot loop
indexes `ks[t % 29]` instead of mul/add per rune.

**Post-ks fair** (`theory_fair_s2_ks.json`):

| Row | runes/s | % of 896B | vs fat-64 wire |
|-----|---------|-----------|----------------|
| `T.theory.progressive` (S2 @841) | **403.1B** | **45.0%** | **~1.18×** (was 343B) |
| `T.theory.compare_caesar` | 835.6B | 93.3% | unchanged |
| `T.theory.s1_lut29` | 689.2B | 76.9% | unchanged |
| `T.theory.caesar_bytecode` (S0) | 68.3B | 7.6% | unchanged |

## Specialize dispatch (fair Caesar bytecode → S1)

`BenchTheorySuite::run_caesar_bytecode` calls `TheoryHistChi2Emit::emit_decrypt_hist`
and, when emitted strategy is S1 LUT-29, launches the S1 kernel (`detail=specialize_S1`)
instead of the interpreter. Hard-S0 shapes (autokey / prefer_branch / caps) stay on
S0. Export already preferred S1/S2 the same way.

**Post-specialize fair** (`theory_fair_specialize.json`):

| Row | runes/s | % of 896B | Notes |
|-----|---------|-----------|-------|
| `T.theory.caesar_bytecode` | **707.6B** | **79.0%** | `specialize_S1` — was ~61–68B interpreter |
| `T.theory.s1_lut29` | **706.0B** | **78.8%** | twin of fair specialize path |
| `T.theory.compare_caesar` | **829.4B** | **92.6%** | Done |
| `T.theory.progressive` (S2 @841) | **396.4B** | **44.2%** | ks29; still compute-bound |

## What failed previously (do not reopen)

| Attempt | Result |
|---------|--------|
| 32 KiB thread-local hist | ~49B — abandoned |
| Register-local + fat≤32 | ~19–35B — abandoned |
| `__match_any_sync` | ~114B — LOSE vs warp-private |

## Artifacts

| File | Role |
|------|------|
| `tile_cap_sweep.txt` | Catch2 sweep stdout |
| `theory_fair.json` | Pre fat-tile fair suite |
| `theory_fair_post_ship.json` | Caesar-only fat-64 |
| `theory_fair_fat64_wired.json` | Fat-64 on S1/S2/F.* |
| `theory_fair_s2_ks.json` | After S2 shared keystream |
| `theory_fair_specialize.json` | Fair suite after S0→S1 specialize dispatch |
| `theory_fair_specialize.txt` | Human-readable fair stdout |
| `slo_extended_fat64.json` | SLO T1 + F.* |
| `capture_log.txt` | Script transcript |
