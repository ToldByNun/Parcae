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

## S2 running residue (2026-10-07)

Hot loop no longer does `ks[t % 29]` per rune: each thread keeps a running
residue `r = (i·4) mod 29`, advances by `(4·stride) mod 29` with
add+conditional-subtract. Same pattern mirrored in S5. `HistTileCap` S2
production default **32** (sweep beat 64/128; Caesar fat-64 untouched).

Quiet theory ×3: S2 median **~683B** (~**76%** of 896B; best **764B**). Stretch
716.8B not median-stable — remaining bound is hist atomics / L2, not `%29`.

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

## Shared-cipher atomic climb (2026-10-07)

Atbash / totient / shape Atbash: Commit-7-class `__restrict__`/`__ldg` (no
`add_private` change). `HistTileCap::kAtbash` + `kTotient` wired; Caesar slot
untouched. A/B @ C=512 T=1M caps **32/64/128**:

| Shape | Production default | Plate note |
|-------|--------------------|------------|
| Atbash catalog + shape twin | **64** | 32↔64 within noise; prod ~1.5–1.75 TB (~76–87% of 2.0 TB) |
| Totient | **32** | cap32 wins vs 64/128; prod ~1.17–1.37 TB |

Catch2 `[cuda][hist][atomic_climb]`. Twin scores ≡. Pass: absolute climb +
model `%peak≤100` on production; Done 1.8 TB median open on busy desktop.

## Caesar restrict + ldg (2026-10-07)

Micro-opts around hist traffic only (`add_private` untouched;
`HistFast::production_tile_cap = 64` unchanged):

- `caesar_chi2_histogram_decrypt_kernel` / shape Caesar twin:
  `__restrict__` on pointers, `__ldg` on shifts + `uchar4` / tail loads
- `chi2_finalize_kernel`: `__restrict__` + `__ldg` on probabilities
- `BenchDslSmartSuite`: Caesar custom/compare **first** (before Atbash C=512)
  so the 896B Done plate is not heated by shared-cipher work

Catch2 `[cuda][hist][roof][caesar][done]`: 5-sample catalog then shape block;
prints `CAESAR_ROOF_DONE`. Hard CI = catalog stretch (≥80%); Done median is
quiet-plate ACCEPTANCE below.

**Plate (desktop util ~26–30%, Roblox/Discord/etc.):** Done plate ×7 → median of
catalog medians **~786B** (~87.7%), shape **~790B** (~88.2%); median of bests
**~836B / ~843B** (both ≥806.4B). One settled plate cleared Done on both
medians (~832B / ~823B). Prior quieter twin median **~827B** Done.

`parcae-bench --suite dsl_smart` ×5 (Caesar-first): custom median **~814.5B**
(**Done**); compare median noisy (~732B) with best-of **~826B** Done.

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
