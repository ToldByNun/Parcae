# Theory CUDA profiling playbook (nsys / ncu)

**Status:** operator playbook for theory fused-χ² / search-export throughput  
**Hardware reference:** RTX 5070 Ti (sm_120) — same plate as [`cuda-throughput.md`](cuda-throughput.md)  
**Canonical catalog metric:** `BenchTimer` — 4 warmups + **median-of-3** `cudaEvent`, setup excluded  
**Theory acceptance (PRIMARY):** measured ≥ **90%** of the theory-shape
`estimated_peak`. Unique-key shapes use the physical DRAM roofline (**896B** @
1 B cipher/rune on RTX 5070 Ti). Shared-cipher Atbash/totient use the **compute
roof** (**2.0 TB**). Soft-fallback to S0 does not waive the S0 peak gate.
Normative contract: [`theory-hist-transpile.md`](theory-hist-transpile.md).
Smart customs (name-irrelevant ShapeId / stretch 80% / Done 90%):
[`dsl-smart-hist.md`](dsl-smart-hist.md).

**Wrapper:** [`scripts/cuda/profile_theory_hist.ps1`](../../scripts/cuda/profile_theory_hist.ps1)

Use this doc before and after every theory-throughput change. Do **not** compare
campaign wall rates at short `T` to catalog `cudaEvent` peaks at `T≈2^20`.

---

## Two metrics (do not mix)

| Name | Formula | What it includes | Use for |
|------|---------|------------------|---------|
| **Kernel SLO** | `repeats × C × T / median cudaEvent` (or ncu duration → `C×T / kernel_s`) | Device hist/finalize only; setup / H2D / D2H / host materialize **excluded** | Pass/fail vs `estimated_peak` × 0.90 |
| **Campaign wall** | `C × T / wall` over `SearchScheduler::run_loop` (see `research/run.log`) | Host prepare/bind, H2D, kernel, D2H, materialize, ingest, Python spawn | Ops / ETA only — **not** the 90% gate |

Catalog families use Kernel SLO via `parcae-bench --suite slo` and
[`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp). Theory Kernel
SLO uses `parcae-bench --suite theory --allow-cuda` (`T.theory.*` Spec rows +
≥90% peak gate at fair T). Hand-written customs vs catalog twins use
`parcae-bench --suite dsl_smart --allow-cuda` → [`profiles/dsl_smart/`](profiles/dsl_smart/).
Campaign wall stays in `research/run.log`.

Optional interim checkpoint **≥50B runes/s** (Kernel SLO) is allowed only when
`estimated_peak ≫ 50B`. It never replaces the 90% peak gate.

---

## Prerequisites

1. CUDA Toolkit with **Nsight Systems** (`nsys`) and **Nsight Compute** (`ncu`) on `PATH`.
2. Release CUDA build:

   ```powershell
   cmake -S . -B build-cuda -DPARCAE_BUILD_CUDA=ON -DPARCAE_BUILD_TOOLS=ON -DPARCAE_BUILD_TESTS=ON
   cmake --build build-cuda --config Release --target parcae-search-cycle parcae-bench
   ```

3. Quiet GPU (close other CUDA apps). Same card for baseline and after-change runs.
4. Output directory (gitignored): `docs/architecture/profiles/` (preferred) or a
   workspace `research/profiles/` folder.

---

## Fair grids (always run both regimes)

| Regime | Typical `(C, T)` | Purpose |
|--------|------------------|---------|
| **Campaign-like** | `(16384, 262)` or job’s real page length | Explains wall / occupancy underfill; matches LP2 chunks |
| **Kernel-fair** | `(812 or 16384, 1048576)` — `T≥2^20` | Apples-to-apples vs catalog; 90% peak gate |

Side-by-side catalog kernels at the **same** `(C,T)`:

| Kernel (mangled name may vary; filter prefix) | Twin |
|-----------------------------------------------|------|
| `theory_chi2_hist_kernel` | Theory S0 bytecode hist ([`theory_chi2_batch.cu`](../../Parcae/Parcae/cuda/theory_chi2_batch.cu)) |
| `theory_hist_chi2_s1_lut_kernel` | Theory S1 LUT-29 ([`theory_hist_chi2_s1.cu`](../../Parcae/Parcae/cuda/theory_hist_chi2_s1.cu)) |
| `theory_hist_chi2_s2_linear_kernel` | Theory S2 linear ([`theory_hist_chi2_s2.cu`](../../Parcae/Parcae/cuda/theory_hist_chi2_s2.cu)) |
| `theory_hist_chi2_shape_atbash_kernel` | Theory ShapeInline Atbash ([`theory_hist_chi2_shape.cu`](../../Parcae/Parcae/cuda/theory_hist_chi2_shape.cu)) |
| `theory_hist_chi2_shape_caesar_kernel` | Theory ShapeInline Caesar (`HistFast::dec_caesar`) |
| `theory_hist_chi2_shape_affine_kernel` | Theory ShapeInline Affine decrypt (`inv(a)·(x−b)`) |
| `caesar` / `atbash_caesar_chi2_hist_kernel` | T1 / compose path |
| `affine_chi2_hist_kernel` | F.affine |
| `atbash_chi2_hist_kernel` | F.atbash |

Post–specialized capture (vs commit-4 baseline):
[`scripts/cuda/capture_theory_specialized.ps1`](../../scripts/cuda/capture_theory_specialized.ps1)
→ [`profiles/specialized/`](profiles/specialized/).

---

## 1. Nsight Compute (kernel)

### Goals

- Duration → **runes/s** = `C × T / (duration_ns × 1e-9)` (one launch; multiply by
  `repeats` only if you time a multi-launch window the way `BenchTimer` does).
- SM / DRAM % of peak, occupancy, top stall reasons.
- Compare theory vs Caesar/Affine at identical `(C,T)`.

### Metrics (minimum)

| Metric / view | Why |
|---------------|-----|
| Duration | Derive runes/s |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute headroom |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Memory bound? |
| `dram__bytes.sum` (+ read/write if available) | Absolute DRAM GB/s and bytes/rune |
| Issue-slot utilization / warp stall reasons | `stall_inst_fetch`, `stall_exec_dependency`, `stall_memory_throttle` (often **n/a on sm_120**) |
| L2 hit rate / L1–L2 sectors on global loads | `lts__t_sector_hit_rate.pct`, L1TEX global-load sectors — L2 vs DRAM |
| Achieved occupancy; grid vs active warps | Underfill at short `T` |

Canonical metric lists live in
[`profile_theory_hist.ps1`](../../scripts/cuda/profile_theory_hist.ps1):

- `-MetricsPreset baseline` — SM/DRAM % + stalls + duration (legacy captures)
- `-MetricsPreset cache_bound` — baseline + `dram__bytes.sum` + L1/L2 counters
- `-ExportCsv` — write `<Tag>_metrics.csv` from the `.ncu-rep`

L2/DRAM absolute plate (Caesar / Atbash / S1 / S2 / Affine):
[`scripts/cuda/capture_cache_bound.ps1`](../../scripts/cuda/capture_cache_bound.ps1)
→ [`profiles/cache_bound/`](profiles/cache_bound/).

### Commands

Prefer the wrapper (writes under `-OutDir`):

```powershell
.\scripts\cuda\profile_theory_hist.ps1 -Mode ncu `
  -Exe .\build-cuda\tools\Release\parcae-search-cycle.exe `
  -ExeArgs '--workspace','lp2-bitmask-blend-allpages','--job','path\to\job.json','--allow-theory-uri','--backend','cuda','--allow-cuda','--data-dir','data' `
  -KernelFilter 'theory_chi2_hist_kernel' `
  -OutDir .\docs\architecture\profiles\baseline `
  -Tag theory_hist
```

Raw `ncu` (same idea):

```powershell
ncu --set full --kernel-name-base demangled `
  --kernel-name regex:theory_chi2_hist_kernel `
  --export docs/architecture/profiles/baseline/theory_hist `
  --force-overwrite `
  -- <exe> <args...>
```

Catalog compare (repeat with `--kernel-name regex:affine_chi2_hist_kernel` etc.,
same workload launcher when available). Preferred Kernel SLO launcher:

```powershell
.\build-cuda\tools\Release\parcae-bench.exe --suite theory --allow-cuda --data-dir data
# short smoke:
.\build-cuda\tools\Release\parcae-bench.exe --suite theory --allow-cuda --tokens 65536 --repeats 2 --data-dir data
# campaign-like underfill row:
.\build-cuda\tools\Release\parcae-bench.exe --suite theory --allow-cuda --campaign-grid --data-dir data
```

Also usable under ncu/nsys via `profile_theory_hist.ps1 -Exe …\parcae-bench.exe -ExeArgs …`.
`parcae-bench --suite slo --extended --allow-cuda` remains the catalog Kernel SLO path.
Customs without presets (fair T, vs F.atbash / Caesar twin):

```powershell
.\build-cuda\tools\Release\parcae-bench.exe --suite dsl_smart --allow-cuda --data-dir data
.\scripts\cuda\capture_dsl_smart.ps1 -BuildDir build-cuda -Runs 3
```

---

## 2. Nsight Systems (timeline: host vs device)

### Goals

- Fraction of wall that is GPU kernel vs memcpy vs CPU gaps.
- H2D traffic on program buffers (`ops`, `imm`, `slots`) vs cipher/probs.
  After host-amortize, `ops`/`imm` should upload **once** per theory URI when a
  shared `TheoryExportCache` is used; only `slots` re-upload per params chunk.
- Cold process spawn vs warm reuse (campaign Python chunks).

### Command

```powershell
.\scripts\cuda\profile_theory_hist.ps1 -Mode nsys `
  -Exe .\build-cuda\tools\Release\parcae-search-cycle.exe `
  -ExeArgs '...' `
  -OutDir .\docs\architecture\profiles\baseline `
  -Tag theory_export_chunk
```

Raw:

```powershell
nsys profile -t cuda,nvtx --stats=true --force-overwrite=true `
  -o docs/architecture/profiles/baseline/theory_export_chunk `
  -- <exe> <args...>
```

(Windows Nsight Systems: use `cuda,nvtx` — `osrt` is not a valid `--trace` value.)

### What to record from the report

| Field | Notes |
|-------|-------|
| GPU active % | Low at `T=262` is expected |
| `cudaMemcpy` H2D/D2H count & bytes | Program re-upload every chunk? |
| Long CPU gaps | prepare / bind_slots / materialize / hyp write |
| Kernel span names | `theory_chi2_hist_kernel`, finalize/patch |
| NVTX ranges | Present on `GpuCandidateExport` / scheduler; microbench may omit export stages |

**NVTX:** ranges such as `prepare_theory`, `bind_slots`, `h2d`, `hist_kernel`,
`finalize`, `d2h`, `materialize`, `ingest` are pushed via `NvtxRange`
([`include/parcae/search/nvtx_range.hpp`](../../include/parcae/search/nvtx_range.hpp))
on the theory fused export / search-cycle path. Requires Toolkit
`<nvtx3/nvToolsExt.h>` at compile time (`NvtxRange::available()`). If the header
was missing, note “no NVTX” in the snapshot row and rely on CUDA API trace only.

**ncu permission:** if ncu prints `ERR_NVGPUCTRPERM`, enable GPU performance
counters for the user (or elevate) before kernel-counter baselines; cudaEvent +
nsys remain valid without that permission.

---

## Export duty cycle

Fair Kernel SLO (cudaEvent, H2D excluded) and export-path wall (host / PCIe /
sync) are different metrics. Capture both with
[`scripts/cuda/capture_export_duty.ps1`](../../scripts/cuda/capture_export_duty.ps1)
→ [`profiles/export_duty/`](profiles/export_duty/) (fair JSON + nsys stress at
`T=65536`, `C=64`, 8 chunks). Write-up:
[`profiles/export_duty/SUMMARY.md`](profiles/export_duty/SUMMARY.md).

Diagnosis: high fair Kernel SLO + low hist wall share ⇒ idle GPU from
host/PCIe/sync — **idle GPU ≠ Kernel SLO PCIe**.

**Track A ACCEPTANCE (2026-10-03):** post-residency stress cut non-kernel NVTX
wall **~53×** vs [`pre_residency/`](profiles/export_duty/pre_residency/) (PASS).

**Metric B ACCEPTANCE (2026-10-03):** quiet fair theory ×3 + slo extended ×2 →
[`profiles/kernel_slo/SUMMARY.md`](profiles/kernel_slo/SUMMARY.md) — **PARTIAL PASS**
(Caesar twin stretch; S1 under 80%; Atbash/totient traffic diary ~**80.7 TB**;
Done gate = compute roof **2.0 TB**; `%peak≤100`).

**DSL smart E2E (2026-10-06):** fair customs Atbash/Caesar/Affine/linear/autokey +
twins → [`profiles/dsl_smart/SUMMARY.md`](profiles/dsl_smart/SUMMARY.md) — **PARTIAL PASS**
(Caesar stretch ~87%; Atbash model OK; linear/autokey twin-class; Affine model open).
Edge tags green. Alloc pool **skipped** (warm `cudaMalloc` ~0.2% API). Fair S1 stayed
≥ ~400B class (**482B**).

---

## 3. Progress rule (every throughput milestone)

After each meaningful change:

1. Save **ncu** report (`.ncu-rep` or exported CSV/text) under
   `docs/architecture/profiles/<tag>/` (when counters are permitted).
2. Save **nsys** report (`.nsys-rep`) + one-line stats summary.
3. Append a row to the progress table below (Ist Kernel SLO vs 90% Ziel).
4. Do **not** mark done until Kernel SLO ≥ **0.90 × estimated_peak** for that
   theory shape. Unique-key peak = DRAM roof
   (`BenchTierSpec::dram_roofline_hist_peak`); shared-cipher Atbash/totient peak =
   compute roof (`kSharedCipherComputeRoofRps`). Document BW / bytes/rune or
   identity-hist calib + date + GPU.

Repro baseline: [`scripts/cuda/capture_theory_baseline.ps1`](../../scripts/cuda/capture_theory_baseline.ps1).  
Repro specialized: [`scripts/cuda/capture_theory_specialized.ps1`](../../scripts/cuda/capture_theory_specialized.ps1).  
**S0 climb baseline (S0 vs Caesar twin only):**
[`scripts/cuda/capture_s0_climb_baseline.ps1`](../../scripts/cuda/capture_s0_climb_baseline.ps1) →
[`profiles/s0-climb/`](profiles/s0-climb/).  
**Cache / DRAM absolute (L2 + `dram__bytes.sum`):**
[`scripts/cuda/capture_cache_bound.ps1`](../../scripts/cuda/capture_cache_bound.ps1) →
[`profiles/cache_bound/`](profiles/cache_bound/).  
Write-ups: [`profiles/baseline/SUMMARY.md`](profiles/baseline/SUMMARY.md),
[`profiles/specialized/SUMMARY.md`](profiles/specialized/SUMMARY.md),
[`profiles/s0-climb/SUMMARY.md`](profiles/s0-climb/SUMMARY.md),
[`profiles/hist_local_caesar/SUMMARY.md`](profiles/hist_local_caesar/SUMMARY.md),
[`profiles/export_duty/SUMMARY.md`](profiles/export_duty/SUMMARY.md),
[`profiles/cache_bound/SUMMARY.md`](profiles/cache_bound/SUMMARY.md).

### Progress log

| Date | Tag | Path | Kernel | `(C,T)` | Kernel runes/s | Top stalls | nsys / ncu | Notes |
|------|-----|------|--------|---------|----------------|------------|------------|-------|
| 2026-10-01 | baseline | `profiles/baseline/` | `theory_chi2_hist_kernel` | 29×1M | **65.87B** cudaEvent | stall metrics n/a on sm_120 | ncu **155.7µs** @ T=262k; SM 74% DRAM 1% | vs Caesar twin **305B** cudaEvent; ncu duration **8.1×** Caesar |
| 2026-10-01 | baseline | `profiles/baseline/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **304.97B** cudaEvent | stall n/a | ncu **19.1µs** @ T=262k; SM 68% DRAM 2.4% | same cipher as theory row |
| 2026-10-01 | baseline | `profiles/baseline/` | catalog `F.affine` | 812×262k | **422.00B** cudaEvent | stall n/a | ncu **478µs**; SM 74% DRAM 0.15% | ~445B from ncu duration |
| 2026-10-01 | host-amortize | (code) | — | — | — | — | tests `[search][export][theory]` | `TheoryExportCache`: host bytecode once/URI; device `ops`/`imm` reused; `theory_scores_only`; scheduler loop shares cache |
| 2026-10-01 | interpreter-qw | (code) | `theory_chi2_hist_kernel` | 29×1M | **69.30B** cudaEvent | — | ncu ~157µs @ T=262k (≈baseline) | shared `ops`/`imm` (≤256) + `eval_at_trusted`; +~5% vs 65.87B baseline; Caesar twin 414B |
| 2026-10-02 | specialized | `profiles/specialized/` | `theory_chi2_hist_kernel` (S0) | 29×1M | **60.53B** cudaEvent | stall n/a | ncu **160.9µs** @ T=262k; SM 57% DRAM 2.6% | vs baseline 65.87B (noise); still ~7.5× slower than S1 ncu |
| 2026-10-02 | specialized | `profiles/specialized/` | `theory_hist_chi2_s1_lut_kernel` | 29×1M | **395.58B** cudaEvent | stall n/a | ncu **21.5µs** @ T=262k; SM 60% DRAM 2.8% | ≈ Caesar twin; **~44%** of 896B DRAM roof |
| 2026-10-02 | specialized | `profiles/specialized/` | `theory_hist_chi2_s2_linear_kernel` | 9×1M | **187.93B** cudaEvent | stall n/a | ncu **12.0µs** @ T=262k; SM 63% DRAM 4.9% | **~6.3×** vs baseline progressive 29.71B; **~21%** of 896B roof @ C=9 |
| 2026-10-02 | remesaure | `lp2-vigenere-lag-allpages` | campaign wall `vigenere_lag_stream` S0 | 140×page T | **~1.65e5** wall mean | — | 53/55 pages `WORKERS=8` | wall only; Kernel SLO gate unchanged (896B roof) |
| 2026-10-02 | specialized | `profiles/specialized/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **383.21B** cudaEvent | stall n/a | ncu **20.5µs** @ T=262k; SM 63% DRAM 2.8% | twin on same capture; S1 ncu duration ≈ Caesar |
| 2026-10-02 | **s0-climb** | `profiles/s0-climb/` | `theory_chi2_hist_kernel` | 29×1M | **59.44B** cudaEvent | stall n/a | ncu **157.4µs** @ T=262k; SM 58% DRAM 0.9% | **~6.6%** of 896B DRAM roof — climb baseline; vs twin **406.6B** (6.8×) |
| 2026-10-02 | **s0-climb** | `profiles/s0-climb/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **406.62B** cudaEvent | stall n/a | ncu **19.6µs** @ T=262k; SM 66% DRAM 2.6% | reference twin on same fair JSON capture |
| 2026-10-02 | **s0-climb-p1** | (code) | `theory_chi2_hist_kernel` | 29×1M | **~69–75B** cudaEvent | — | fair ~7–8% of 896B | tiles ~16 tok/thread; shared slots; trusted split; `__ldg` cipher |
| 2026-10-02 | **s0-climb-p2** | (code) | `theory_chi2_hist_kernel` | 29×1M | **~67–71B** cudaEvent | — | no net win vs p1 | tried uchar4 packs / 32-tok / launch_bounds — all **regressed** (compute-bound); keep p1 launch |
| 2026-10-02 | **dram-roof** | `BenchTierSpec` | all fused-hist shapes | — | — | — | Spec peak → **896B** (448B T3) | peak = physical GDDR7 / bytes/rune; **not** measured quiet max |
| 2026-10-02 | **hist_local_caesar** | `profiles/hist_local_caesar/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **48.77B** T1 / **51.70B** compare_caesar | stall n/a | ncu **563.9µs** @ T=262k; SM **25%** DRAM **0.24%** | HistFast `*_local` / `flush_local` **abandoned** — 32KiB shared; **~5.4%** of 896B; ~**8×** down vs prior ~383–407B |
| 2026-10-02 | **hist_local_caesar_regs** | `profiles/hist_local_caesar/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **18.90B** T1 / **19.32B** compare_caesar | stall n/a | ncu **1375.9µs** @ T=262k; SM **14.5%** DRAM **12.7%** | prod path: `add_local_regs` + `flush_regs_via_warp` (~1KiB shared); **~2.1%** of 896B; worse fair than 32KiB attempt and ~**20×** down vs ~400B twin (still grid.y=1024 over-tile) |
| 2026-10-03 | **hist_local_caesar_fat** | `profiles/hist_local_caesar/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **34.76B** T1 | stall n/a | ncu **813.0µs** @ T=262k; SM **3.77%** DRAM **8.42%**; grid **(29,32,1)** | `tiles_for` cap **32**; **~3.88%** of 896B; still ≪ twin |
| 2026-10-03 | **hist_local_caesar_revert** | `profiles/hist_local_caesar/` | `caesar_chi2_histogram_decrypt_kernel` | — | — | — | — | **prod restored to warp-private**; local/regs not shipped; see SUMMARY progression A→C |
| 2026-10-03 | **hist_warp_match** | `profiles/hist_local_caesar/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **113.8B** T1 | — | — | `add_private_match`; **12.7%** of 896B; **LOSE** vs ~356B; **F.\* not wired** |
| 2026-10-03 | **hist_local_s1** | `profiles/hist_local_caesar/` | `theory_hist_chi2_s1_lut_kernel` | 29×1M | **36.06B** fair S1 | — | — | regs + fat≤32; **~4%** of 896B; **LOSE** vs ~396B; **reverted** to warp-private (confirm **397.37B**) |
| 2026-10-03 | **hist_local_s2** | `profiles/hist_local_caesar/` | `theory_hist_chi2_s2_linear_kernel` | **841**×1M | **30.30B** local / **245.58B** warp | — | — | regs + fat≤32; **~3.4%** of 896B; **LOSE** vs warp@841 (**~27%**); **reverted**; suite default C→**841** |
| 2026-10-03 | **export_duty** | `profiles/export_duty/pre_residency/` | fair S0/S1/S2 + export stress | 29×1M fair; stress 64×65k×8 | **67.83B** S0 / **395.31B** S1 / **257.86B** S2 / twin **390.32B** | — | nsys: hist **94%** of kern; NVTX `h2d` **99%** range wall; `cudaMalloc` **94%** API | pre-residency baseline; idle GPU = host/PCIe/sync; **idle GPU ≠ Kernel SLO PCIe** |
| 2026-10-03 | **export_duty_accept** | `profiles/export_duty/` | fair S0/S1/S2 + pipeline stress | 29×1M fair; stress 64×65k×8 | **61.82B** S0 / **482.40B** S1 / **244.43B** S2 / twin **396.65B** | — | NVTX hist **27%**; `h2d` **17%** (was 99%); `cudaMalloc` **0.2%** API; non-kernel wall **~53×** ↓ | Track A metric A **PASS**; alloc pool **skipped** |
| 2026-10-03 | **roof_hist_fat64** | `profiles/roof_hist/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | sweep **~830–837B** / fair suite **783.5B** (was **509B**) | — | tile sweep; scores ≡ | fat-tile **WIN** — ship `production_tile_cap=64`; stretch **87%** fair / **~93%** sweep |
| 2026-10-03 | **roof_hist_wire** | `profiles/roof_hist/` | S1/S2/F.* + Caesar | 29×1M; S2 C=841 | Caesar **838B** / S1 **690B** / S2 **343B**; F.atbash **1525B** / F.affine **838B** | — | `HistFast::tiles_for` → fat-64; goldens green | wire fat-64; Caesar **Done** (~93.5%); S1 stretch; S0 unchanged |
| 2026-10-03 | **s2_ks29** | `profiles/roof_hist/` | `theory_hist_chi2_s2_linear_kernel` | 841×1M | **403.1B** S2 (was **343B**) | — | shared `ks[29]` precompute | ALU cut; ~**45%** of 896B; Caesar/S1 flat |
| 2026-10-03 | **s0_specialize** | `profiles/roof_hist/` | fair `T.theory.caesar_bytecode` → S1 | 29×1M | **707.6B** (`specialize_S1`; was ~61–68B S0) | — | emit prefer S1 in suite; export already S1/S2 | ~**79%** of 896B; ≈ S1 twin; hard-S0 unchanged |
| 2026-10-03 | **kernel_slo_accept** | `profiles/kernel_slo/` | theory fair ×3 + slo extended ×2 | fair 29×1M; F.* grids | twin med **783B** / specialize med **681B** / S1 noisy; F.vigenere·beaufort ≥90%; Atbash·totient `%peak>100` | — | quiet ACCEPTANCE | metric B **PARTIAL PASS** — see SUMMARY |
| 2026-10-06 | **traffic_model** | `profiles/traffic_model/` | ncu Atbash+totient hist | C=512 T=262144 | bytes/rune **0.01111** → peak **≈80.7 TB** | DRAM SoL ~0.4–1% | Spec not quiet-max | shared-cipher occupancy class |
| 2026-10-06 | **kernel_slo_requiet** | `profiles/kernel_slo/` + `dsl_smart/` | slo extended ×2 + dsl_smart ×3 | fair F.*/customs | Atbash/totient `%peak≤100` under 80.7 TB | — | re-quiet ACCEPTANCE | model gate **PASS**; stretch still compute-bound |
| 2026-10-06 | **dsl_smart_e2e** | `profiles/dsl_smart/` | fair customs ×3 (Atbash/Caesar/Affine/linear/autokey + twins) | fair T=1M | Caesar stretch ~**87%**; Atbash model OK ~2% of 80.7 TB; linear ~43%; autokey ~53%; Affine `%peak>100` intermittent | — | quiet E2E ACCEPTANCE | **PARTIAL PASS** — see SUMMARY; edge tags green |
| 2026-10-06 | **cache_bound** | `profiles/cache_bound/` | Caesar / Atbash / S1 / S2 / Affine | ncu T=262k + fair JSON | Affine **17.08 GB/s** DRAM (≈16.9); Caesar 28.6; Atbash 4.8; S1 27.4; S2 7.7 | L2 hit Atbash **98.5%** / Affine **97.6%**; DRAM SoL 0.5–3.1% | `-MetricsPreset cache_bound` + CSV | **PASS harness** — all ≪896 GB/s; bytes/rune ≪1 (shared cipher L2) |
| 2026-10-06 | **affine_traffic** | `profiles/traffic_model/` + Spec | `affine_chi2_hist_kernel` | C=812 T=262k | bytes/rune **0.01906** → peak **≈47.0 TB** | L2 97.6% DRAM 1.8% | Spec `kHistBytesPerRuneAffineSharedCipher` | **PASS model** — F.affine / dsl_smart Affine `%peak≤100`; not quiet-max |
| 2026-10-06 | **compute_roof** | Spec + `HistOccupancyRoof` | identity hist vs Atbash @ C=512 T=1M | fair | identity ~1.2–2.2 TB; Spec freeze **2.0 TB** | DRAM diary 80.7 TB | `kSharedCipherComputeRoofRps` | **PASS Spec** — Atbash/totient Done = compute roof; DRAM-bound N/A |
| 2026-10-07 | **s2_running_residue** | S2/S5 kernels + `HistTileCap` | `theory_hist_chi2_s2_linear_kernel` | 841×1M | med **~683B** (~76%; best **764B**) was **~402B** | — | running residue (no hot `%29`); S2 default tile **32** | **PARTIAL** — +~70% vs ks29; stretch 716.8B not median-stable |
| 2026-10-07 | **s1_lut_residency** | export scratch + fair suite | `theory_hist_chi2_s1_lut_kernel` | 29×1M | quiet band **~670–733B** (med ~**705B**, ~79%) | — | bake outside fair timer; export skips rebake when program+slots match | **PARTIAL** — noise tightened vs 303–692B; stretch 716.8B not median-stable |
| 2026-10-07 | **caesar_roof_done** | `profiles/roof_hist/` + dsl_smart | Caesar decrypt + shape twin | 29×1M | twin med-of-med **~786–790B** (~88%); bests **~836–843B**; dsl custom med **~814.5B Done** | — | `__restrict__`/`__ldg`; Caesar-first dsl_smart; tile **64** | **PARTIAL Done** — dsl custom ≥806.4B; twin median stretch on busy util |
| 2026-10-07 | **shared_cipher_atomic_climb** | Atbash/totient + shape Atbash | `atbash_chi2_hist` / totient / shape twin | 512×1M | Atbash prod **~1.5–1.75 TB** (~76–87% of 2.0 TB); totient tile32 **~1.17–1.37 TB**; A/B 32/64/128 | — | `__restrict__`/`__ldg`; `HistTileCap::kAtbash`/`kTotient`; totient default **32** | **PARTIAL** — absolute climb vs pre-~1.1–1.6 TB; Done 1.8 TB not median-stable; Caesar tile64 untouched |

Physical DRAM-roofline Spec (`BenchTierSpec`, RTX 5070 Ti):

| Theory shape | Spec id | `estimated_peak` | 90% gate | Fair cudaEvent | Status |
|--------------|---------|------------------|----------|----------------|--------|
| Caesar catalog twin (fat-64) | `T.theory.compare_caesar` | **896B** | **806.4B** | med-of-med **~786B** (~88%; bests **~836B** Done; quieter **~827B**) | **stretch** — dsl custom **Done**; twin median busy-noise |
| Caesar fair (specialize S1 when eligible) | `T.theory.caesar_bytecode` | **896B** | **806.4B** | med **680.8B** (~76%, `specialize_S1`) | **not stretch** — <80% |
| S1 LUT-29 (fat-64, resident LUT) | `T.theory.s1_lut29` | **896B** | **806.4B** | quiet **~670–733B** (~79%; bake outside timer) | **not stretch** — noise tightened; median still under 716.8B |
| bitmask_blend / progressive S2 linear | `T.theory.s2_linear` / `T.theory.progressive` | **896B** | **806.4B** | med **~683B** (~76% @ C=841, residue+tile32) | **not stretch** — climb from ~45%; atomic/L2 bound remains |

Kernel SLO ACCEPTANCE (metric B): [`profiles/kernel_slo/SUMMARY.md`](profiles/kernel_slo/SUMMARY.md).

`BenchTheorySuite` (`parcae-bench --suite theory`) gates fair rows with
`BenchTierSpec::pass_tier` (≥90% peak + `slo_min`). Short T is
`underfill_not_slo_gate`. When peak ≥100B, row detail annotates
`checkpoint_50B=hit|miss` (never replaces the 90% gate).

Full specialized write-up: [`profiles/specialized/SUMMARY.md`](profiles/specialized/SUMMARY.md).

---

## 4. Acceptance reminder

```text
done ⇔ kernel_runes_per_s >= 0.90 * estimated_peak(theory_shape)

# optional interim only if estimated_peak >> 50B:
checkpoint_50B ⇔ kernel_runes_per_s >= 50e9   # alone ≠ done
```

Campaign wall and `cells/s` stay **ops** metrics — document them separately;
never use them as the 90% gate.

Related:

- Throughput ceilings (catalog): [`cuda-throughput.md`](cuda-throughput.md)
- Bench timer contract: [`include/parcae/bench/bench_timer.hpp`](../../include/parcae/bench/bench_timer.hpp)
- Search operator guide: [`search-handbook.md`](search-handbook.md)
- Theory export kernel: [`Parcae/Parcae/cuda/theory_chi2_batch.cu`](../../Parcae/Parcae/cuda/theory_chi2_batch.cu)
