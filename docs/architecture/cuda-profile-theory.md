# Theory CUDA profiling playbook (nsys / ncu)

**Status:** operator playbook for theory fused-χ² / search-export throughput  
**Hardware reference:** RTX 5070 Ti (sm_120) — same plate as [`cuda-throughput.md`](cuda-throughput.md)  
**Canonical catalog metric:** `BenchTimer` — 4 warmups + **median-of-3** `cudaEvent`, setup excluded  
**Theory acceptance (PRIMARY):** measured ≥ **90%** of the theory-shape
`estimated_peak` — **physical DRAM roofline** (**896B** @ 1 B cipher/rune on
RTX 5070 Ti), **not** a measured quiet max. Applies to **every** strategy that
runs, **including S0 bytecode**. Soft-fallback to S0 does not waive the S0 peak
gate. Normative contract: [`theory-hist-transpile.md`](theory-hist-transpile.md).
  
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
≥90% peak gate at fair T). Campaign wall stays in `research/run.log`.

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
| Issue-slot utilization / warp stall reasons | `stall_inst_fetch`, `stall_exec_dependency`, `stall_memory_throttle` |
| L1/L2 sectors on global loads | Bytecode `ops`/`imm` streaming |
| Achieved occupancy; grid vs active warps | Underfill at short `T` |

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

## 3. Progress rule (every throughput milestone)

After each meaningful change:

1. Save **ncu** report (`.ncu-rep` or exported CSV/text) under
   `docs/architecture/profiles/<tag>/` (when counters are permitted).
2. Save **nsys** report (`.nsys-rep`) + one-line stats summary.
3. Append a row to the progress table below (Ist Kernel SLO vs 90% Ziel).
4. Do **not** mark done until Kernel SLO ≥ **0.90 × estimated_peak** for that
   theory shape. Peak = DRAM roof (`BenchTierSpec::dram_roofline_hist_peak`),
   documented with BW assumption + bytes/rune + date + GPU.

Repro baseline: [`scripts/cuda/capture_theory_baseline.ps1`](../../scripts/cuda/capture_theory_baseline.ps1).  
Repro specialized: [`scripts/cuda/capture_theory_specialized.ps1`](../../scripts/cuda/capture_theory_specialized.ps1).  
**S0 climb baseline (S0 vs Caesar twin only):**
[`scripts/cuda/capture_s0_climb_baseline.ps1`](../../scripts/cuda/capture_s0_climb_baseline.ps1) →
[`profiles/s0-climb/`](profiles/s0-climb/).  
Write-ups: [`profiles/baseline/SUMMARY.md`](profiles/baseline/SUMMARY.md),
[`profiles/specialized/SUMMARY.md`](profiles/specialized/SUMMARY.md),
[`profiles/s0-climb/SUMMARY.md`](profiles/s0-climb/SUMMARY.md),
[`profiles/hist_local_caesar/SUMMARY.md`](profiles/hist_local_caesar/SUMMARY.md).

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

Physical DRAM-roofline Spec (`BenchTierSpec`, RTX 5070 Ti):

| Theory shape | Spec id | `estimated_peak` | 90% gate | Fair cudaEvent (2026-10-02) | Status |
|--------------|---------|------------------|----------|------------------------------|--------|
| Caesar-as-bytecode (S0 interpreter) | `T.theory.caesar_bytecode` | **896B** | **806.4B** | ~69–75B (~8%) | **not Done** — compute-bound |
| S1 LUT-29 (caesar/affine-shaped) | `T.theory.s1_lut29` | **896B** | **806.4B** | ~396–404B (~44%) | **not Done** — need DRAM SoL |
| bitmask_blend / progressive S2 linear | `T.theory.s2_linear` / `T.theory.progressive` | **896B** | **806.4B** | ~188–189B (~21% @ C=9) | **not Done** |

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
