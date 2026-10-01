# Theory CUDA profiling playbook (nsys / ncu)

**Status:** operator playbook for theory fused-χ² / search-export throughput  
**Hardware reference:** RTX 5070 Ti (sm_120) — same plate as [`cuda-throughput.md`](cuda-throughput.md)  
**Canonical catalog metric:** `BenchTimer` — 4 warmups + **median-of-3** `cudaEvent`, setup excluded  
**Theory acceptance (PRIMARY):** measured ≥ **90%** of the theory-shape `estimated_peak` (peak must be documented before claiming done)  
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
[`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp). Theory search
today reports Campaign wall in `research/run.log` (`engine=parcae`,
`export_backend=cuda`). Until a theory `BenchTierSpec` row exists, derive Kernel SLO from
`parcae-bench --suite theory --allow-cuda` (cudaEvent median-of-3 via
`BenchTimer`) or from **ncu** duration on `theory_chi2_hist_kernel` at fair
`(C,T)`. Campaign wall stays in `research/run.log`.

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
| `theory_chi2_hist_kernel` | Theory bytecode hist ([`theory_chi2_batch.cu`](../../Parcae/Parcae/cuda/theory_chi2_batch.cu)) |
| `caesar` / `atbash_caesar_chi2_hist_kernel` | T1 / compose path |
| `affine_chi2_hist_kernel` | F.affine |
| `atbash_chi2_hist_kernel` | F.atbash |

When a specialized theory hist lands, add its `__global__` name to the filter
list and keep the bytecode kernel as the regression baseline.

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
   theory shape, with peak documented (method + `(C,T)` + date + GPU).

Repro: [`scripts/cuda/capture_theory_baseline.ps1`](../../scripts/cuda/capture_theory_baseline.ps1).
Full write-up: [`profiles/baseline/SUMMARY.md`](profiles/baseline/SUMMARY.md).

### Progress log

| Date | Tag | Path | Kernel | `(C,T)` | Kernel runes/s | Top stalls | nsys / ncu | Notes |
|------|-----|------|--------|---------|----------------|------------|------------|-------|
| 2026-10-01 | baseline | `profiles/baseline/` | `theory_chi2_hist_kernel` | 29×1M | **65.87B** cudaEvent | stall metrics n/a on sm_120 | ncu **155.7µs** @ T=262k; SM 74% DRAM 1% | vs Caesar twin **305B** cudaEvent; ncu duration **8.1×** Caesar |
| 2026-10-01 | baseline | `profiles/baseline/` | `caesar_chi2_histogram_decrypt_kernel` | 29×1M | **304.97B** cudaEvent | stall n/a | ncu **19.1µs** @ T=262k; SM 68% DRAM 2.4% | same cipher as theory row |
| 2026-10-01 | baseline | `profiles/baseline/` | catalog `F.affine` | 812×262k | **422.00B** cudaEvent | stall n/a | ncu **478µs**; SM 74% DRAM 0.15% | ~445B from ncu duration |
| 2026-10-01 | host-amortize | (code) | — | — | — | — | tests `[search][export][theory]` | `TheoryExportCache`: host bytecode once/URI; device `ops`/`imm` reused; `theory_scores_only`; scheduler loop shares cache |
| 2026-10-01 | interpreter-qw | (code) | `theory_chi2_hist_kernel` | 29×1M | **69.30B** cudaEvent | — | ncu ~157µs @ T=262k (≈baseline) | shared `ops`/`imm` (≤256) + `eval_at_trusted`; +~5% vs 65.87B baseline; Caesar twin 414B |

Peak calibration rows (fill when specialized emit exists):

| Theory shape | `estimated_peak` | 90% gate | Method | Date |
|--------------|------------------|----------|--------|------|
| Caesar-as-bytecode (S0 interpreter) | TBD (interim **~69B** fair after shared+trusted) | TBD | cudaEvent @ T≥2^20 | 2026-10-01 |
| bitmask_blend S2 (planned) | | | cudaEvent / ncu @ T≥2^20 | |

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
