# Toolkit exit: CUDA catalog parity + DSL Pack D

**Status:** Exit freeze — toolkit **1.1.0** (tag cut is user-owned)  
**Exit tag (locked):** `v1.1.0`  
**Toolkit SemVer (locked):** `1.1.0`  
**Language SemVer (locked with Pack D):** `dsl_spec_version` **`1.1.0`** (MINOR)

**Upstream:** Toolkit **1.0.0** (`v1.0.0`) — ℤ₂₉ Hill / autokey / grid catalog on CPU;
prior CUDA twins through `CudaFamilyId` Identity…Compose (`v0.3.0-cuda-parity`);
smart DSL + console (`v0.8.0-dsl-console`); Pack D deferred there  
**Themes:** GPU twins + bit-identity for every **1.0 catalog transform**; search
export for `hill_*` / CTAK / PTAK no longer CPU-only; DSL Pack D authoring surface  
**Cut procedure:** [`release.md`](release.md)  
**Pack D detail checklist:** [`dsl-pack-d.md`](dsl-pack-d.md)

```text
CPU ApplyTransform / ComposeTransform
  ==  CudaBackend / ComposeDriver twins     (byte-identical Index29)

SearchJob hill_2|hill_3|ciphertext_autokey|plaintext_autokey + backend=cuda
  →  GpuCandidateExport  (χ² decrypt; top-k ↔ CPU oracle)

theory.py  →  parcae-compile (multi-diag + DslOptimize + z29_match)
  →  artifact dsl_spec_version 1.1.0
```

---

## Locked decisions

| Topic | Decision |
|-------|----------|
| Exit tag | `v1.1.0` (toolkit **1.1.0**) |
| Version order | `1.0.0` catalog cut → **`1.1.0` CUDA catalog parity + Pack D** |
| C++ / CUDA style (HARD) | **No `namespace`s** (including anonymous namespaces in new or touched `.cu`); one top-level `class Name` per header; `#ifndef NAME_HPP` / `#endif // NAME_HPP` |
| Naming | Descriptive kebab paths only — **no** “phase*” filenames, tags, docs titles, or CI job names |
| Agent git | User owns commits, annotated tags, and release pushes; agent may code and test only |
| Twin scope | All toolkit **1.0** catalog transforms: `hill_2`, `hill_3`, `ciphertext_autokey`, `plaintext_autokey`, `variable_delay_autokey`, `spiral_read`, `boustrophedon_read`, `diagonal_read`, `columnar_transposition` — plus existing CudaFamily stages remain |
| ComposeDriver | Host-orchestrated stage launches; new families as stages; nested compose still unsupported |
| Search export | `hill_2` / `hill_3` / `ciphertext_autokey` / `plaintext_autokey` remain `is_cpu_export_only_family`; **`theory` fused χ²** via `GpuCandidateExport::theory_explicit_params` — prefers `TheoryHistChi2Emit` **S1/S2** specialized hist, soft-fallback **S0** `TheoryChi2Batch` (`has_fused_cuda_chi2_export("theory")`) |
| Search families | VDA / spiral / columnar / boustrophedon / diagonal remain decode/compose catalog — **not** new `SearchJob.family` values in this cut |
| Fused χ² | Decrypt + `chi2_english_gp_v0` only (same contract as today’s fused export) |
| `dsl_spec_version` | MINOR bump to **`1.1.0`** with Pack D; MAJOR stays **1**; artifacts stamped `1.0.0` remain loadable |
| `dsl_ast_json_version` | Stays **`1.1.0`** (already shipping directives) |
| Pack D required items | `load_page`, multi-diag, `z29_match`, `DslOptimize` hooked in `DslCompile` — see [`dsl-pack-d.md`](dsl-pack-d.md) |
| Pack D deferred further | `--dump-ir` / `--dump-scopes`, IR source maps, `--strict-portable`, testing-decorator verify schedules, compile cache — **not** required for `v1.1.0` |

Architecture companions: [`cuda-handoff.md`](cuda-handoff.md), [`cuda-reference.md`](cuda-reference.md),
[`cuda-abi.md`](cuda-abi.md), [`search-handbook.md`](search-handbook.md),
[`python-transpiler.md`](python-transpiler.md), [`dsl-pack-d.md`](dsl-pack-d.md).  
Headers: [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/),
[`include/parcae/search/`](../../include/parcae/search/),
[`include/parcae/dsl/`](../../include/parcae/dsl/).

---

## Baseline gaps (at freeze)

Facts the implementation must close — not optional stretch goals.

| Surface | Today |
|---------|--------|
| `CudaFamilyId` | Identity…Compose only (`params.hpp`) |
| `CudaBackend` / `ComposeDriver` | No Hill, CTAK/PTAK/VDA, or grid-read stages |
| Hill / matrix on GPU | `Z29Matrix{2,3}Device` + `Z29MatrixDeviceOps` only — no stream Hill twin |
| CTAK | `CiphertextAutokeyKernel` + `DeepScoreBatch` autokey χ² exist — **not** in Backend/ComposeDriver |
| PTAK / VDA / grids | CPU transforms only |
| Search | `SearchJob::is_cpu_export_only_family` forces CPU for hill / CTAK / PTAK; `theory` uses fused CUDA χ² (S1/S2 prefer, S0 soft-fallback) when `has_fused_cuda_chi2_export("theory")` |
| `DslOptimize` | Library + hooked in `DslCompile` after BuildIr (`[dsl][compile][optimize]`) |
| Pack D language | `dsl_spec_version` still **1.0.0**; no `z29_match`; no multi-error bag; no `parcae.corpus.load_page` |

---

## Exit criteria → `v1.1.0`

Declare toolkit **1.1.0** complete when **all required** boxes below are green
(then the user cuts the annotated tag). Pack D boxes are required for this tag
(detail in [`dsl-pack-d.md`](dsl-pack-d.md)).

### CUDA family / params

- [ ] `CudaFamilyId` extended for Hill2/3, CiphertextAutokey, PlaintextAutokey,
      VariableDelayAutokey, SpiralRead, BoustrophedonRead, DiagonalRead,
      ColumnarTransposition
- [ ] `CudaFamilyIdUtil::from_transform_id` maps every 1.0 catalog `TransformId`
- [ ] `ComposeStageParams` + `ComposeParamsHost` arenas for matrices / order indices
      (key arena pattern unchanged for keyed stages)
- [ ] `CudaParamsJson` parses stage JSON for the new families (nested compose still rejected)
- [ ] Catch2 `[cuda][params]` covers new PODs / JSON

### Transform twins (bit-identity)

- [ ] `Hill2Kernel` / `Hill3Kernel` stream twins on `Z29Matrix{2,3}Device`
- [ ] `CiphertextAutokeyKernel` wired through `CudaBackend` (kernel already exists)
- [ ] `PlaintextAutokeyKernel` twin
- [ ] `VariableDelayAutokeyKernel` twin (`L == p` mode matches CTAK/PTAK)
- [ ] `GridPermuteKernel` + façades for spiral / boustrophedon / diagonal / columnar
- [ ] Catch2 bit-identity vs CPU:
      `[cuda][parity][hill]`, `[cuda][parity][autokey][ptak]`,
      `[cuda][parity][autokey][vda]`, `[cuda][parity][grid]`
      (device skip when `PARCAE_HAS_CUDA` unset)

### Backend + ComposeDriver

- [ ] `CudaBackend::apply_into` / `prepare_params` dispatch for all new families
- [ ] `ComposeDriver` stage launches for all new families
- [ ] Touched ComposeDriver helpers use a **class** (no anonymous `namespace` in
      new or rewritten `.cu` helpers)
- [ ] Compose recipe parity vs `ComposeTransform` (incl. hill / autokey / grid stages)
      — `[cuda][compose]` (and/or dedicated compose parity cases)

### Search GPU export

- [ ] `GpuCandidateExport` paths for `hill_2` / `hill_3` (CUDA apply + host χ²
      and/or fused hist; decrypt + `chi2_english_gp_v0`)
- [ ] CTAK fused export via `DeepScoreBatch::launch_autokey_chi2_async` (or equivalent)
- [ ] PTAK CUDA export path with CPU top-k ordering parity
- [x] `SearchJob::is_cpu_export_only_family` no longer includes hill / CTAK / PTAK
      (`theory` not hard CPU-only; `has_fused_cuda_chi2_export("theory")` true with
      `GpuCandidateExport::theory_explicit_params` — S1/S2 specialized prefer +
      S0 `TheoryChi2Batch` soft-fallback)
- [ ] `SearchScheduler::export_cuda_fused` accepts the new families under
      `allow_extended_families`
- [x] Catch2 `[search][export][theory][parity]` CPU↔CUDA theory top-k on fixed grids
- [x] Catch2 `[cuda][theory][edge]` specialized vs bytecode top-k + Autokey→S0 /
      Div0 +inf / interrupt reject
- [ ] Catch2 `[search][export][parity]` (or sibling tags) CPU↔CUDA top-k on fixed grids
      for hill / CTAK / PTAK
- [ ] Specs updated: [`search-loop.md`](../spec/search-loop.md),
      [`search-handbook.md`](search-handbook.md),
      [`search-engine.md`](search-engine.md),
      [`include/parcae/search/README.md`](../../include/parcae/search/README.md)

### DSL Pack D (required on this tag)

- [ ] All required boxes in [`dsl-pack-d.md`](dsl-pack-d.md) green
      (`dsl_spec` **1.1.0**, `load_page`, multi-diag, `z29_match`, Optimize in `DslCompile`)

### Gates & release cut

- [ ] Hosted CI remains CPU-default; CUDA parity tags documented in
      [`cuda-build.md`](cuda-build.md) § Catch2 tags where new tags land
- [ ] Root README roadmap marks **catalog CUDA parity + Pack D** done with exit tag
- [ ] [`RELEASE_NOTES.md`](../../release/RELEASE_NOTES.md) +
      [`release/1.1.0/`](../../release/1.1.0/) for 1.1.0
- [ ] Toolkit version **1.1.0** (CMake + `version.hpp` + smoke asserts)
- [ ] Smoke: `Version` + `parcae-search-cycle --status` /
      `parcae-compile --status` → toolkit **1.1.0**; compile status reports
      `dsl_spec_version` **1.1.0**
- [ ] Annotated tag `v1.1.0`; Release workflow publishes `Parcae-v1.1.0-…` artifacts
      (**user-owned**)

---

## Out of scope (explicit)

- Replacing hand CUDA Tier-A twins with DSL-only kernels
- New search families for VDA or grid-read transforms
- Pack D stretch: `--dump-ir` / `--dump-scopes`, IR source maps, `--strict-portable`,
  testing-decorator verify schedules, Param `where=`, compile cache
- Agent creating/pushing annotated tags or GitHub release assets
- Treating toolkit **1.1.0** as a `dsl_spec_version` **MAJOR** bump
- “phase*” naming anywhere in files, tags, or CI
- Claiming theory Kernel SLO **done** until fair runs hit ≥90% of the
  **physical DRAM roofline** (`BenchTierSpec` 896B @ 1 B/rune) —
  [`cuda-profile-theory.md`](cuda-profile-theory.md); S1/S2 specialized emit
  already landed — measured ~44% / ~21% of roof today
---

## Document history

Freeze checklist for toolkit **1.1.0**: CUDA twins for the 1.0 catalog,
search export lift for hill/autokey, and DSL Pack D. Engineering narrative stays
in [`cuda-reference.md`](cuda-reference.md), [`search-handbook.md`](search-handbook.md),
and [`python-transpiler.md`](python-transpiler.md); Pack D authoring checklist in
[`dsl-pack-d.md`](dsl-pack-d.md); packaging in [`release.md`](release.md).
