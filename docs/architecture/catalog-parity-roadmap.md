# Catalog CUDA parity + Pack D commit roadmap

**Status:** Frozen granular commit list for toolkit **1.1.0**  
**Plan freeze:** [`cuda-catalog-parity.md`](cuda-catalog-parity.md)  
**Pack D detail:** [`dsl-pack-d.md`](dsl-pack-d.md)  
**Exit tag:** `v1.1.0` (toolkit SemVer **1.1.0**; `dsl_spec_version` **1.1.0**)  

**Upstream:** Toolkit **1.0.0** (`v1.0.0`) — ℤ₂₉ Hill / autokey / grid catalog on CPU;
CUDA twins through `CudaFamilyId` Identity…Compose (`v0.3.0-cuda-parity`);
smart DSL + console (`v0.8.0-dsl-console`); Pack D deferred there.

North star:

```text
CPU ApplyTransform / ComposeTransform
  ==  CudaBackend / ComposeDriver     (byte-identical Index29)

SearchJob hill_2|hill_3|ciphertext_autokey|plaintext_autokey + backend=cuda
  →  GpuCandidateExport  (χ² decrypt; top-k ↔ CPU)

theory.py → parcae-compile (multi-diag + DslOptimize + z29_match)
  → artifact dsl_spec_version 1.1.0
```

Numbering is **local to this roadmap** (not a continuation of CUDA 1–42,
search-engine 1–52, or agent-tooling 1–35).

C++ / CUDA style (HARD): **no `namespace`s** (including anonymous namespaces in
new or touched `.cu`) — top-level `class Name` + `#ifndef NAME_HPP` headers only.

Naming: descriptive kebab paths only — **no** “phase*” filenames, tags, docs
titles, or CI job names.

**Agent git:** User owns commits, annotated tags, and release pushes; agent may
code and test only.

**Hardware reference (local CUDA):** RTX 5070 Ti (`sm_120`), Ryzen 7 7800X3D,
32 GB DDR5-6000. CMake and VS `CudaCompile` both include `sm_120`. Hosted CI
stays CPU-default.

---

## Progress

| Band | Commits | Status |
|------|---------|--------|
| A — Freeze sync | 1–3 | **done** |
| B — CUDA family params | 4–8 | **done** |
| C — Hill stream twins | 9–13 | next **9** |
| D — Autokey twins | 14–19 | pending |
| E — Grid permute twins | 20–24 | pending |
| F — ComposeDriver | 25–28 | pending |
| G — Search GPU export | 29–36 | pending |
| H — Pack D multi-diag | 37–41 | pending |
| I — Pack D `z29_match` | 42–47 | pending |
| J — Pack D `load_page` | 48–50 | pending |
| K — Pack D language stamp | 51–55 | pending |
| L — Exit gates | 56–62 | pending (tag/release **user-owned**) |

---

## A — Freeze sync

| # | Commit | Status |
|---|--------|--------|
| 1 | docs: add 1.1.0 commit roadmap mirror (**this**) + architecture README pointer | **done** |
| 2 | docs: correct cuda-catalog-parity cpu-export checkbox (`[x]` → `[ ]` until code lands) | **done** |
| 3 | docs: cuda-build arches list includes sm_120 (75;86;89;120) + Catch2 tag section prep | **done** |

---

## B — CUDA family params

| # | Commit | Status |
|---|--------|--------|
| 4 | feat(cuda): extend `CudaFamilyId` for 1.0 catalog families | **done** |
| 5 | feat(cuda): `ComposeStageParams` matrix/order arena fields | **done** |
| 6 | feat(cuda): `CudaParamsJson` parsers for hill/autokey/grid | **done** |
| 7 | test(cuda): `[cuda][params]` coverage for new PODs/JSON | **done** |
| 8 | build(cuda): register new kernel sources; add `sm_120` to `Parcae.vcxproj` CudaCompile | **done** |

---

## C — Hill stream twins

| # | Commit | Status |
|---|--------|--------|
| 9 | feat(cuda): `Hill2Kernel` stream twin on `Z29Matrix2Device` | pending |
| 10 | feat(cuda): `Hill3Kernel` stream twin on `Z29Matrix3Device` | pending |
| 11 | test(cuda): `[cuda][parity][hill]` bit-identity and length/singular edges | pending |
| 12 | feat(cuda): wire Hill into `CudaBackend` apply_into/prepare_params | pending |
| 13 | test(cuda): Hill via `CudaBackend` matches `Hill2`/`Hill3Transform` | pending |

---

## D — Autokey twins

| # | Commit | Status |
|---|--------|--------|
| 14 | feat(cuda): wire `CiphertextAutokeyKernel` through `CudaBackend` | pending |
| 15 | feat(cuda): `PlaintextAutokeyKernel` dense and interrupt twin | pending |
| 16 | test(cuda): `[cuda][parity][autokey][ptak]` CPU vs CUDA | pending |
| 17 | feat(cuda): `VariableDelayAutokeyKernel` twin | pending |
| 18 | test(cuda): `[cuda][parity][autokey][vda]` including lag-equals-primer equivalence | pending |
| 19 | test(cuda): autokey empty-key / size-mismatch / skip edges for PTAK and VDA | pending |

---

## E — Grid permute twins

| # | Commit | Status |
|---|--------|--------|
| 20 | feat(cuda): `GridPermuteKernel` gather/scatter device twin | pending |
| 21 | feat(cuda): spiral / boustrophedon / diagonal / columnar façades with host order fill | pending |
| 22 | test(cuda): `[cuda][parity][grid]` vs CPU transforms | pending |
| 23 | feat(cuda): wire grid families into `CudaBackend` | pending |
| 24 | test(cuda): grid in-place and rows×cols mismatch rejection | pending |

---

## F — ComposeDriver

| # | Commit | Status |
|---|--------|--------|
| 25 | refactor(cuda): replace `compose_driver` anonymous namespace with class | pending |
| 26 | feat(cuda): `ComposeDriver` stage launches for hill / autokey / grid | pending |
| 27 | test(cuda): `[cuda][compose]` recipes including hill/autokey/grid vs `ComposeTransform` | pending |
| 28 | test(cuda): compose rejects nested compose and bad arena slices | pending |

---

## G — Search GPU export

| # | Commit | Status |
|---|--------|--------|
| 29 | feat(search): `GpuCandidateExport` hill_2/hill_3 CUDA apply + host χ² | pending |
| 30 | feat(search): `GpuCandidateExport` CTAK via `DeepScoreBatch` autokey χ² | pending |
| 31 | feat(search): `GpuCandidateExport` PTAK CUDA path + host χ² | pending |
| 32 | feat(search): clear cpu-export-only for hill/CTAK/PTAK; extend `has_fused_cuda_chi2_export` | pending |
| 33 | feat(search): `SearchScheduler` CUDA export accepts new families under `allow_extended` | pending |
| 34 | test(search): `[search][export][parity]` hill/CTAK/PTAK CPU vs CUDA top-k | pending |
| 35 | test(search): export edges (odd hill length, empty key, max_candidates) | pending |
| 36 | docs: search-loop / handbook / engine / search README for hill/autokey CUDA export | pending |

---

## H — Pack D multi-diag

| # | Commit | Status |
|---|--------|--------|
| 37 | feat(dsl): `DslDiagList` accumulator class | pending |
| 38 | feat(dsl): gates collect multiple errors instead of first-fail | pending |
| 39 | feat(cli): `parcae-compile` prints all diagnostics; JSON `diagnostics[]` | pending |
| 40 | fix(dsl): any error yields non-zero exit and no ready artifact | pending |
| 41 | test(dsl): multi-error source yields at least two diagnostics | pending |

---

## I — Pack D `z29_match`

| # | Commit | Status |
|---|--------|--------|
| 42 | feat(dsl): list `z29_match` in `DslZ29Builtins` and `dsl.md` | pending |
| 43 | feat(dsl): BuildIr lowers `z29_match` to cascaded Select; E032 malformed | pending |
| 44 | feat(python): fail-loud `z29_match` stub in `parcae.dsl.math` | pending |
| 45 | feat(dsl): emit / verify / applicator path for match-lowered Select | pending |
| 46 | feat(examples): `theories/examples/match_select_example.py` | pending |
| 47 | test(dsl): `[dsl][match]` arms / fold / malformed / default-only | pending |

---

## J — Pack D `load_page`

| # | Commit | Status |
|---|--------|--------|
| 48 | feat(python): `parcae.corpus.load_page` / `LoadedPage` / LP2 map | pending |
| 49 | test(python): pytest resolution 0..57 and locked-fixture warning | pending |
| 50 | docs: cross-link `load_page` in search-handbook / python-transpiler / dsl-stubs | pending |

---

## K — Pack D language stamp

| # | Commit | Status |
|---|--------|--------|
| 51 | feat(dsl): bump `DslSpecVersion::current_string` to **1.1.0** | pending |
| 52 | docs(spec): `dsl.md` header **1.1.0** and `z29_match` section | pending |
| 53 | test(dsl): registry accepts **1.0.0** artifacts; MAJOR mismatch rejected | pending |
| 54 | chore(dsl): restamp example theory artifacts needing current minor | pending |
| 55 | docs: mark `DslOptimize`-in-`DslCompile` Pack D boxes done (already shipping) | pending |

---

## L — Exit gates

| # | Commit | Status |
|---|--------|--------|
| 56 | docs: cuda-build Catch2 tags for hill / autokey / grid / parity / export | pending |
| 57 | docs: check off cuda-catalog-parity and dsl-pack-d exit boxes | pending |
| 58 | docs: root README roadmap marks catalog CUDA parity + Pack D done | pending |
| 59 | docs: `release/1.1.0/RELEASE_NOTES.md` for toolkit **1.1.0** | pending |
| 60 | chore: bump toolkit version to **1.1.0** (CMake + `version.hpp`) | pending |
| 61 | test: `Version` + CLI `--status` assert toolkit **1.1.0** and `dsl_spec` **1.1.0** | pending |
| 62 | docs: catalog-parity-roadmap progress table all done | pending |

**User-owned after 62:** annotated tag `v1.1.0` → push → Release workflow →
verify `Parcae-v1.1.0-…` assets + `SHA256SUMS`.

---

## Dependency sketch

```text
Freeze sync (A)
  → CUDA params (B) → Hill (C) → Autokey (D) → Grid (E)
       → ComposeDriver refactor+stages (F)
       → Search GPU export (G)
  ↘ Pack D multi-diag (H) → z29_match (I) → load_page (J) → dsl_spec 1.1.0 (K)
  → Exit docs/version (L) → user tag/release
```

`DslOptimize` is already hooked in `DslCompile` (`[dsl][compile][optimize]`);
band K commit **55** only syncs Pack D checkboxes.

Search export for VDA / grid-read families stays **out of scope** (decode/compose
twins only). Nested compose stays unsupported.

---

## Related

| Doc | Role |
|-----|------|
| [`cuda-catalog-parity.md`](cuda-catalog-parity.md) | Exit freeze + locked decisions |
| [`dsl-pack-d.md`](dsl-pack-d.md) | Pack D required checklist |
| [`cuda-build.md`](cuda-build.md) | Local CUDA / Catch2 tags (CI CPU-default) |
| [`cuda-handoff.md`](cuda-handoff.md) | Twin list + parity procedure |
| [`cuda-reference.md`](cuda-reference.md) | CUDA module narrative |
| [`search-handbook.md`](search-handbook.md) | Operator guide (export families) |
| [`search-loop.md`](../spec/search-loop.md) | Normative search schemas |
| [`dsl.md`](../spec/dsl.md) | Normative DSL (`dsl_spec_version`) |
| [`python-transpiler.md`](python-transpiler.md) | Compiler architecture |
| [`release.md`](release.md) | Tag-only packaging cut |
| [`cuda-roadmap.md`](cuda-roadmap.md) | Prior CUDA twin roadmap (`v0.3.0`) |
| [`search-roadmap.md`](search-roadmap.md) | Prior search-engine commit list |
