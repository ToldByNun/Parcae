# DSL smart hist — customs without presets at catalog speed

**Status:** normative architecture contract for **smart** theory fused-χ² specialize  
**Hardware reference:** RTX 5070 Ti (sm_120), Ryzen 7 7800X3D, 32 GB DDR5-6000 —
same plate as [`cuda-throughput.md`](cuda-throughput.md)  
**Canonical peaks / pass rule:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp)  
**Strategy / Done base contract:** [`theory-hist-transpile.md`](theory-hist-transpile.md)  
**Fair Kernel SLO tool (customs):** `parcae-bench --suite dsl_smart --allow-cuda`  
**Fair Kernel SLO tool (theory S0–S2):** `parcae-bench --suite theory --allow-cuda`  
**Profiling playbook:** [`cuda-profile-theory.md`](cuda-profile-theory.md)  
**Emit / classify (today):** [`theory_hist_chi2_emit.hpp`](../../include/parcae/dsl/theory_hist_chi2_emit.hpp)  
**Launch façade:** [`theory_hist_chi2_launch.hpp`](../../Parcae/Parcae/cuda/theory_hist_chi2_launch.hpp)  
**Artifact hist layout:** [`theory-artifact.md`](../spec/theory-artifact.md)  
**Authoring:** [`python-transpiler.md`](python-transpiler.md)

This document freezes **how** self-written HotLoop math (no catalog
`TransformId`, no compose builtin preset) must reach the **same Kernel SLO
class** as equivalent catalog twins. It extends — does not replace —
[`theory-hist-transpile.md`](theory-hist-transpile.md).

Agents implement and test only; they do **not** create git commits, tags, or
GitHub PRs.

---

## 1. Product rule (binding)

```text
same_math  ⇒  same_fast_hist_path

# Matching is by algebra on Z29Expr after normalize.
# Matching MUST NOT depend on:
#   - theory name / family string
#   - TransformId / catalog API / compose step id
#   - presence of z29_atbash (or other) Call nodes when arith is equivalent
```

Example: HotLoop `return 28 - x` (or equivalent) **must** be eligible for the
same `HistFast::dec_atbash` hist twin class as catalog `F.atbash`, without the
author calling `z29_atbash` or setting `family=atbash`.

Unbounded DSL cannot promise every program hits Atbash absolute RPS. Done is
always vs the **emitted** shape’s Spec peak (physical DRAM roof unless the
traffic model for that shape is corrected). Soft-fallback S0 stays correct and
is **not** a performance free pass.

---

## 2. Pipeline (target)

```text
Python HotLoop decrypt_step
  → TheoryIr + Z29Expr
  → Z29ExprNormalize          (algebraic rewrite; CPU; idempotent)
  → TheoryShapeMatch          (ShapeId + param bindings; name-irrelevant)
  → TheoryHistChi2Emit        (plans / sources; prefer shape before generic S1)
  → TheoryHistChi2Launch      (shape twins | S1 | S2 | S3/module | S4 | S0)
  → GpuCandidateExport        (prefer specialized; soft-fallback S0)
```

| Layer | Class (planned / shipping) | Role |
|-------|----------------------------|------|
| Normalize | `Z29ExprNormalize` ([`z29_expr_normalize.hpp`](../../include/parcae/dsl/z29_expr_normalize.hpp)) | Fold, commute, atbash-as-arith, affine/caesar/linear normal forms — **shipping** |
| Match | `TheoryShapeMatch` ([`theory_shape_match.hpp`](../../include/parcae/dsl/theory_shape_match.hpp)) | `ShapeId` from normalized tree + `cipher_var` — **shipping** |
| Emit | `TheoryHistChi2Emit` | Existing façade; must run normalize+match before generic S1 |
| Lower (long tail) | `TheoryHistExprLower` | Device decrypt fragment for S3 / module — **shipping** (bounded caps) |
| Module | `TheoryHistModule` | Optional cubin/NVRTC by URI+digest; soft S0 on fail |
| Hist primitives | `HistFast` | Shared decode with catalog (`dec_atbash`, caesar, affine, …) |

Shipping today (honest): normalize+match before S1; `ShapeInline` Atbash/Caesar/
Affine-decrypt use in-lib `TheoryHistChi2Shape`; Affine encrypt-form still S1
soft twin; residual FxOnly uses **device LUT bake** into `TheoryDeviceScratch`
(bind slots on host; bake+hist on device; domain → +inf via `lane_err`); launch
prefer `ShapeInline→S1→S2→S0`; emit sources still discarded at export;
`has_specialized` still false for NVRTC modules.

---

## 3. ShapeId ladder

Prefer order when multiple apply (first match wins after normalize):

| Prefer | ShapeId | Math (after normalize) | Runtime twin (target) | Spec peak class |
|--------|---------|------------------------|-----------------------|-----------------|
| 1 | `Atbash` | `atbash(x)` equivalent (incl. pure arith) | Shape hist on `HistFast::dec_atbash` | **896B** (fix model if `%peak>100`) |
| 2 | `Caesar` | `x ± shift` | Shape hist / Caesar decode | **896B** |
| 3 | `Affine` | invertible `a·x+b` | Shape hist / affine decode | **896B** |
| 4 | `LinearKeystream` | `x ± (b0 + b1·i)` (+ widened linear) | `TheoryHistChi2S2` | **896B** (`T.theory.s2_linear`) |
| 5 | `FxOnly` | other `f(x; params)`, no stream `i` | S1 device LUT bake → hist (no host `eval_at×29×C`) | **896B** (`T.theory.s1_lut29`) |
| 6 | `KeyedGeneral` | uses `i`, not linear S2 | S3 expr-inline or module | **896B** (row when Spec exists) |
| 7 | `Autokey` | contains `z29_autokey_shift` | S4 AutokeyRing+hist | **896B** (unless traffic differs) |
| 8 | `PolyKeystream` | low-degree / bitmask-like beyond linear | S5 twin or S3 module | **896B** |
| 9 | `Unknown` | caps / unsupported / prefer_branch (until twin) | S0 bytecode | **896B** (soft-fallback gate) |

Export prefer order (document; implement as each lands):

```text
ShapeInline → S1 → S2 → S5 → S3 → S4 → S0
```

`ShapeInline` covers `Atbash` / `Caesar` / `Affine` twins. Catalog
`TransformId` search remains a **separate** path; theory customs must not need
it.

### Name-irrelevant matching (examples)

| Authoring | Must match |
|-----------|------------|
| `28 - x` / `Add(Lit(28), Neg(x))` / `z29_atbash(x)` | `Atbash` |
| `x + shift` / `x - shift` (param `shift`) | `Caesar` |
| `a * x + b` with inv(a) in domain | `Affine` |
| `x - (b0 + b1 * i)` progressive-style | `LinearKeystream` |
| `x - (b1 * i)` / `x ± i` / const `b0`/`b1` Lits | `LinearKeystream` (widened) |
| Theory named `foo_bar` that is Atbash arith | still `Atbash` |
| `family` / URI string mentioning “caesar” but math is affine | `Affine` (math wins) |

`DslPeakSanity::suggest_tier` name heuristics are **ops / diary only**. They
MUST NOT drive Kernel SLO launch routing.

---

## 4. Done and stretch vs 896B DRAM roof

Canonical peak: physical GDDR7 roof on the reference plate —

```text
estimated_peak = 896e9 / 1 B_cipher_per_rune = 896B runes/s
```

| Gate | Rule | Notes |
|------|------|-------|
| **Done (PRIMARY)** | `fair_kernel_runes_per_s ≥ 0.90 × estimated_peak(emitted_shape)` **and** ≥ `slo_floor` | Same as `BenchTierSpec::pass_tier`; fair `T ≥ 2^20` |
| **Stretch** | ≥ **80%** of 896B (≈716.8B) | Interim climb bar; never replaces Done |
| **checkpoint_50B** | Annotation only when peak ≫ 50B | Never replaces Done |
| **Model** | Printed `%peak` must stay ≤ **100** | If `>100`, fix bytes/rune (or shape peak), **do not** lower Spec to quiet max |
| **Campaign wall** | Never PRIMARY | Short page `T` / scheduler wall = ops only |

```text
done(emitted)  ⇔  KernelSLO ≥ 0.90 × peak(emitted)  ∧  ≥ slo_floor(emitted)
stretch(emitted) ⇔  KernelSLO ≥ 0.80 × peak(emitted)
```

Soft-fallback to S0: scores ≡ bytecode oracle; PRIMARY for that launch is the
**S0 / hard-fallback** shape peak (still 896B under the 1 B/rune model until a
distinct traffic row exists). Specialize-away raises absolute RPS; it does not
delete the soft-fallback correctness duty.

Self-written shape twins share the **same** traffic model as their catalog
counterparts (e.g. Atbash shape twin ↔ `F.atbash`). If catalog Atbash prints
`%peak>100`, that is a **model defect** for both paths.

Metric B quiet plate: [`profiles/kernel_slo/SUMMARY.md`](profiles/kernel_slo/SUMMARY.md).  
Smart-customs ACCEPTANCE digests: [`profiles/dsl_smart/SUMMARY.md`](profiles/dsl_smart/SUMMARY.md)
(capture via [`scripts/cuda/capture_dsl_smart.ps1`](../../scripts/cuda/capture_dsl_smart.ps1)).

---

## 5. Normalize rules (contract minimum)

`Z29ExprNormalize` is pure CPU, deterministic, idempotent. On ambiguity it
**fails closed** (returns input unchanged; no wrong specialize).

Minimum pack (each needs goldens when implemented):

| Rule | Intent |
|------|--------|
| Lit / LitMod fold | Constant subgraphs reduce mod 29 where safe |
| Add/Mul commute sort | `b+x` ≡ `x+b` for matching |
| Atbash-as-arith | `Sub(Lit(28), x)` and equivalents → Atbash normal form |
| Caesar form | `Add/Sub(x, shift)` → Caesar normal form |
| Affine form | `Add(Mul(a,x), b)` (and safe rearrangements) → Affine normal form |
| Linear S2 form | `x ± (b0 + b1·i)` permutations, bare `b1·i`, const Lits → LinearKeystream |
| Autokey / prefer_branch | **No** rewrite across `z29_autokey_shift` or `prefer_branch` Select |

Builtin Call nodes (`z29_atbash`, …) remain valid input; arith-only trees must
reach the same normal form without them.

---

## 6. Edge-case matrix

| Case | Required behavior |
|------|-------------------|
| Self-written Atbash arith vs `z29_atbash` Call | Same `ShapeId`; scores ≡; both specialized when twins ship |
| Affine with non-invertible `a` | No false Affine match; lane `+inf` / soft S0; no UB |
| Div0 / inv domain | Lane `+inf`; interrupt rules unchanged |
| Autokey in decrypt | S4 when shipped; until then hard S0; scores ≡ CPU |
| `prefer_branch` Select | S0 until a measured divergent specialized path exists |
| Hoists on S1 | Soft S0 until wired; test locks the policy |
| Caps (`ops` / `stack` / `slots`) | S0 or stable reject; never UB |
| Empty cipher / `C=0` / `T=0` | Reject |
| Non-empty host interrupt | Reject export |
| Soft-fallback after shape/S1/S2/S3/module fail | Scores ≡ bytecode; `export_backend=cuda` |
| Cipher page change | Re-H2D via residency scratch pipeline |
| Module / NVRTC fail / OOM | Soft S0; `Status` surfaced; no partial scores |
| `%peak > 100` | Traffic-model bug — fix model |
| Theory name unrelated to math | Math still matches |
| Catalog `family=atbash` job | Unchanged `TransformId` path; not required for theory customs |
| Short campaign `T` | Duty may stay host-bound; PRIMARY remains fair `T≥2^20` |

Suggested Catch2 tags (no roadmap words in tags): `[dsl][normalize]`,
`[dsl][shape]`, `[dsl][emit][hist][shape]`, `[cuda][theory][edge][shape]`,
`[search][export][theory][shape]`.

---

## 7. Engineering rules (agent + code)

### Agent policy

- **Code and test only.** No `git commit`, push, tag, or `gh` from the agent.
- Do not invent “roadmap labels” in the tree: **no** `phase*` filenames, comments,
  NVTX ranges, Catch tags, or profile dir names; **no** “Commit N” strings in
  repo files (plans outside the tree may number work units).

### C++ / CUDA style

- **No** C++ or CUDA `namespace` (including anonymous).
- One top-level `class` per header; file-scope `__global__` + class façades.
- Header skeleton:

```cpp
#ifndef NAME_HPP
#define NAME_HPP

class Name {
public:

private:

};

#endif // NAME_HPP
```

- C++20; CUDA **sm_120**; primarily headers under `include/parcae/dsl/` and
  kernels under `Parcae/Parcae/cuda/`.

### Hardware plate

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5070 Ti (sm_120), ~896 GB/s GDDR7 |
| CPU | AMD Ryzen 7 7800X3D |
| RAM | 32 GB DDR5-6000 |
| Storage / board | MSI M560 2TB; PRO B850-S WIFI6E (MS-7E80) |

---

## 8. Honest shipping snapshot vs this contract

| Capability | Today | Contract target |
|------------|-------|-----------------|
| `Z29ExprNormalize` | **Shipping** (`[dsl][normalize]`) | Idempotent arith → Atbash / commute forms |
| `TheoryShapeMatch` | **Shipping** (`[dsl][shape]`) | ShapeId ladder; name-irrelevant; autokey / prefer_branch / non-inv affine |
| Self-written Atbash/Caesar/Affine-decrypt | `ShapeInline` + `TheoryHistChi2Shape` twins | Affine encrypt-form S1 soft; S3/S4/S5 |
| Name / API required for fast path | Effective yes (catalog) | **No** — algebra only |
| S3 / S4 / S5 | S3 ExprLower+launch shipping; S4/S5 stub | Real launch or module |
| Emit sources / `hist_module` | `hist_plan.json` (+ optional `hist/*.{hpp,cu}`); module null | `hist_plan` + optional module load |
| `has_specialized` | Always false | True when module/shape plan present |
| Fair specialize Caesar bytecode | S1 when eligible | Keep; shape twins supersede when richer |

---

## 9. Exit checklist (smart hist)

Declare this smartness workstream complete only when **all** apply:

- [ ] Self-written Atbash / Caesar / Affine (pure arith HotLoop) → shape twin; scores ≡ bytecode; fair Kernel SLO in catalog twin class (`parcae-bench --suite dsl_smart`; noise under `profiles/dsl_smart/`)
- [x] Self-written linear `x±(b0+b1·i)` → S2 (widened), not S0
- [x] Non-linear `i` customs → S3 or module (unless caps / prefer_branch / pre-S4 autokey)
- [ ] Autokey customs → S4 when implemented
- [ ] Normalize + match name-irrelevant goldens green
- [ ] Edge matrix above locked in Catch2
- [ ] `%peak ≤ 100` after traffic-model pass; Spec not lowered to quiet max
- [ ] Style: no namespaces; no `phase*` / Commit-N labels in tree
- [ ] Campaign wall never used as Done evidence
- [ ] Base contract [`theory-hist-transpile.md`](theory-hist-transpile.md) S0–S5 peak gates still hold for the **emitted** strategy

---

## Related

| Doc | Role |
|-----|------|
| [`theory-hist-transpile.md`](theory-hist-transpile.md) | Base S0–S5 Done / soft-fallback contract |
| [`cuda-throughput.md`](cuda-throughput.md) | Spec ceilings; Kernel SLO vs wall |
| [`cuda-profile-theory.md`](cuda-profile-theory.md) | nsys/ncu recipes; progress log |
| [`python-transpiler.md`](python-transpiler.md) | DSL compile; stream twin vs fused hist |
| [`theory-artifact.md`](../spec/theory-artifact.md) | `hist/` / `hist_plan` / `hist_module` |
| [`profiles/kernel_slo/SUMMARY.md`](profiles/kernel_slo/SUMMARY.md) | Metric B quiet ACCEPTANCE |
| [`profiles/dsl_smart/`](profiles/dsl_smart/) | Customs-without-presets ACCEPTANCE (when present) |
| [`include/parcae/dsl/README.md`](../../include/parcae/dsl/README.md) | Header map |
| [`hist_fast.hpp`](../../Parcae/Parcae/cuda/hist_fast.hpp) | Shared hist decode primitives |
