# Theory hist transpile contract (S0–S5)

**Status:** normative architecture contract for theory fused-χ² search throughput  
**Hardware reference:** RTX 5070 Ti (sm_120) — same plate as [`cuda-throughput.md`](cuda-throughput.md)  
**Canonical peaks / pass rule:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp)  
**Fair Kernel SLO tool:** `parcae-bench --suite theory --allow-cuda`  
**Profiling playbook:** [`cuda-profile-theory.md`](cuda-profile-theory.md)  
**Emit / classify:** [`theory_hist_chi2_emit.hpp`](../../include/parcae/dsl/theory_hist_chi2_emit.hpp)  
**Launch façade:** [`theory_hist_chi2_launch.hpp`](../../Parcae/Parcae/cuda/theory_hist_chi2_launch.hpp)

This document freezes **what “done” means** for theory search fused χ²: every
hist strategy that actually runs — **including S0 bytecode** — must meet the
**≥90% shape-peak** Kernel SLO gate on a quiet 5070 Ti. Soft-fallback to S0 is
allowed for coverage; it is **not** an exemption from the S0 peak gate.

Agents implement and test only; they do **not** create git commits or tags.

---

## 1. PRIMARY gate (binding)

```text
done(strategy) ⇔  fair_kernel_runes_per_s >= 0.90 * BenchTierSpec.estimated_peak(shape_id)
                  AND  fair_kernel_runes_per_s >= BenchTierSpec.slo_floor(shape_id)
```

| Rule | Detail |
|------|--------|
| Metric | `BenchTimer`: 4 warmups + **median-of-3** `cudaEvent`; setup / H2D / D2H / host prepare **excluded** |
| Fair grid | Token length **`T ≥ 2^20`** (same floor as `BenchTierSpec::fair_gate_tokens()`). Shorter T is underfill — measurement-only, **not** a fail gate |
| Shape id | Spec row for the **strategy that actually launched** (S0 / S1 / S2 / …), not “wishful” specialized id after soft fallback |
| Pass math | Identical to `BenchTierSpec::pass_tier` (`peak_band_pct = 90`, raw band 89.5 so printed 90% matches) |
| Catalog stretch | Document `%` of catalog ceilings (`F.atbash` 550B, Caesar twin, …). Stretch **never** replaces PRIMARY |
| Campaign wall | `research/run.log` / scheduler wall / short page `T` — **ops diary only**, never PRIMARY |

### S0 is not a free pass

| Situation | Score correctness | Throughput Done? |
|-----------|-------------------|------------------|
| Soft-fallback S0 after S1/S2/S3 emit or module load fail | Must match CPU/bytecode oracle | PRIMARY still requires **S0 shape peak ≥90%** via fair suite |
| Theory classified hard-S0 (autokey / prefer_branch / caps) today | Oracle parity | Same — S0 fair row must pass; specialize-away is a separate climb |
| Fair S0 today ~69–75B vs Spec 75B (≥90%) | OK | **pass_tier** after climb pass 1 (tiles/trusted/residency) |

Two parallel workstreams (implementation roadmap, not doc names):

1. **S0 climb** — tune `TheoryChi2Batch` (and remesaure Spec peak honestly) until `T.theory.caesar_bytecode` **pass_tier**. Repro: [`capture_s0_climb_baseline.ps1`](../../scripts/cuda/capture_s0_climb_baseline.ps1) + [`profiles/s0-climb/SUMMARY.md`](profiles/s0-climb/SUMMARY.md).
2. **Specialize-away** — S3/S4/S5 (+ artifact/module) so customs leave S0 for higher absolute runes/s, without dropping the S0 gate.

---

## 2. Two metrics (do not mix)

| Name | Formula / tool | Includes | Gate? |
|------|----------------|----------|-------|
| **Kernel SLO** | `repeats × C × T / median cudaEvent` via `parcae-bench --suite theory` | Device hist + finalize only | **Yes** — PRIMARY |
| **Campaign wall** | `runes_work / wall` over `SearchScheduler` (`research/run.log`) | Host prepare/bind, H2D, kernel, D2H, materialize, ingest | **No** |

Do **not** compare LP2 page-length wall rates to fair `T≈2^20` peaks.

Optional interim `checkpoint_50B` (when Spec peak ≫ 50B) is annotation only — see `BenchTierSpec::checkpoint_50B_*`. It never replaces ≥90% peak.

---

## 3. Strategy ladder (S0–S5)

Classify decrypt HotLoop via `TheoryHistChi2Emit::select_strategy` / emit. Prefer specialized in `GpuCandidateExport`; soft-fallback S0 keeps `export_backend=cuda`.

| Id | Enum / Spec (planned or shipping) | When | Runtime twin (shipping / target) | Shape peak class (5070 Ti) | ≥90% PRIMARY |
|----|-----------------------------------|------|----------------------------------|----------------------------|--------------|
| **S0** | `S0Bytecode` / `T.theory.caesar_bytecode` (+ optional autokey-shaped row) | Soft-fallback; caps; unmatched; (today) autokey / prefer_branch | `TheoryChi2Batch` | ~75B (remeasure if ceiling moves) | **Required** |
| **S1** | `S1Lut29` / `T.theory.s1_lut29` | Decrypt `f(x; params)` only — no stream `i`, no autokey | `TheoryHistChi2S1` | ~420B | Required |
| **S2** | `S2Uchar4Inline` / `T.theory.s2_linear` / `T.theory.progressive` | `x ± (b0 + b1·i)` (+ widened linear family over time) | `TheoryHistChi2S2` | ~200B+ | Required |
| **S3** | `S3ScalarInline` / planned `T.theory.s3_*` | Uses `i` / general `x ± g(i; params)` without autokey; not simple linear S2 | Planned expr-inline hist / module | Calibrate 150–350B | Required once Spec row exists |
| **S4** | planned / `T.theory.s4_autokey` | `z29_autokey_shift` shapes (e.g. vigenere_lag) | Planned `TheoryHistChi2S4` AutokeyRing+hist | Calibrate 80–250B | Required once Spec row exists |
| **S5** | planned / poly keystream Spec | Low-degree poly / bitmask-like keystreams beyond `b0+b1·i` | Twin or S3 module | Form peak | Required once Spec row exists |

### Shipping vs planned (honest snapshot)

| Strategy | Classify | Emit specialized sources | In-lib / module launch | Fair ≥90% |
|----------|----------|--------------------------|------------------------|-----------|
| S0 | yes | n/a (bytecode) | yes (`TheoryChi2Batch`) | **pass** after climb pass 1 (~92–100% of 75B) |
| S1 | yes | yes | yes | **pass** (~94% of 420B) |
| S2 | yes (linear match) | yes when `match_s2_linear` | yes | **pass** (~92% of 200B) |
| S3 | yes | **stub → soft S0** | no | blocked on S0 until emit+runtime |
| S4 | **hard S0 today** | no | no | blocked on S0 / future S4 peak |
| S5 | partial (S2 classify, emit often soft S0) | no | no | blocked until match/emit |

Non-linear `x ± g(i)` (e.g. quadratic) may **classify** S2 then soft-fall S0 when linear match fails — treat as S3/S5 work, not as “S2 done.”

---

## 4. Prefer order & soft fallback

Intended export prefer order (as strategies land):

```text
S1 → S2 → S5 → S3 (module/artifact) → S4 → S0
```

Rules:

- Soft-fallback **must** preserve χ² / top-k parity vs CPU bytecode oracle (fixed grids).
- Domain errors (e.g. Div0): lane `+inf` (same contract as `TheoryChi2Batch`).
- Non-empty interrupt policy: **hard reject** export (already).
- Missing cubin / module load fail: soft S0 + diagnostic; S0 PRIMARY gate still applies.
- `TheoryHistChi2Launch::has_specialized` today always `false` (shared in-lib S1/S2 via plans, not per-URI modules). Future artifact/module work must not invent C++ namespaces.

---

## 5. Two compile products (do not conflate)

| Product | Producer | Consumer | Role |
|---------|----------|----------|------|
| **Stream twin** | `DslEmitCuda` @ `parcae-compile` | Transform / apply | Bit-identity 1D device apply |
| **Fused hist** | `TheoryHistChi2Emit` (+ future artifact/module) | `GpuCandidateExport` / `TheoryExportCache` | Search χ² Kernel SLO |

Normative artifact layout and manifest fields:
[`theory-artifact.md`](../spec/theory-artifact.md) § Directory layout / § `paths` /
§ `hist` / § `hist/hist_plan.json`.

| On disk (under `data/theories/<name>/<ver>/`) | Product |
|-----------------------------------------------|---------|
| `emitted/*Kernel.{hpp,cu}` via `paths.cuda_*` | Stream twin only |
| `hist/hist_plan.json`, optional `hist/*.{cu,cubin}` via `paths.hist_*` | Fused hist (optional until writers land) |

**Shipping today:** compile persists **stream** twins. Search hist is mostly
**runtime** classify + in-lib S1/S2 (`TheoryExportCache` keeps plans; EmitBundle
hist source text is not the launch path). `hist: null` / missing `paths.hist_*`
is valid. Kernel SLO Done stays tied to **hist strategies** in this contract —
never to stream-twin throughput alone.

---

## 6. Style (HARD)

- **No** C++ `namespace` (including anonymous) in new or touched theory-hist CUDA/C++.
- One top-level `class Name` per header; `#ifndef NAME_HPP` / `#endif // NAME_HPP`.
- File-scope `__global__` kernels + class façades (same rule as search/CUDA climb workstreams).
- No `phase*` substrings in filenames, doc titles, profile tags, CI job names, or Spec ids.

```cpp
#ifndef NAME_HPP
#define NAME_HPP
class Name {
public:
  // …
private:
  Name() = delete; // when static-only
};
#endif // NAME_HPP
```

---

## 7. Edgecases (contract)

| Case | Required behavior |
|------|-------------------|
| Autokey HotLoop | Today S0; target S4 when matched. Scores must match oracle either way |
| Div0 / inv domain | Lane `+inf`; soft S0 if specialized path cannot bind |
| Non-empty interrupt | Reject export |
| Empty interrupt | Allowed |
| `prefer_branch` Select | S0 until a measured divergent specialized path exists; S0 peak gate still holds |
| Decrypt hoists | Must not silently wrong-score; wire into S1/S3 or soft S0 with parity |
| Caps (ops/stack/slots/C/T) | S0 or stable `DslRuleId` reject — no UB |
| Top-k parity | CPU bytecode ≡ specialized (and ≡ S0) on fixed grids |
| Short T / campaign | Document as wall only |

Catch2 anchors (extend as strategies land): `[cuda][theory][edge]`, `[cuda][golden]`, `[search][export][theory]`, `[bench][theory]`, `[bench][spec]`.

---

## 8. Peak calibration rule

For each shape id (including S0 and future S3/S4/S5):

1. Quiet GPU; fair `T≥2^20`; record cudaEvent median (and ncu when counters allowed).
2. Set `BenchTierSpec.estimated_peak` slightly **above** observed max (round up).
3. Confirm subsequent fair runs print `%peak` roughly **90–99**, not permanently &lt;90 and not routinely &gt;100.
4. Append a row to the progress table in [`cuda-profile-theory.md`](cuda-profile-theory.md). Profile dirs use descriptive tags (`s0_climb`, `s3_expr`, `s4_autokey`) — never `phase*`.

If S0 hardware ceiling is honestly ~70B, **lower** the Spec peak and hold ≥90% of that — do not leave a 75B ceiling that never passes. If climb exceeds 75B, raise the peak.

Optional separate Spec row for **autokey-as-S0** if Caesar-as-bytecode peak is an unfair gate for lag HotLoops until S4 ships.

---

## 9. Exit checklist (this contract)

Declare the theory-hist transpile throughput workstream complete only when **all** apply:

- [ ] **S0** fair suite row(s) **pass** `pass_tier` (≥90% S0 Spec peak)
- [ ] **S1** and **S2** continue to pass their Spec peaks
- [ ] **S3 / S4 / S5** each have Spec peaks and pass ≥90% once implemented
- [ ] Soft-fallback remains correct (parity + Div0 +inf + interrupt reject)
- [ ] Customs without presets: specialized when eligible; otherwise S0 with S0 gate green
- [ ] Progress log has nsys/ncu (or documented counter-permission skip) for each Perf milestone
- [ ] Style: no namespaces; no `phase*` names
- [ ] Campaign wall never used as Done evidence

---

## Related

| Doc | Role |
|-----|------|
| [`cuda-throughput.md`](cuda-throughput.md) | Catalog + theory Spec ceilings table |
| [`cuda-profile-theory.md`](cuda-profile-theory.md) | nsys/ncu recipes; progress log |
| [`python-transpiler.md`](python-transpiler.md) | DSL compile + hist strategy summary |
| [`search-handbook.md`](search-handbook.md) | Operator theory URI + CUDA flags |
| [`theory-artifact.md`](../spec/theory-artifact.md) | Artifact layout: stream `emitted/` vs fused-hist `hist/` + manifest fields |
| [`include/parcae/dsl/README.md`](../../include/parcae/dsl/README.md) | Header map |
| [`include/parcae/bench/bench_tier_spec.hpp`](../../include/parcae/bench/bench_tier_spec.hpp) | Single source of peak / `pass_tier` |
