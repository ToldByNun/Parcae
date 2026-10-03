# Kernel SLO ACCEPTANCE — metric B (2026-10-03, RTX 5070 Ti)

Quiet fair Kernel SLO vs physical DRAM roof **896B** runes/s
(@ 1 B cipher/rune Spec model). Separate from export duty (metric A).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Tool:** `parcae-bench --suite theory --allow-cuda --json` and
`parcae-bench --suite slo --extended --allow-cuda --json`  
**Protocol:** `BenchTimer` cudaEvent (4 warmups + median-of-3), setup/H2D excluded  
**Playbook:** [`cuda-profile-theory.md`](../../cuda-profile-theory.md)

**idle GPU ≠ Kernel SLO PCIe** — metric B is kernel rates only.

## Gates

| Gate | Rule | Applies when |
|------|------|--------------|
| **Stretch** | ≥**80%** of 896B (≈716.8B) | Caesar / S1 / Atbash-class quiet fair |
| **Done** | ≥**90%** of 896B (≈806.4B) | Shape is **DRAM-bound** (ncu DRAM SoL high) |
| **Model** | `%peak` must stay ≤100 | Always — if >, fix bytes/rune model, do **not** lower Spec |

Prior ncu on hist kernels: DRAM SoL ~**1–5%** ⇒ not DRAM-bound yet. Done is
therefore **not required** for this plate; stretch is the active ACCEPTANCE bar.
Shapes with `%peak > 100` are **model defects**, not super-linear silicon.

## Theory fair (3 quiet runs → median)

Sources: `theory_fair_run{1,2,3}.json` (`T=1048576`).

| Row | run1 | run2 | run3 | **median** | % of 896B | Stretch | Done | Notes |
|-----|------|------|------|------------|-----------|---------|------|-------|
| `T.theory.compare_caesar` | 849.3B | 782.9B | 780.1B | **782.9B** | **87.4%** | **PASS** | 1/3 pass (run1 94.8%) | fat-64 catalog twin |
| `T.theory.caesar_bytecode` | 679.9B | 693.4B | 680.8B | **680.8B** | **76.0%** | **fail** | fail | `specialize_S1`; stable ~76–77% |
| `T.theory.s1_lut29` | 303.4B | 487.0B | 692.2B | **487.0B** | **54.4%** | **fail** | fail | high run-to-run noise |
| `T.theory.progressive` (S2 @841) | 401.6B | 394.0B | 403.5B | **401.6B** | **44.8%** | fail | fail | ks29; still compute-bound |

Best single-plate S1/specialize earlier today (`theory_fair_specialize` path):
~**707B** (~79%) — still just under stretch.

## Catalog SLO extended (2 quiet runs)

Sources: `slo_extended.json`, `slo_extended_run2.json`.

| Row | run1 | run2 | Stretch (≥80%) | Done (≥90%) | Model |
|-----|------|------|----------------|-------------|-------|
| T1 Caesar | 743.7B (83%) | 700.0B (78%) | borderline | fail | OK |
| F.atbash | 1392B (155%) | 1428B (159%) | **invalid** | **invalid** | **`%peak>100` — fix traffic model** |
| F.affine | 944B (105%) | 841B (94%) | stretch / Done* | *run2 Done; run1 model break | intermittent `>100` |
| F.vigenere | 846B (94%) | 868B (97%) | **PASS** | **PASS** | OK under 1 B/rune |
| F.beaufort | 840B (94%) | 843B (94%) | **PASS** | **PASS** | OK |
| F.totient | 1105B (123%) | 1079B (120%) | **invalid** | **invalid** | **`%peak>100` — fix traffic model** |
| C.koan1_fused | ~608B (68%) | ~611B (68%) | fail | fail | OK |
| T2 / T3 | — | — | fail | fail | T3 roof 448B; not hist focus |

\*Done claims on F.* with `%peak>100` are rejected even if `pass_tier` printed true.

## ACCEPTANCE verdict (metric B)

| Claim | Result |
|-------|--------|
| Stretch ≥80% — Caesar-class (`compare_caesar` / T1) | **PASS** (median twin **87%**; T1 often ~80%) |
| Stretch ≥80% — S1 / fair specialize | **FAIL** (median specialize **76%**; S1 noisy) |
| Stretch / Done — Atbash-class | **BLOCKED** — Atbash/totient `%peak>100` ⇒ revisit bytes/rune |
| Done ≥90% while DRAM-bound | **N/A** — hist still compute-bound (DRAM SoL ≪ roof) |
| Spec lowered to quiet max | **No** — Spec stays **896B** |

**Overall: PARTIAL PASS**

Landed: Caesar catalog twin stretch; several F.* (vigenere/beaufort) ≥90% under
the current model. Open: S1/specialize stretch (≥80%), S2 roof climb, and a
traffic-model pass for Atbash/totient (and any other `%peak>100` row) before
calling those shapes Done.

## Artifacts

| File | Role |
|------|------|
| `theory_fair_run{1,2,3}.json` | Quiet theory suite (3×) |
| `theory_fair.json` | Copy of run3 (representative hot S1) |
| `theory_fair.txt` | Human table (pre-JSON capture) |
| `slo_extended.json` / `slo_extended_run2.json` | Quiet SLO extended (2×) |
| `slo_extended.txt` | Human table |

## Next

1. Stabilize S1 fair ≥**716.8B** (stretch) — same fat-64 path as specialize.
2. Fix Atbash/totient traffic model (L2 / effective bytes per rune) so `%peak≤100`.
3. Combined write-up: [`gpu_full_load/SUMMARY.md`](../gpu_full_load/SUMMARY.md)
   (metric A duty + metric B roof).
