# Kernel SLO ACCEPTANCE — metric B (requiet 2026-10-06)

Quiet fair Kernel SLO vs physical DRAM roof. Default fused hist @ **1 B/rune →
896B**; Atbash/totient occupancy pad uses shared-cipher class → **≈80.7 TB**
(ncu 0.01111 B/rune — [`../traffic_model/SUMMARY.md`](../traffic_model/SUMMARY.md)).

**Hardware:** NVIDIA GeForce RTX 5070 Ti (sm_120)  
**Tool:** `parcae-bench --suite theory --allow-cuda --json` and
`parcae-bench --suite slo --extended --allow-cuda --json`  
**Protocol:** `BenchTimer` cudaEvent (4 warmups + median-of-3), setup/H2D excluded  
**Playbook:** [`cuda-profile-theory.md`](../../cuda-profile-theory.md)

**idle GPU ≠ Kernel SLO PCIe** — metric B is kernel rates only.

## Gates

| Gate | Rule | Applies when |
|------|------|--------------|
| **Stretch** | ≥**80%** of shape `estimated_peak` | Caesar / S1 / Atbash-class quiet fair |
| **Done** | ≥**90%** of shape `estimated_peak` | Shape is **DRAM-bound** (ncu DRAM SoL high) |
| **Model** | `%peak` must stay ≤100 | Always — if >, fix bytes/rune model, do **not** lower Spec |

Prior ncu on hist kernels: DRAM SoL ~**1–5%** (Atbash/totient ~0.4–1%) ⇒ not
DRAM-bound yet. Done is therefore **not required** for this plate; stretch is
the active ACCEPTANCE bar for unique-key shapes. Shared-cipher Atbash/totient
sit at ~**2%** of their physical roof (compute-bound; model OK).

## Theory fair (3 quiet runs → median) — plate 2026-10-03

Sources: `theory_fair_run{1,2,3}.json` (`T=1048576`). Unchanged by traffic-model
Spec (still 896B @ 1 B/rune).

| Row | run1 | run2 | run3 | **median** | % of 896B | Stretch | Done | Notes |
|-----|------|------|------|------------|-----------|---------|------|-------|
| `T.theory.compare_caesar` | 849.3B | 782.9B | 780.1B | **782.9B** | **87.4%** | **PASS** | 1/3 pass (run1 94.8%) | fat-64 catalog twin |
| `T.theory.caesar_bytecode` | 679.9B | 693.4B | 680.8B | **680.8B** | **76.0%** | **fail** | fail | `specialize_S1`; stable ~76–77% |
| `T.theory.s1_lut29` | 303.4B | 487.0B | 692.2B | **487.0B** | **54.4%** | **fail** | fail | high run-to-run noise |
| `T.theory.progressive` (S2 @841) | 401.6B | 394.0B | 403.5B | **401.6B** | **44.8%** | fail | fail | ks29; still compute-bound |

## Catalog SLO extended (requiet 2026-10-06, 2 quiet runs)

Sources: `slo_extended.json`, `slo_extended_run2.json`.

| Row | run1 | run2 | Stretch (≥80%) | Done (≥90%) | Model |
|-----|------|------|----------------|-------------|-------|
| T1 Caesar | 759.3B (85%) | 600.2B (67%) | borderline | fail | OK |
| F.atbash | 1600B (**1.98%** of 80.7 TB) | 1437B (**1.78%**) | fail (compute-bound) | fail | **OK** (`%peak≤100`) |
| F.affine | 819B (91%) | 771B (86%) | stretch / stretch | Done* / fail | OK this plate |
| F.vigenere | 835B (93%) | 857B (96%) | **PASS** | **PASS** | OK under 1 B/rune |
| F.beaufort | 854B (95%) | 872B (97%) | **PASS** | **PASS** | OK |
| F.totient | 1109B (**1.37%** of 80.7 TB) | 1091B (**1.35%**) | fail (compute-bound) | fail | **OK** (`%peak≤100`) |
| C.koan1_fused | 720B (80%) | 575B (64%) | borderline | fail | OK |
| T2 / T3 | — | — | fail | fail | T3 roof 448B; not hist focus |

\*Done claims require DRAM-bound (ncu). Affine can still print intermittent
`%peak>100` on other plates — unique-key 1 B/rune Spec unchanged.

## ACCEPTANCE verdict (metric B)

| Claim | Result |
|-------|--------|
| Stretch ≥80% — Caesar-class (`compare_caesar` / T1) | **PASS** (median twin **87%**; T1 often ~80%) |
| Stretch ≥80% — S1 / fair specialize | **FAIL** (median specialize **76%**; S1 noisy) |
| Model — Atbash / totient `%peak≤100` | **PASS** — shared-cipher occupancy roof (~80.7 TB) |
| Stretch / Done — Atbash-class vs physical roof | **FAIL / N/A** — ~2% of roof; DRAM SoL ≪ bound |
| Done ≥90% while DRAM-bound | **N/A** — hist still compute-bound |
| Spec lowered to quiet max | **No** — bytes/rune from ncu; Spec stays physical |

**Overall: PARTIAL PASS**

Landed: Caesar catalog twin stretch; F.vigenere/beaufort ≥90% under 1 B/rune;
Atbash/totient **model gate** after traffic-model pass. Open: S1/specialize
stretch (≥80%), S2 roof climb, Atbash/totient DRAM-bound climb before Done.

## Artifacts

| File | Role |
|------|------|
| `theory_fair_run{1,2,3}.json` | Quiet theory suite (3×, 2026-10-03) |
| `theory_fair.json` | Copy of run3 (representative hot S1) |
| `theory_fair.txt` | Human table (pre-JSON capture) |
| `slo_extended.json` / `slo_extended_run2.json` | Quiet SLO extended requiet (2×, 2026-10-06) |
| `slo_extended.txt` | Human table |

## Next

1. Stabilize S1 fair ≥**716.8B** (stretch) — same fat-64 path as specialize.
2. Atbash/totient DRAM climb (ncu SoL) before claiming Done on shared-cipher roof.
3. Combined write-up: [`gpu_full_load/SUMMARY.md`](../gpu_full_load/SUMMARY.md)
   (metric A duty + metric B roof).
