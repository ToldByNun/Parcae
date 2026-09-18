# CUDA score reduction associativity

**Status:** Normative for score twins (roadmap commit 21; implements commits 22–27)  
**CPU truth:** [`include/parcae/score/`](../../include/parcae/score/), [`BatchOrdering`](../../include/parcae/batch/batch_ordering.hpp)  
**Specs:** [`scores.md`](../spec/scores.md), [`parity.md`](../spec/parity.md)  
**ABI:** [`cuda-abi.md`](cuda-abi.md)

North star (scores):

```text
CPU_reference_score(stream, tables, …)
  ==  CUDA_twin_score(…)     (identical IEEE-754 binary64 bits)
```

Transforms already require byte-identical `Index29`. Scores add a second gate:
**floating-point reduction order**. This doc locks what may be parallelized and
what MUST stay serial / fixed-order before any CUDA score kernel claims parity.

---

## Two reduction layers

| Layer | What is reduced | Parity rule |
|-------|-----------------|-------------|
| **A — Per-stream score** | Accumulators inside one candidate’s Index29 stream → one `double` | Same algorithm + same finalization order as CPU headers |
| **B — Batch top-k** | `scores[0..C)` → ordered hits | Materialize all scores, then **single-thread** `BatchOrdering` (CPU or host after D2H) |

Layer B is already locked in [`scores.md`](../spec/scores.md) (serial / parallel
CPU table). CUDA MUST follow the same rule: never schedule-dependent heaps,
never “first finish wins” insertion into a top-k structure.

Layer A is what this doc specifies for CUDA twins.

---

## Hard rules (all score twins)

1. **No fast-math** on score compile units (`-ffast-math`, `--use_fast_math`,
   unsafe FP contraction). Matches [`parity.md`](../spec/parity.md).
2. **No unordered `atomicAdd` on `double`** for a parity-claimed path. Atomically
   summing FP partials in arrival order is **not** bit-identical across runs.
3. Prefer **integer / fixed-size accumulators**, then **one** (or a **fixed
   small loop of**) `double` operation(s) in a **documented index order**.
4. Empty / too-short inputs: return `Status` error — same as CPU (no silent `0.0`,
   no NaN).
5. CPU single-thread remains the source of truth. A CUDA score is wrong if it
   disagrees with `ScoreRegistry` / the Tier A class for the same bytes.

---

## Tier A — per-score associativity

Symbols: \(N\) = stream length, \(n_c\) = count of symbol \(c\in\{0,\ldots,28\}\).

### `exact_match`

| | |
|--|--|
| CPU | Length equal and all indices equal → `1.0`, else `0.0` |
| Parallelism | Elementwise compare; early-exit allowed **only if** result is still exact (any mismatch → `0.0`) |
| FP reduction | None (constants only) |
| Associativity | Trivial |

### `hamming_agreement`

| | |
|--|--|
| CPU | `matches / N` with integer `matches` |
| Parallelism | Partition the stream; **integer** sum of match counts (associative, commutative) |
| FP reduction | Single divide: `double(matches) / double(N)` after the full integer sum |
| Associativity | Integer sum is fine under any tree; **do not** sum `1.0/N` partial floats |

### `ic_mod29`

\[
IC = \frac{\sum_{c=0}^{28} n_c(n_c-1)}{N(N-1)}\quad (N\ge 2)
\]

| | |
|--|--|
| CPU | Histogram `uint64[29]`, then integer numerator, then one `double` divide |
| Parallelism | Per-block local histograms → **integer** reduce into 29 bins (tree/`atomicAdd` on `uint64` OK) |
| FP reduction | After global `n_c` known: `numerator = Σ n_c(n_c-1)` in **c = 0..28 order**, then `/ (N(N-1))` once |
| Associativity | Histogram merge is associative. **Do not** accumulate `n_c(n_c-1)/denom` as running floats |

### `self_repeat_rate`

\[
R = \frac{|\{i:x_i=x_{i+1}\}|}{N-1}\quad (N\ge 2)
\]

| | |
|--|--|
| CPU | Integer adjacent-equal count, one divide |
| Parallelism | Partition interior edges; care at **partition boundaries** (pair \((x_{b-1},x_b)\) must be counted exactly once). Integer sum of repeats |
| FP reduction | `double(repeats) / double(N-1)` once |
| Associativity | Same as hamming: integer first |

### `chi2_english_gp_v0`

\[
\chi^2 = \sum_{c=0}^{28} \frac{(o_c - e_c)^2}{e_c},\quad e_c = p_c\cdot N
\]

| | |
|--|--|
| CPU | Histogram `o_c`, then loop **c = 0..28**, accumulate `chi2 += (diff*diff)/e` in that order |
| Parallelism | Histogram with integer reduce (as IC). The **χ² sum is ordered FP** |
| FP reduction | **MUST** walk `c` from `0` to `28` inclusive, same as [`Chi2EnglishGp`](../../include/parcae/score/chi2_english_gp.hpp). No parallel FP tree over the 29 terms for v0 parity |
| Associativity | `(a+b)+c ≠ a+(b+c)` in binary64 — treat the 29-term sum as **non-associative** for parity |

v0 MAY compute χ² on the host after D2H of the 29-bin histogram if that is
simpler; device and host must still use the fixed `c` order.

---

## Allowed CUDA patterns (v0)

```text
stream  ──►  integer histogram / match counts  ──►  (optional D2H)
                 │
                 ▼
         fixed-order FP finalize  ──►  double score
```

| Pattern | Allowed for parity? |
|---------|---------------------|
| Parallel map over candidates (one score each), write `scores[i]` | Yes |
| Integer histogram reduce (shared mem + tree / `atomicAdd` u64) | Yes |
| Host finalize after D2H of counts | Yes |
| Warp/block FP tree reduce of partial χ² / IC terms | **No** (v0) |
| `atomicAdd` on `double` score accumulator | **No** |
| Top-k insert from racing blocks | **No** — use layer B |

---

## Batch layer B (reminder)

After `scores[0..C)` exists (CPU or CUDA):

1. Better score per `ScoreOrder` (`exact`/`hamming`/`ic` → desc; `chi2` → asc).
2. Lexicographically smaller `candidate_id`.
3. Smaller `source_index`.

Implement with a **deterministic** sort/select on the host in v0
([`cuda-abi.md`](cuda-abi.md): top-k MAY be host-side). Device top-k is a later
optimization only if it reproduces this total order bit-for-bit on the hit list
(not merely “same best score”).

---

## Test expectations (commits 22–27)

| Check | Requirement |
|-------|-------------|
| Unit | Each CUDA score twin vs CPU on synthetic Index29 streams |
| Solved / golden | Same streams as transform parity where useful; compare `memcmp` on `double` bits or exact `==` |
| Batch | Parallel score fill + `BatchOrdering` matches serial CPU top-k |
| Negative | Build with fast-math on score TU MUST NOT be a supported config |

Catch2 tags (planned): `[cuda][score]` / per-id subtags; suite gate in commit 27.

---

## Non-goals

- Changing Tier A formulas or `ScoreOrder` tables
- Claiming bit-identical results under `--use_fast_math`
- Replacing CPU `[solved]` / score registry as the reference
- Device-side UTF-8 / expected-frequency JSON parse (tables stay host-loaded POD)

---

## Doc map

| Doc | Role |
|-----|------|
| This file | Score FP / histogram reduction policy for CUDA |
| [`scores.md`](../spec/scores.md) | Tier A formulas + batch top-k |
| [`parity.md`](../spec/parity.md) | Equality layers (Index29 / scores / Latin) |
| [`cuda-abi.md`](cuda-abi.md) | `scores[C]` buffer; host top-k |
| [`cuda-handoff.md`](cuda-handoff.md) | Twin checklist |
| [`cuda-roadmap.md`](cuda-roadmap.md) | Commits 22–27 |
