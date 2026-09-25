# Spec: Score Suite

**Status:** Normative  
**Headers (planned):** `parcae/score/*.hpp`  
**Research:** [hypotheses.md](../research/hypotheses.md)

## Design rules

1. Scores are **pure** functions of an `Index29` sequence (+ optional tables).
2. Same input ⇒ same IEEE bit pattern for a given `score_id` and version
   (document NaN policy: scores MUST NOT return NaN on empty without an error).
3. Empty input: return `Status` error (do not return 0 silently) unless a score
   explicitly defines empty behavior.
4. Scores never mutate inputs.

Registry key: string `score_id` + semver-ish `score_version` (initial: `"v0"`).

---

## Tier A — Required

### `exact_match`

| Field | Value |
|-------|-------|
| Inputs | candidate indices, reference indices |
| Output | `1.0` if equal length and all equal; else `0.0` |
| Use | Fixture validation |

### `hamming_agreement`

| Field | Value |
|-------|-------|
| Output | `matches / length` in `[0,1]` |
| Mismatched lengths | hard error |

### `ic_mod29`

Index of coincidence over 29 symbols:

\[
IC = \frac{\sum_{c=0}^{28} n_c(n_c-1)}{N(N-1)}
\]

for \(N \ge 2\). For \(N < 2\), hard error.

### `chi2_english_gp_v0`

Chi-square against an expected frequency table derived from **concatenated solved
fixture plaintext** (built from locked fixtures only).

\[
\chi^2 = \sum_{c=0}^{28} \frac{(o_c - e_c)^2}{e_c}
\]

Table file (planned): `data/profiles/scores/english-gp-expected-v0.json` with
provenance hash. Until built, score id exists but tests MAY use a synthetic table.

Lower is “closer” to English-GP; document whether CLI prints raw χ² or a
transformed score (pick one and lock it in tests).

### `self_repeat_rate`

Adjacent equal-rune rate:

\[
R = \frac{|\{i : x_i = x_{i+1}\}|}{N-1}
\]

for \(N \ge 2\).

This is a **generic statistic**. It is useful for exploration and negative
controls. It is **not** a normative endorsement of any external essay’s claimed
global Liber Primus value.

---

## Tier B — Optional hooks (API reserved)

### `log_bigram_gp_v0`

Log-probability under a conditional bigram model over Index29 (alphabet size 29).

| Field | Value |
|-------|-------|
| Inputs | candidate indices \(x_0,\ldots,x_{N-1}\), bigram model table \(L\) |
| Output | \(S = \sum_{i=0}^{N-2} L[x_i,\ x_{i+1}]\) (IEEE `double`) |
| Order | `desc` (higher \(S\) is better) |
| \(N < 2\) or empty | hard `Status` error (same policy as `ic_mod29` / χ²) |

\[
S = \sum_{i=0}^{N-2} L[x_i,\ x_{i+1}]
\]

**Table layout:** \(L\) is a \(29 \times 29\) matrix in **row-major** order: entry
\((a,b)\) is at index `a * 29 + b` (841 entries). This matches the CUDA bigram
index convention `bigram_s[y0 * 29 + y1]`.

**Model construction (normative for `english-gp-bigram-v0`):**

1. Count raw bigrams \(c[a][b]\) over consecutive consumable Index29 symbols from
   the **same locked fixture plaintexts** used by `english-gp-expected-v0`
   (`source_fixture_ids` MUST match that Unigram profile).
2. Add-one smoothing: \(\tilde{c}[a][b] = c[a][b] + 1\), row sum
   \(\tilde{Z}_a = \sum_{b=0}^{28} \tilde{c}[a][b]\).
3. Conditional log-probability with **natural log**:
   \(L[a,b] = \ln(\tilde{c}[a][b] / \tilde{Z}_a)\). Persist `log_base: "ln"` in
   the model JSON.

**Table file:** `data/profiles/scores/english-gp-bigram-v0.json` (schema
`parcae.bigram_model.v0`) with `raw_counts` (841 ints), `log_probs` (841
doubles), provenance, and a sibling `.sha256`. Until the file exists in-tree,
the score id MAY be registered but MUST fail with a clear `Status` whose message
includes `not implemented` or `model missing` when the model cannot be loaded
(MUST NOT abort; MUST NOT silently fall back to χ² or another score).

**Registry vs DeepScoreBatch sign (locked):**

- `ScoreRegistry` / `log_bigram_gp_v0` returns **\(S\)** (sum of log-probs) and
  declares **`order: desc`**.
- The fused CUDA throughput path `DeepScoreBatch` (Caesar bigram LL) may store
  **\(-S\)** in its score buffer for Asc-compatible top-k with synthetic tables.
  That kernel convention is **not** the registry contract. Implementations MUST
  NOT “fix” DeepScoreBatch `-S` by changing registry order or by negating
  registry output to match the bench kernel.

**Search path (see [`search-loop.md`](search-loop.md)):** normal Liber Primus
candidate export scores this id through the **CPU** `ScoreRegistry` /
`RankCandidates` path. Fused `GpuCandidateExport` remains **χ²-only**. When a
`SearchJob` requests `backend=cuda` with `log_bigram_gp_v0` (or any non-χ²
score), the scheduler MUST fall back to CPU export rather than hard-erroring or
silently scoring χ² instead.

### `dictionary_hit_ratio_v0`

Fraction of segmented Latin words found in a word list. Word list optional.

---

## Tier C — Explicitly non-normative defaults

Any score or reject rule that hard-codes external claims such as
“self-follow must be ≈ 0.664%” or “additive Vigenère is impossible on 0–55”
is **out of scope** for required scores. See research Tier C
([hypotheses.md](../research/hypotheses.md)).

Implementations MAY later add an optional score
`self_repeat_distance_to_target` with an explicit params object:

```json
{ "target_rate": 0.00664, "source_note": "unverified:artwaste" }
```

but it MUST default **off** and MUST carry the unverified provenance string.

---

## Batch / top-k

Bounded batch runner:

```text
for each candidate:
  score = registry[score_id](candidate)
keep top-k by score with deterministic tie-break:
  1) better score (define higher-is-better vs lower-is-better per score_id)
  2) lexicographically smaller candidate_id
  3) smaller source_index (input order) — absolute total order for parity
```

Each `score_id` MUST declare `order: "asc" | "desc"`.

| score_id | order |
|----------|-------|
| `exact_match` | desc |
| `hamming_agreement` | desc |
| `ic_mod29` | desc (typical plaintext preference — document; tests lock behavior) |
| `chi2_english_gp_v0` | asc |
| `self_repeat_rate` | neither assumed globally — report raw; no default reject |
| `log_bigram_gp_v0` | desc (higher log-prob sum is better; see Tier B) |

### Batch parallelism (CPU)

| Mode | Score evaluation | Top-k reduction |
|------|------------------|-----------------|
| `serial` (default) | Input order, calling thread | `BatchOrdering` sort |
| `parallel` | Chunked `std::async` pool; workers write disjoint `scores[i]` | **Same** single-thread `BatchOrdering` sort |

**Parity rule:** multi-thread / future `std::execution` / CUDA batches MAY compute
scores concurrently, but MUST NOT use schedule-dependent heaps or “first finish
wins” insertion. Materialize `scores[0..N)`, then reduce with the total order
above. Serial CPU remains the source of truth (`docs/spec/parity.md`).

CUDA score **per-stream** floating-point / histogram reduction order is locked in
[`docs/architecture/cuda-score-reduction.md`](../architecture/cuda-score-reduction.md)
(integer accumulate → fixed-order FP finalize; no unordered `double` atomics).

---

## CLI / tool surface

`parcae-score --score-id ic_mod29 --input ...` prints a single floating value and
exits nonzero on error. Machine-readable JSON mode SHOULD exist:

```json
{ "score_id": "ic_mod29", "score_version": "v0", "value": 0.0412 }
```
