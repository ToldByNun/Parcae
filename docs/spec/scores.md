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

Log-probability under a bigram model. Model file optional; if missing, tool
returns `Status` not-implemented (MUST NOT crash).

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
```

Each `score_id` MUST declare `order: "asc" | "desc"`.

| score_id | order |
|----------|-------|
| `exact_match` | desc |
| `hamming_agreement` | desc |
| `ic_mod29` | desc (typical plaintext preference — document; tests lock behavior) |
| `chi2_english_gp_v0` | asc |
| `self_repeat_rate` | neither assumed globally — report raw; no default reject |

---

## CLI / tool surface

`parcae-score --score-id ic_mod29 --input ...` prints a single floating value and
exits nonzero on error. Machine-readable JSON mode SHOULD exist:

```json
{ "score_id": "ic_mod29", "score_version": "v0", "value": 0.0412 }
```
