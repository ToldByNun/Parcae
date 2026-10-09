# Hist alphabet remap (Mono / Spalten / Lag / Bigram)

**Status:** normative traffic + math contract for production fused χ² after
once-count remaps  
**Spec peaks / dual rates:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp),
[`BenchMetric`](../../include/parcae/bench/bench_metric.hpp)  
**Host math:** [`HistAlphabetMap`](../../include/parcae/score/hist_alphabet_map.hpp)  
**Device primitives:** `CipherHistOnce`, `ColumnHistOnce`, `LagDiffHistOnce`,
`BigramCountOnce`, `AlphabetChi2Batch` under [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/)  
**Throughput ceilings:** [`cuda-throughput.md`](cuda-throughput.md)

Production search no longer re-decodes every token for every candidate on
monoalphabetic / period-29 / lag / Caesar-bigram-LL shapes. It counts the
ciphertext **once**, then remaps bins (or bigram cells) per candidate.

Agents implement and test only; they do **not** create git commits or tags.

---

## 1. Why 93.5% of 896B was misleading

Quiet `roof_hist` plates (2026-10-03) printed Caesar twin **~838B runes/s ≈
93.5% of 896B** and labeled it **Done**. That percentage used the **1 B
cipher/rune DRAM model**:

```text
logical_runes/s = repeats × C × T / elapsed
estimated_peak  = 896e9 / 1 B/rune = 896B   # assumed each (c,t) streams 1 B
```

ncu `cache_bound` (2026-10-06) showed absolute DRAM only **tens of GB/s**
(~1–3% of 896 GB/s) and **bytes/rune ≪ 1** — cipher is L2-resident / shared
across lanes. Calling 93.5% “DRAM Done” conflated:

| Quantity | Meaning |
|----------|---------|
| **logical** `C·T` runes/s | How many candidate×token pairs were scored |
| **physical** cipher bytes/s | How much GDDR7 traffic the once-count actually moved |
| **896B Spec** | Valid only when traffic ≈ 1 B/logical-rune **and** L2-miss bound |

After remap, cipher traffic is **O(T)** (one hist / column / lag / bigram pass),
not **O(C·T)**. Logical RPS can exceed 896B without being memory-bound.
**Done gates for remap shapes use Remap roofs** (logical runes/s), not 896B.

Historical 93.5% / %-of-896B numbers remain useful **diary** of the decode-hist
era; they are **not** the current production Done model. See
[`profiles/roof_hist/SUMMARY.md`](profiles/roof_hist/SUMMARY.md) and
[`profiles/cache_bound/SUMMARY.md`](profiles/cache_bound/SUMMARY.md).

---

## 2. Dual metrics

```text
logical_runes_per_sec  = repeats × C × T / seconds     # BenchMetric (historical)
cipher_bytes_per_sec   = repeats × T × bytes_per_token / seconds
```

| Path | Typical `bytes_per_token` | Compare `cipher_B/s` to |
|------|---------------------------|-------------------------|
| Mono / column / lag remap | **1** | GDDR7 **896 GB/s** (diary) |
| Bigram once | **~2** | same |
| Legacy decode→hist (hard-S0) | **~1 × C** effective | 896B logical roof still OK |

`%peak` for PRIMARY uses **logical** RPS vs `BenchTierSpec::estimated_peak`
(Remap roof or 896B diary). Do **not** report `% of 896B` as Done for remap
shapes.

---

## 3. Remap roofs (Done / Stretch)

Interim freezes (identity-hist / CipherHistOnce plate class; re-calibrate on a
quiet remap plate when available):

| Constant | Value | Production shapes |
|----------|-------|-------------------|
| `kAlphabetRemapHistRoofRps` | **2.0 TB** | T1 Caesar, S1, ShapeInline mono, Affine remap |
| `kColumnRemapHistRoofRps` | **2.0 TB** | S2, S5, Vigenère, Beaufort |
| `kLagRemapHistRoofRps` | **2.0 TB** | T2 CTAK class, S4 AutokeyRing |
| `kBigramRemapHistRoofRps` | **1.0 TB** | T3 Caesar bigram-LL (Dict separate) |
| `kSharedCipherComputeRoofRps` | **2.0 TB** | F.atbash / F.totient (same numeric freeze) |
| `kDramRooflineHistPeak` | **896B** | hard-S0 / koan stages only |

Affine decode-era ncu (~0.01906 B/rune → ~47 TB) is **diary**, not Affine Remap
Done. Atbash DRAM occupancy (~80.7 TB) is diary; Done = compute / Remap roof.

---

## 4. Mono alphabet remap

**Once:** `CipherHistOnce` → `H[x] = |{t : in[t]=x}|` (29 bins).  
**Remap:** permute / rotate bins per candidate; finalize χ² on remapped `P`.

| Shape | Remap | Facade |
|-------|-------|--------|
| Caesar decrypt | `P[b] = H[(b+s) mod 29]` | `CaesarChi2Batch` / Shape / Alphabet |
| Atbash | `P[b] = H[28−b]` | `FamilyChi2Batch` / Shape |
| Atbash∘Caesar-encrypt | `P[b] = H[(28+s−b) mod 29]` | Family |
| Affine decrypt | `P[inv(a)·(x−b)] = H[x]` | Family / Shape |
| S1 mono-LUT | `P[lut[x]] += H[x]` (`C×29` rows) | `TheoryHistChi2S1` |

Host: `HistAlphabetMap::rotate_*`, `mirror_atbash`, `permute_affine_decrypt`,
`apply_bin_map`. Legacy decode→hist kernels remain for golden A/B only.

**Complexity:** O(T) hist + O(C·29) remap (vs O(C·T) decode-hist).

---

## 5. Spalten / period remap

**Once:** `ColumnHistOnce(L)` → `Col[j·29 + x]` counts `in[t]=x` with `t % L = j`.  
Interrupt-free only (`t % L`). Skips → **no remap**, legacy decode.

| Shape | Period | Remap | Facade |
|-------|--------|-------|--------|
| Vigenère decrypt | mixed L | `P[b] = Σ_j Col[j][(b+key[j]) mod 29]` | Family / Alphabet |
| Beaufort | mixed L | `P[b] = Σ_j Col[j][(key[j]−b) mod 29]` | Family / Alphabet |
| S2 linear | **L=29** | `ks[j]=b0+b1·j`; `out = in ∓ ks` | `TheoryHistChi2S2` |
| S5 poly | **L=29** | `ks[j]=b0+b1·j+b2·j²`; same ∓ | `TheoryHistChi2S5` |

Host: `vigenere_plain_hist_from_columns`, `beaufort_plain_hist_from_columns`,
`keystream_plain_hist_from_columns`, `linear_period29_*`, `poly_period29_*`.

**Complexity:** O(T) columns + O(C·L·29) remap; S2/S5 fixed L=29 → O(C·29²).

---

## 6. Lag-diff / Autokey remap

Dense CTAK / AutokeyRing without skips:

```text
t ≥ L:  out_t ≡ in_t − in_{t−L}   (or + for add form)
t < L:  prefix from primer (CTAK) or key=0 (Ring)
```

**Once:** `LagDiffHistOnce` (and lag-sum for add form) + prefix hist.  
**Merge:** `P[b] = D[b] + Prefix[b]` (`HistAlphabetMap::merge_lag_diff_and_prefix`).

| Shape | Semantics | Facade |
|-------|-----------|--------|
| CTAK | primer length L; lag-diff + primer prefix | `DeepScoreBatch` |
| S4 AutokeyRing | lag L; prefix key=0; ∓ | `TheoryHistChi2S4` |

Edge: `L ≥ T` → lag hist empty / all-prefix. Host:
`ctak_plain_hist_from_once`, `ring_plain_hist_from_once`.

**Complexity:** O(U·T) unique lags + O(Σ L_c) prefix (vs O(C·T) decode).

---

## 7. Bigram remap (Caesar LL)

**Once:** `BigramCountOnce` → `B[x0·29 + x1]` adjacent pair counts.  
**Score (canonical order):**

```text
score_s = − Σ_{x0=0..28} Σ_{x1=0..28} B[x0,x1] · ll[(x0−s) mod 29][(x1−s) mod 29]
```

Equivalent to rotating bigram bins then dotted with the LL table. Host:
`HistAlphabetMap::bigram_ll_dot_rotated` (production contract); stream
brute-force is golden only.

**Dict / n-gram dictionary matching** stays stream work — not remapped here.
`DeepScoreBatch` Caesar bigram-LL production uses Once+Dot; Dict kernel
untouched.

**Complexity:** O(T) bigrams + O(C·841) Dot.

---

## 8. What still uses 896B / decode-hist

| Path | Why |
|------|-----|
| Hard-S0 bytecode interpreter | Still streams ≈1 B/logical-rune per lane |
| Soft-fallback S0 | Same; PRIMARY = S0 diary roof until specialized |
| Totient / skip-interrupt shapes | Indexing breaks once-count contracts |
| Koan multi-stage compose | Stages may still materialize / decode |
| Legacy `launch_*_decode_hist_*` | Golden A/B only |

---

## 9. Related

- Ceilings / pass rule: [`cuda-throughput.md`](cuda-throughput.md)
- Theory S0–S5 contract: [`theory-hist-transpile.md`](theory-hist-transpile.md)
- Smart ShapeId: [`dsl-smart-hist.md`](dsl-smart-hist.md)
- ncu traffic evidence: [`profiles/cache_bound/SUMMARY.md`](profiles/cache_bound/SUMMARY.md)
- Historical fat-tile / 93.5% diary: [`profiles/roof_hist/SUMMARY.md`](profiles/roof_hist/SUMMARY.md)
- Profiling playbook: [`cuda-profile-theory.md`](cuda-profile-theory.md)
