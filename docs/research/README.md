# Phase 0 — Research

Parcae is a Liber Primus / Cicada 3301 cryptanalysis toolkit. Phase 0 freezes the
facts we will build against so later phases do not re-litigate alphabet order,
separator grammar, or solved-page methods.

## Confidence labels used in this folder

| Label | Meaning |
|-------|---------|
| **Frozen fact** | Independently attested across primary community sources (wiki methods, Boxentriq tools, multiple transcriptions) and required for Phase 2 oracles |
| **Working assumption** | Widely used convention; must be locked for tooling but can be revisited if fixtures fail |
| **Unverified single-source** | Live page exists and was read; claims are **not** independently reproduced by Parcae yet — must not gate architecture as hard fact |

## Why these 29 runes (not Elder Futhark / another set)

Liber Primus does **not** use historical Elder Futhark as the cipher alphabet.
Cicada published **Gematria Primus**: a fixed 29-symbol set with attached primes
(first 29 primes) and Latin values. Solved pages only decrypt correctly when
indices follow that published order (`ᚠ`=0 … `ᛠ`=28). Parcae freezes exactly that
table (see [gematria-primus.md](gematria-primus.md)) and ignores alternate
historical rune inventories on purpose — they are the wrong modulus and the wrong
glyph→value map for this puzzle.

## Research questions

1. What known properties does Liber Primus / Gematria Primus have?
2. Which transforms over \(\mathbb{Z}_{29}\) are actually attested or plausible?
3. Which modulo-29 hypotheses are confirmed, open, or statistically constrained?
4. Which sections make good Phase 2 test material (solved oracles)?

## Documents in this folder

| File | Purpose |
|------|---------|
| [gematria-primus.md](gematria-primus.md) | Frozen 29-rune alphabet, indices vs primes, Latin aliases |
| [separators.md](separators.md) | Transcript separator grammar (ASCII + Unicode maps) |
| [solved-methods.md](solved-methods.md) | Catalog of solved pages and exact decrypt methods |
| [hypotheses.md](hypotheses.md) | Plausible vs implausible mod-29 transform families |
| [test-material.md](test-material.md) | Fixture IDs selected for Phase 2 |
| [phase0-exit.md](phase0-exit.md) | Exit criteria for Phase 0 |

## Non-goals (Phase 0)

- No implementation code
- No CUDA, agents, or search engine design beyond noting constraints
- No solve attempts on unsolved LP2 pages `0.jpg`–`55.jpg`
- No claim of new plaintext

## Primary sources

| Source | Use |
|--------|-----|
| [Uncovering Cicada — How the solved pages were solved](https://uncovering-cicada.fandom.com/wiki/How_the_solved_pages_of_the_Liber_Primus_were_solved) | Method formulas, skip indices, ciphertext excerpts |
| [Uncovering Cicada — Liber Primus post-2014](https://uncovering-cicada.fandom.com/wiki/What_Happened_Liber_Primus_(Post_2014)) | Totient / page 56 notes |
| [CicadaSolvers quickstart](https://www.cicadasolvers.com/quickstart/) | LP1/LP2 page counts; 56.jpg / 57.jpg status |
| [Boxentriq — Gematria Primus translator](https://www.boxentriq.com/encodings/gematria-primus-translator) | Alphabet table and aliases |
| [Boxentriq — Totient cipher](https://www.boxentriq.com/ciphers/cicada-3301-totient-cipher) | Prime−1 stream + F-skip behavior |
| [Boxentriq — Liber Primus guide](https://www.boxentriq.com/guides/cicada-3301-liber-primus) | Solved-page walkthrough summary |
| [scream314/cicada3301 `liber_primus.md`](https://github.com/scream314/cicada3301/blob/master/liber_primus.md) | Long-form page archive |
| [rtkd Liber Primus transcription](https://github.com/rtkd/iddqd/blob/master/liber-primus__transcription--master/liber-primus__transcription--master.txt) | ASCII separator convention (`-` `.` `/` `%` `&`) |
| [relikd/LiberPrayground](https://github.com/relikd/liberprayground) | Unicode punctuation map (`•` `⁘` `⁚` `⁖` `⁜`) |
| [NoxxGames/LiberPrimus-GPU](https://github.com/NoxxGames/LiberPrimus-GPU) | Prior-art profile/separator docs (independent; Parcae is C++-first) |

### Secondary / unverified single-source

| Source | Verification (2026-03-15 session) | Use |
|--------|-----------------------------------|-----|
| [artwaste — One Bit of Structure](https://artwaste.land/strata/liber-primus-measured/) | **Page exists** (~39 KB essay; claims ~0.664% self-follow, OTP-class battery, plaintext-F interrupt table). **Not** independently recomputed or peer-reviewed in Parcae. | Optional background for unsolved-page search heuristics only. **Not** a Phase 0/2 freeze fact. See [hypotheses.md](hypotheses.md) Tier C. |

## Corpus split (community convention)

| Set | Pages | Status (community) |
|-----|-------|--------------------|
| LP1 | 17 pages (named sections) | All solved |
| LP2 | `0.jpg` … `57.jpg` (58 pages) | `56.jpg` and `57.jpg` solved; `0`–`55` unsolved |

Page numbering differs across archives (e.g. scream314 complete-set ids vs LP2 numeric names). Fixture IDs in Parcae are stable names, not raw JPG filenames.

## Phase boundaries

```text
Phase 0  research facts (this folder)
    ↓
Phase 1  mathematical / tool specifications
    ↓
Phase 2  CPU reference implementation + solved fixtures
    ↓
Phase 3+ CUDA parity, agents, search (out of scope here)
```
