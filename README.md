# Parcae

**C++20 cryptanalysis toolkit for Cicada 3301’s Liber Primus** — deterministic
transforms, scores, and fixtures on a 29-symbol alphabet, with optional CUDA
twins, a theory DSL, and an LLM agent that only calls allow-listed CLIs.

| | |
|---|---|
| **License** | [MIT](LICENSE) |
| **Toolkit version** | **0.5.0** (`v0.5.0-theory-dsl`) |
| **Language (core)** | C++20 (header-first library + CLIs) |
| **GPU** | Optional CUDA twins (CI stays CPU-only) |
| **Python** | IDE stubs + AST dump + optional CMD agent — **not** the crypto core |

---

## Why this exists

Cicada’s Liber Primus (LP) is a rune ciphertext. Community work has **solved**
the early pages with known methods (Atbash, Vigenère + skips, totient streams,
…). Pages that remain unsolved are usually labeled **LP2 `0`–`55`** (image
indices, not “page numbers in a book”).

Parcae’s job is not to ship vibes. It is to:

1. Encode the frozen **Gematria Primus** alphabet (\(\mathbb{Z}_{29}\)).
2. Reproduce **solved** pages as regression fixtures so the code cannot drift.
3. Give you fast, deterministic tools to **search, score, and record hypotheses**
   for the unsolved material — including GPU batches and an optional agent.

---

## Numbers worth memorizing

| Number | Meaning |
|--------|---------|
| **29** | Alphabet size. Every rune maps to an index `0…28` (`Index29`). All crypto math is mod 29. |
| **0…28** | Valid `Index29` values. Latin “letters” are a *view* of those indices, not a second alphabet. |
| **LP2 `0`–`55`** | Still-unsolved Liber Primus image range the toolkit aims to push forward. |
| **0.5.0** | Current toolkit version (theory DSL compiler complete). |
| **CMake ≥ 3.25** | Build requirement. |
| **Python ≥ 3.11** | Only for DSL stubs / `ast_dump` / optional `parcae-agent`. |

Solved-page oracles live under [`data/fixtures/solved/`](data/fixtures/solved/)
(e.g. `a-warning`, `welcome`, `koan-1`). Research background:
[`docs/research/`](docs/research/README.md).

---

## 60-second mental model

```text
ciphertext / runes
      │
      ▼
 Index29 stream          ← 29-symbol Gematria Primus
      │
      ├── tokenize / decode / score / validate     (C++ CLIs)
      ├── generate → rank → hypothesis             (candidate search)
      ├── CUDA twins                               (optional, bit-parity with CPU)
      └── theory.py ──► parcae-compile ──► artifact (verified theory)
                              ▲
                     Python stubs = IDE only
```

**Rule of thumb:** if it changes ciphertext → plaintext math, it lives in **C++**
(and optionally CUDA). Python is for authoring hints and the optional agent loop.

---

## Quick start (beginner)

### 1. Build the C++ toolkit

Needs a C++20 compiler, CMake ≥ 3.25, and network on the **first** configure
(FetchContent pulls Catch2 `v3.7.1` and nlohmann/json `v3.11.3`).

```bash
cmake -S . -B build -DPARCAE_BUILD_TESTS=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

| CMake option | Default | What it does |
|--------------|---------|--------------|
| `PARCAE_BUILD_TESTS` | `ON` | Build `parcae_tests` (Catch2) |
| `PARCAE_BUILD_TOOLS` | `ON` | Build all `parcae-*` CLIs |
| `PARCAE_BUILD_CUDA` | `OFF` | Build CUDA twins (needs nvcc / Toolkit) |

Binaries:

- Multi-config (Visual Studio): `build/tools/Release/`
- Single-config (Ninja/Make): `build/tools/`

Examples below use `BIN=build/tools/Release` — change if your generator differs.

### 2. Run your first commands

All tools take `--data-dir` (or `PARCAE_DATA_DIR`) pointing at the repo `data/`
root. Full CLI contracts: [`docs/spec/tools.md`](docs/spec/tools.md).

**Validate a locked solved fixture** (sanity check that your build matches ground truth):

```bash
$BIN/parcae-validate --data-dir data --id a-warning --require-locked
```

**Tokenize ciphertext → JSON:**

```bash
$BIN/parcae-tokenize --data-dir data --json \
  data/fixtures/solved/a-warning/ciphertext.txt
```

**Decode with the fixture’s known method:**

```bash
$BIN/parcae-decode --data-dir data \
  --manifest data/fixtures/solved/a-warning
```

**Score Latin plaintext with χ² against Gematria Primus English expectations:**

```bash
$BIN/parcae-score --data-dir data --score-id chi2_english_gp_v0 --latin \
  --input data/fixtures/solved/a-warning/plaintext.txt --json
```

Exit codes for `parcae-validate`: `0` pass, `1` fixture failure, `2` usage/I/O.

### 3. (Optional) Install Python DSL stubs for editor autocomplete

```bash
cd python
pip install -e ".[dev]"
pytest -m ci -q    # must raise ParcaeDslStubError on fake “execution”
```

Stubs **import** for IDE help. They **do not** compile or verify theories.
See [Python in this repo](#python-in-this-repo) below.

---

## What you can do with Parcae

| Goal | Start here |
|------|------------|
| Learn the 29-rune alphabet | [`docs/research/gematria-primus.md`](docs/research/gematria-primus.md) |
| Reproduce a solved page | `parcae-decode --manifest …` + `parcae-validate` |
| Score a candidate plaintext | `parcae-score` |
| Enumerate small transform grids | `parcae-generate` → `parcae-rank` |
| Record a research hypothesis | `parcae-hypothesis` + [`docs/spec/hypothesis-workspace.md`](docs/spec/hypothesis-workspace.md) |
| Author a new theory in Python syntax | [`theories/examples/`](theories/examples/) → `parcae-compile` |
| Drive tools from an LLM | [`agents/`](agents/README.md) (`parcae-agent`) |
| GPU fused search / throughput | [`docs/architecture/cuda-build.md`](docs/architecture/cuda-build.md) |

---

## Python in this repo

There are **two** Python packages. Neither replaces the C++ crypto core.

### A) Theory DSL stubs — `python/` (`parcae.dsl`)

| | |
|---|---|
| **Purpose** | Autocomplete / type stubs while writing `theory.py` |
| **Install** | `cd python && pip install -e ".[dev]"` |
| **Compiler?** | **No.** Calling ops / `apply` / test runners raises `ParcaeDslStubError` |
| **Real compile** | `parcae-compile path/to/theory.py` (C++ pipeline) |
| **Syntax-only dump** | `python -m parcae.dsl.ast_dump theory.py` → AST JSON (still not verification) |

```text
theory.py  --import-->  parcae.dsl stubs     (IDE only; calls raise)
    │
    ├── ast_dump ----->  dsl_ast_json.v0     (syntax wire format)
    │
    └── parcae-compile -> data/theories/…    (only verified path)
```

| Doc | Role |
|-----|------|
| [`docs/architecture/dsl-stubs.md`](docs/architecture/dsl-stubs.md) | Stubs vs compiler (read first) |
| [`docs/architecture/python-transpiler.md`](docs/architecture/python-transpiler.md) | Full compiler architecture |
| [`docs/spec/dsl.md`](docs/spec/dsl.md) | Language rules |
| [`theories/examples/README.md`](theories/examples/README.md) | Example theories |

**Compile an example** (after building tools):

```bash
$BIN/parcae-compile --data-dir data theories/examples/new_math_example.py
$BIN/parcae-validate --data-dir data --theory quadratic_polynomial_stream@1
```

Artifacts land under `data/theories/<name>/<version>/` with URI form
`parcae://theories/<name>@<version>`.

### B) CMD agent — `agents/` (`parcae-agent`)

| | |
|---|---|
| **Purpose** | LLM chooses **allow-listed** C++ tools; never reimplements \(\mathbb{Z}_{29}\) |
| **Install** | `cd agents && pip install -e ".[dev]"` |
| **Crypto?** | **No.** Subprocess → `parcae-* --json` only |
| **Handbook** | [`docs/architecture/agent-handbook.md`](docs/architecture/agent-handbook.md) |

```bash
cd agents
python -m parcae_agent doctor --config configs/ollama.example.yaml
python -m parcae_agent run --config configs/ollama.example.yaml \
  --prompt "List catalog transforms for a-warning"
```

---

## CLI toolkit (overview)

Normative contracts: [`docs/spec/tools.md`](docs/spec/tools.md). Agent allow/deny:
[`docs/spec/agent-tools.md`](docs/spec/agent-tools.md).

| CLI | Role |
|-----|------|
| `parcae-tokenize` | Ciphertext → tokens / indices |
| `parcae-decode` | Apply a catalog transform (or fixture manifest) |
| `parcae-score` | Deterministic scores (`ic_mod29`, `chi2_english_gp_v0`, …) |
| `parcae-validate` | Fixture oracles **or** theory artifacts |
| `parcae-catalog` | List transforms, scores, generators, theories |
| `parcae-generate` | Expand bounded candidate grids |
| `parcae-rank` | Top-k score candidates |
| `parcae-hypothesis` | Workspace hypothesis CRUD / score |
| `parcae-compile` | Theory DSL → verified artifact |
| `parcae-sweep` | Expand theory `param_grid` plans |
| `parcae-search-run` | Fused search / throughput-style sweeps |
| `parcae-parity` / `parcae-parity-gen` | CPU↔CUDA parity records |
| `parcae-blind-crack` | Research battery on locked fixtures |
| `parcae-throughput-tiers` | CUDA SLO tiers (GPU build) |

### More decode / score examples

Manifest mode:

```bash
$BIN/parcae-decode --data-dir data \
  --manifest data/fixtures/solved/a-warning
```

Flag mode (Atbash):

```bash
$BIN/parcae-decode --data-dir data \
  --input data/fixtures/solved/a-warning/ciphertext.txt \
  --transform-id atbash --direction decrypt
```

Keyed solved page (Welcome / DIVINITY + skips):

```bash
$BIN/parcae-decode --data-dir data \
  --manifest data/fixtures/solved/welcome
```

Score helpers:

```bash
$BIN/parcae-score --data-dir data --list
$BIN/parcae-score --data-dir data --score-id ic_mod29 --indices \
  --input data/fixtures/cli/score-indices.txt
```

---

## Documentation map

Start at the top of each column; go deeper as needed.

| If you want… | Read |
|--------------|------|
| Research / alphabet / solved methods | [`docs/research/`](docs/research/README.md) |
| Binding specs (MUST/SHOULD) | [`docs/spec/`](docs/spec/README.md) |
| How the C++ / CUDA / DSL stack fits | [`docs/architecture/`](docs/architecture/README.md) |
| Theory DSL compiler | [`docs/architecture/python-transpiler.md`](docs/architecture/python-transpiler.md) |
| DSL stubs vs compile | [`docs/architecture/dsl-stubs.md`](docs/architecture/dsl-stubs.md) |
| CMD agent operator guide | [`docs/architecture/agent-handbook.md`](docs/architecture/agent-handbook.md) |
| Local CUDA build notes | [`docs/architecture/cuda-build.md`](docs/architecture/cuda-build.md) |
| Example theories | [`theories/examples/`](theories/examples/) |
| CUDA sources (VS) | [`Parcae/Parcae/cuda/`](Parcae/Parcae/cuda/) |

Architecture hub: [`docs/architecture/README.md`](docs/architecture/README.md).

---

## Status & roadmap

```text
research + specs      →  docs/research, docs/spec           done
CPU reference         →  include/, fixtures, scores, CLIs   done
CUDA parity           →  Parcae/Parcae/cuda/ twins          done — v0.3.0-cuda-parity
Theory DSL compiler   →  theories/ + include/parcae/dsl/    done — v0.5.0-theory-dsl
CMD agent tooling     →  agents/ + agent-facing CLIs        in progress — v0.6.0-agent-tools
Search engine loop    →  GPU ↔ candidates ↔ hypotheses      planned — v0.7.0-search-engine
Open-source polish    →  packaging, contribution docs       later
```

Frozen CUDA commit list: [`docs/architecture/cuda-roadmap.md`](docs/architecture/cuda-roadmap.md).  
CMD-agent plan: [`docs/architecture/agent-tooling.md`](docs/architecture/agent-tooling.md).

### Goals

1. **Advance unsolved LP2 `0`–`55`** with systematic search and scored hypotheses.
2. **Ground truth from solved pages** — fixtures must keep reproducing known decrypts.
3. **CPU reference ↔ CUDA parity** — same `Index29` math, comparable scores under documented FP rules.
4. **Deterministic agent tools** — LLMs call CLIs; they do not own the crypto.
5. **C++-first core** — Python is authoring DX + optional agent, not a second math engine.

### Non-goals

- Embedding original puzzle JPGs in the repo
- Publishing “solutions” without reproducible fixture-grade evidence
- Putting an LLM inside transform / score kernels
- Requiring a GPU for everyday CPU development or hosted CI

---

## CUDA (optional)

Day-to-day GPU work: open [`Parcae/Parcae.slnx`](Parcae/Parcae.slnx), build **x64**
with the CUDA Toolkit VS integration. Sources:
[`Parcae/Parcae/cuda/`](Parcae/Parcae/cuda/).

```bash
cmake -S . -B build-cuda -DPARCAE_BUILD_CUDA=ON -DPARCAE_BUILD_TESTS=ON -DPARCAE_BUILD_TOOLS=ON
```

Full notes (flags, Catch2 tags, skip behavior when Toolkit is absent):
[`docs/architecture/cuda-build.md`](docs/architecture/cuda-build.md).

Hosted CI (`.github/workflows/ci.yml`) stays **CPU-default** (`PARCAE_BUILD_CUDA=OFF`).
It also gates theory examples (`[dsl][examples]`), stale-spec rejection
(`[dsl][registry][stale]`), stub pytest, and `scripts/check-dsl-examples.sh`.

---

## Style & contributing

- C++: [`.clang-format`](.clang-format) (LLVM-ish, 4-space), light [`.clang-tidy`](.clang-tidy).
- Prefer top-level **classes** and `#ifndef` headers; the project avoids C++
  namespaces in library code.
- Specs use RFC-style **MUST / SHOULD / MAY** ([`docs/spec/README.md`](docs/spec/README.md)).

Issues and PRs welcome once you can show a green `ctest` (and, for DSL changes,
`[dsl][examples]` / stub tests as relevant).
