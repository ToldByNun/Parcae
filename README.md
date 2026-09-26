# Parcae

**Turn Liber Primus runes into scored, reproducible cryptanalysis.**

Parcae is a C++20 toolkit for classical cipher search over Cicada 3301's
29-symbol alphabet. It freezes solved pages as regression oracles, ships
deterministic CLIs (optional CUDA twins), and compiles Python-authored theories
into verified artifacts -- so you (or an LLM agent) can search what remains
unsolved without inventing Z29 math.

**Who this is for:** researchers, engineers, and agent builders who want
fixture-grade evidence on Liber Primus (or any mod-29 / classical puzzle) --
not another cipher notebook full of vibes.

|                     |                                                                     |
| ------------------- | ------------------------------------------------------------------- |
| **License**         | [MIT](LICENSE)                                                      |
| **Toolkit version** | **0.9.0** (`v0.9.0-bench`)                                          |
| **Language (core)** | C++20 (header-first library + CLIs)                                 |
| **GPU**             | Optional CUDA twins (CI stays CPU-only)                             |
| **Python**          | IDE stubs + AST dump + optional CMD agent -- **not** the crypto core |

**Start here:** [5-minute tutorial](#5-minute-tutorial) / [Theory DSL examples](#4-write-a-theory-dsl) / [Docs map](#documentation-map)

---

## 5-minute tutorial

**Goal:** decrypt a known Liber Primus page, then compile a real `@Theory`.
If every command below exits `0`, you are done.

### 0. Get the tools

**Fastest (release installer):** download a CPU package from GitHub **Releases**
(see [`docs/architecture/release.md`](docs/architecture/release.md)). The
Windows `.exe` installs CLIs + `data/` and puts tools on `PATH`.

**From source** (C++20 compiler, CMake >= 3.25; first configure needs network for
FetchContent):

```bash
cmake -S . -B build -DPARCAE_BUILD_TOOLS=ON -DPARCAE_BUILD_TESTS=ON
cmake --build build --config Release
```

Binaries:

- Visual Studio / multi-config: `build/tools/Release/`
- Ninja / Make: `build/tools/`

```bash
# bash / Git Bash
export BIN=build/tools/Release   # or build/tools
export PARCAE_DATA_DIR="$PWD/data"
```

```powershell
# PowerShell
$BIN = "build/tools/Release"     # or build/tools
$env:PARCAE_DATA_DIR = "$PWD/data"
```

### 1. Prove the oracle locks (~10s)

```bash
$BIN/parcae-validate --data-dir data --id a-warning --require-locked
```

Expect `PASS a-warning` and exit `0`.

### 2. Decrypt "A Warning" (~10s)

```bash
$BIN/parcae-decode --data-dir data \
  --manifest data/fixtures/solved/a-warning --json
```

`result.latin` should start with `AWARNINGBELIEUE...` (Atbash over the frozen
Gematria Primus alphabet).

### 3. Generate -> rank one candidate (~15s)

```bash
$BIN/parcae-generate --data-dir data --generator-id gen_atbash \
  --input data/fixtures/solved/a-warning/ciphertext.txt --runes --json \
| $BIN/parcae-rank --data-dir data --candidates - \
  --score-id chi2_english_gp_v0 --k 1 --json
```

Top hit `candidate_id` is `atbash`; Latin again starts with `AWARNING...`.

### 4. Write a theory (DSL)

Theories are **not** ordinary Python programs. They are a restricted DSL that
looks like Python: only `from parcae.dsl.* import ...` is allowed (no `numpy`,
no stdlib, no bare `import`). Decorators (`@Theory`, `@ComposedTheory`,
`@define_primitive`) mark structure for `parcae-compile`. IDE stubs autocomplete
but **refuse to run** crypto -- only the C++ compiler verifies and emits.

Optional stubs for editor help:

```bash
pip install -e ./python
```

#### Elementwise `@Theory` (Caesar-style shift)

```python
from parcae.dsl.math import Z29Expr
from parcae.dsl.theory import Theory, Param

@Theory(
    name="hello_shift",
    family="elementwise",
    tier="A",
)
class HelloShift:
    shift: Param[int] = Param(min=0, max=28)

    def encrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x + self.shift

    def decrypt_step(self, x: Z29Expr) -> Z29Expr:
        return x - self.shift
```

#### Compose `@ComposedTheory` (catalog stages)

```python
from parcae.dsl.theory import ComposedTheory, Param

@ComposedTheory(
    name="hello_atbash_caesar",
    steps=["atbash", "caesar"],
    tier="A",
)
class HelloAtbashCaesar:
    caesar_shift: Param[int] = Param(min=0, max=28)

    def step_params(self) -> dict:
        return {"atbash": {}, "caesar": {"shift": self.caesar_shift}}
```

#### New math: `@define_primitive` + keyed `@Theory`

```python
from parcae.dsl.math import Z29Expr
from parcae.dsl.primitives import define_primitive
from parcae.dsl.theory import Theory, Param

@define_primitive(
    name="poly2_mod29",
    signature="(i: Z29, c2: Z29, c1: Z29, c0: Z29) -> Z29",
)
def poly2_mod29(i: Z29Expr, c2: Z29Expr, c1: Z29Expr, c0: Z29Expr) -> Z29Expr:
    return (c2 * i * i) + (c1 * i) + c0

@Theory(
    name="quadratic_polynomial_stream",
    family="keyed_stream",
    tier="B",
    interrupts="none_by_design",
)
class QuadraticPolynomialStream:
    c2: Param[int] = Param(min=0, max=28)
    c1: Param[int] = Param(min=0, max=28)
    c0: Param[int] = Param(min=0, max=28)

    def structural_claim(self) -> str:
        return (
            "Speculative. Quadratic keystream on Z29; exhaustive verify proves "
            "totality/determinism only -- not LP statistics (Tier B)."
        )

    def keystream_at(self, i: Z29Expr) -> Z29Expr:
        return poly2_mod29(i, self.c2, self.c1, self.c0)

    def encrypt_step(self, x: Z29Expr, i: Z29Expr) -> Z29Expr:
        return x + self.keystream_at(i)

    def decrypt_step(self, x: Z29Expr, i: Z29Expr) -> Z29Expr:
        return x - self.keystream_at(i)
```

Save a file under `theories/` (or use the repo copies) and compile:

```bash
# Compose (Atbash o Caesar) -- same idea as Liber Primus "Koan 1" style
$BIN/parcae-compile --data-dir data theories/examples/full_lifecycle_example.py
$BIN/parcae-validate --data-dir data --theory koan1_style@1

# New-math keystream
$BIN/parcae-compile --data-dir data theories/examples/new_math_example.py
$BIN/parcae-validate --data-dir data --theory quadratic_polynomial_stream@1
```

Full sources: [`theories/examples/`](theories/examples/). Language rules:
[`docs/spec/dsl.md`](docs/spec/dsl.md) (whitelist imports, tiers, HotLoop vs
OuterControl). Stubs vs compiler:
[`docs/architecture/dsl-stubs.md`](docs/architecture/dsl-stubs.md).

Stuck on CLIs? [`docs/spec/tools.md`](docs/spec/tools.md). Alphabet background
(optional): [`docs/research/gematria-primus.md`](docs/research/gematria-primus.md).

---

## Why this exists

Liber Primus is a rune ciphertext. Early pages are **solved** with known methods
(Atbash, Vigenère + skips, totient streams, ...). The rest is still open.

Parcae's job is not vibes. It is to:

1. Encode the frozen **Gematria Primus** alphabet Z29.
2. Reproduce **solved** pages as regression fixtures so the code cannot drift.
3. Give you fast, deterministic tools to **search, score, and record hypotheses**
   for the unsolved material -- including GPU batches and an optional agent.

| Number | Meaning |
| ------ | ------- |
| **29** | Alphabet size. Every rune -> index `0...28` (`Index29`). All crypto math is mod 29. |
| **LP2** `0`-`55` | Still-unsolved Liber Primus image range this toolkit aims to push forward. |
| **CMake ≥ 3.25** / **Python ≥ 3.11** | Build / stubs+agent requirements. |

Solved-page oracles: [`data/fixtures/solved/`](data/fixtures/solved/). Research:
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
      ├── generate -> rank -> hypothesis             (candidate search)
      ├── CUDA twins                               (optional, bit-parity with CPU)
      └── theory.py ──► parcae-compile ──► artifact (verified theory)
                              ▲
                     Python stubs = IDE only
```

**Rule of thumb:** if it changes ciphertext -> plaintext math, it lives in **C++**
(and optionally CUDA). Python is for authoring hints and the optional agent loop.

---

## What you can do with Parcae


| Goal                                  | Start here                                                                                                              |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Learn the 29-rune alphabet            | [`docs/research/gematria-primus.md`](docs/research/gematria-primus.md)                                                  |
| Reproduce a solved page               | `parcae-decode --manifest ...` + `parcae-validate`                                                                        |
| Score a candidate plaintext           | `parcae-score`                                                                                                          |
| Enumerate small transform grids       | `parcae-generate` -> `parcae-rank`                                                                                       |
| Record a research hypothesis          | `parcae-hypothesis` + [`docs/spec/hypothesis-workspace.md`](docs/spec/hypothesis-workspace.md)                          |
| Author a new theory (DSL)             | [Theory DSL examples](#4-write-a-theory-dsl); [`theories/examples/`](theories/examples/)  |
| Drive tools from an LLM               | [`agents/`](agents/README.md) (`parcae-agent`)                                                                          |
| Closed-loop search (workspace cycles) | [`docs/architecture/search-handbook.md`](docs/architecture/search-handbook.md) (`parcae-search-cycle` / `search_cycle`) |
| GPU fused search / throughput         | [`docs/architecture/cuda-build.md`](docs/architecture/cuda-build.md)                                                    |


---



## Python in this repo

There are **two** Python packages. Neither replaces the C++ crypto core.

### A) Theory DSL stubs -- `python/` (`parcae.dsl`)


|                      |                                                                               |
| -------------------- | ----------------------------------------------------------------------------- |
| **Purpose**          | Autocomplete / type stubs while writing `theory.py`                           |
| **Install**          | `cd python && pip install -e ".[dev]"`                                        |
| **Compiler?**        | **No.** Calling ops / `apply` / test runners raises `ParcaeDslStubError`      |
| **Real compile**     | `parcae-compile path/to/theory.py` (C++ pipeline)                             |
| **Syntax-only dump** | `python -m parcae.dsl.ast_dump theory.py` -> AST JSON (still not verification) |


```text
theory.py  --import-->  parcae.dsl stubs     (IDE only; calls raise)
    │
    ├── ast_dump ----->  dsl_ast_json.v0     (syntax wire format)
    │
    └── parcae-compile -> data/theories/...    (only verified path)
```


| Doc                                                                                | Role                           |
| ---------------------------------------------------------------------------------- | ------------------------------ |
| [`docs/architecture/dsl-stubs.md`](docs/architecture/dsl-stubs.md)                 | Stubs vs compiler (read first) |
| [`docs/architecture/python-transpiler.md`](docs/architecture/python-transpiler.md) | Full compiler architecture     |
| [`docs/spec/dsl.md`](docs/spec/dsl.md)                                             | Language rules                 |
| [`theories/examples/README.md`](theories/examples/README.md)                       | Example theories               |


**Compile an example** (after building tools):

```bash
$BIN/parcae-compile --data-dir data theories/examples/full_lifecycle_example.py
$BIN/parcae-validate --data-dir data --theory koan1_style@1
```

Artifacts land under `data/theories/<name>/<version>/` with URI form
`parcae://theories/<name>@<version>`.

### B) CMD agent -- `agents/` (`parcae-agent`)


|              |                                                                              |
| ------------ | ---------------------------------------------------------------------------- |
| **Purpose**  | LLM chooses **allow-listed** C++ tools; never reimplements Z29   |
| **Install**  | `cd agents && pip install -e ".[dev]"`                                       |
| **Crypto?**  | **No.** Subprocess -> `parcae-* --json` only                                  |
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


| CLI                                   | Role                                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `parcae-tokenize`                     | Ciphertext -> tokens / indices                                                                           |
| `parcae-decode`                       | Apply a catalog transform (or fixture manifest)                                                         |
| `parcae-score`                        | Deterministic scores (`ic_mod29`, `chi2_english_gp_v0`, ...)                                              |
| `parcae-validate`                     | Fixture oracles **or** theory artifacts                                                                 |
| `parcae-catalog`                      | List transforms, scores, generators, theories                                                           |
| `parcae-generate`                     | Expand bounded candidate grids                                                                          |
| `parcae-rank`                         | Top-k score candidates                                                                                  |
| `parcae-hypothesis`                   | Workspace hypothesis CRUD / score                                                                       |
| `parcae-compile`                      | Theory DSL -> verified artifact                                                                          |
| `parcae-sweep`                        | Expand theory `param_grid` plans                                                                        |
| `parcae-search-run`                   | Fused search / throughput-style sweeps                                                                  |
| `parcae-parity` / `parcae-parity-gen` | CPU↔CUDA parity records                                                                                 |
| `parcae-blind-crack`                  | Research battery on locked fixtures                                                                     |
| `parcae-bench`                        | Benchmark & diagnostics (`slo` / `accuracy` / `hardware` / `probe` / `all`; **deny-listed** for agents) |
| `parcae-throughput-tiers`             | CUDA SLO compat ≡ `parcae-bench --suite slo --extended --allow-cuda` (**deny-listed**)                  |




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


| If you want...                         | Read                                                                               |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| Research / alphabet / solved methods | [`docs/research/`](docs/research/README.md)                                        |
| Binding specs (MUST/SHOULD)          | [`docs/spec/`](docs/spec/README.md)                                                |
| How the C++ / CUDA / DSL stack fits  | [`docs/architecture/`](docs/architecture/README.md)                                |
| Theory DSL compiler                  | [`docs/architecture/python-transpiler.md`](docs/architecture/python-transpiler.md) |
| DSL stubs vs compile                 | [`docs/architecture/dsl-stubs.md`](docs/architecture/dsl-stubs.md)                 |
| CMD agent operator guide             | [`docs/architecture/agent-handbook.md`](docs/architecture/agent-handbook.md)       |
| Local CUDA build notes               | [`docs/architecture/cuda-build.md`](docs/architecture/cuda-build.md)               |
| Example theories                     | [`theories/examples/`](theories/examples/)                                         |
| CUDA sources (VS)                    | [`Parcae/Parcae/cuda/`](Parcae/Parcae/cuda/)                                       |


Architecture hub: [`docs/architecture/README.md`](docs/architecture/README.md).

---



## Status & roadmap

```text
research + specs      ->  docs/research, docs/spec           done
CPU reference         ->  include/, fixtures, scores, CLIs   done
CUDA parity           ->  Parcae/Parcae/cuda/ twins          done -- v0.3.0-cuda-parity
Theory DSL compiler   ->  theories/ + include/parcae/dsl/    done -- v0.5.0-theory-dsl
CMD agent tooling     ->  agents/ + agent-facing CLIs        in progress -- v0.6.0-agent-tools
Search engine loop    ->  GPU ↔ candidates ↔ hypotheses      done -- v0.7.0-search-engine
Smart DSL + console   ->  scopes/Select/#ignore + dashboard  done -- v0.8.0-dsl-console
Bench & diagnostics   ->  parcae-bench SLO/accuracy/hw/probe done -- v0.9.0-bench
Open-source polish    ->  packaging, contribution docs       later
```

Frozen CUDA commit list: [`docs/architecture/cuda-roadmap.md`](docs/architecture/cuda-roadmap.md).  
CMD-agent plan: [`docs/architecture/agent-tooling.md`](docs/architecture/agent-tooling.md).  
Search engine: [`docs/architecture/search-engine.md`](docs/architecture/search-engine.md)
(exit checklist engineering-green) |
commit list [`docs/architecture/search-roadmap.md`](docs/architecture/search-roadmap.md) |
operator guide [`docs/architecture/search-handbook.md`](docs/architecture/search-handbook.md) |
spec [`docs/spec/search-loop.md`](docs/spec/search-loop.md).  
Smart DSL + console exit: [`docs/architecture/dsl-console-exit.md`](docs/architecture/dsl-console-exit.md)
(`v0.8.0-dsl-console`) | compiler guide
[`docs/architecture/python-transpiler.md`](docs/architecture/python-transpiler.md) |
console progress in [`docs/architecture/search-handbook.md`](docs/architecture/search-handbook.md).  
Bench exit: [`docs/architecture/bench-exit.md`](docs/architecture/bench-exit.md)
(`v0.9.0-bench`) | operator guide
[`docs/architecture/bench-diagnostics.md`](docs/architecture/bench-diagnostics.md).

### Goals

1. **Advance unsolved LP2** `0`**-**`55` with systematic search and scored hypotheses.
2. **Ground truth from solved pages** -- fixtures must keep reproducing known decrypts.
3. **CPU reference ↔ CUDA parity** -- same `Index29` math, comparable scores under documented FP rules.
4. **Deterministic agent tools** -- LLMs call CLIs; they do not own the crypto.
5. **C++-first core** -- Python is authoring DX + optional agent, not a second math engine.



### Non-goals

- Embedding original puzzle JPGs in the repo
- Publishing "solutions" without reproducible fixture-grade evidence
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
It also gates smart DSL (`[dsl-smart]`), console progress (`[cli-progress]`),
deterministic bench (`[bench]` -- spec/probe/accuracy/report; no absolute runes/s),
theory examples (`[dsl][examples]`), stale-spec rejection (`[dsl][registry][stale]`),
stub pytest, and `scripts/check-dsl-examples.sh`.

---



## Style & contributing

- C++: [`.clang-format`](.clang-format) (LLVM-ish, 4-space), light [`.clang-tidy`](.clang-tidy).
- Prefer top-level **classes** and `#ifndef` headers; the project avoids C++
namespaces in library code.
- Specs use RFC-style **MUST / SHOULD / MAY** ([`docs/spec/README.md`](docs/spec/README.md)).

Issues and PRs welcome once you can show a green `ctest` (and, for DSL changes,
`[dsl][examples]` / stub tests as relevant).