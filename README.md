# Parcae

C++20 toolkit for Liber Primus / Cicada 3301 cryptanalysis. The **main goal** is
to push the still-unsolved LP2 pages (`0.jpg`–`55.jpg`) forward — new structure,
stronger hypotheses, and eventually real decrypts — with a deterministic,
GPU-ready core and agent-callable tools.

License: [MIT](LICENSE).

## Goals

1. **Advance unsolved Liber Primus (LP2 `0`–`55`)** — systematic search, scoring,
   and hypothesis tooling aimed at new discoveries, not museum reproduction.
2. **Ground truth from solved pages** — reproduce known methods (Atbash, Vigenère
   + cleartext-F skips, totient / prime−1 stream, identity) as regression oracles
   so the pipeline cannot silently drift.
3. **CPU reference → CUDA parity** — every hot transform is a pure function on
   `Index29` streams so GPU batches can match CPU bit-for-bit.
4. **Deterministic tools for agents** — tokenize / decode / score / validate only;
   no LLM inside the crypto core.
5. **C++-first** — no Python orchestration plane.

## Non-goals (for now)

- Shipping CUDA kernels or agent runtimes before the CPU reference is solid
- Embedding original puzzle JPGs
- Publishing unverified “solutions” without reproducible fixture-grade evidence

## Documentation map

| Area | Location |
|------|----------|
| Research (alphabet, solved methods, hypotheses) | [docs/research/](docs/research/README.md) |
| Normative specs (Z29, tokens, transforms, fixtures, tools, parity) | [docs/spec/](docs/spec/README.md) |
| Architecture (CPU map, CUDA ABI / roadmap / throughput / Phase 4) | [docs/architecture/](docs/architecture/README.md) |
| CUDA sources (Visual Studio) | [Parcae/Parcae/cuda/](Parcae/Parcae/cuda/) |

## Roadmap (high level)

```text
research + specs     →  docs/research, docs/spec          (done)
CPU reference        →  include/, fixtures, scores, CLIs  (done)
CUDA parity          →  Parcae/Parcae/cuda/ twins         (done — v0.3.0-cuda-parity)
AI tooling           →  CMD agent + deterministic tools   (Phase 4 — planned)
search on LP2 0–55   →  candidates ↔ hypotheses → new discoveries
open source polish   →  packaging, contribution docs
```

Frozen CUDA commit list: [docs/architecture/cuda-roadmap.md](docs/architecture/cuda-roadmap.md).  
Frozen Phase 4 commit list: [docs/architecture/phase4-agent-tooling.md](docs/architecture/phase4-agent-tooling.md).

## Build

Requires CMake ≥ 3.25, a C++20 compiler, and network on first configure
(FetchContent pulls Catch2 `v3.7.1` and nlohmann/json `v3.11.3`).

```bash
cmake -S . -B build -DPARCAE_BUILD_TESTS=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Options:

| Option | Default | Meaning |
|--------|---------|---------|
| `PARCAE_BUILD_TESTS` | `ON` | Fetch Catch2 and build `parcae_tests` |
| `PARCAE_BUILD_TOOLS` | `ON` | Build CLIs: `parcae-tokenize`, `parcae-decode`, `parcae-score`, `parcae-validate` |
| `PARCAE_BUILD_CUDA` | `OFF` | Build `parcae_cuda` from [`Parcae/Parcae/cuda/`](Parcae/Parcae/cuda/) (needs nvcc) |

**Visual Studio (CUDA day-to-day):** open [`Parcae/Parcae.slnx`](Parcae/Parcae.slnx), build **x64** with the CUDA Toolkit VS integration installed. Sources live under `Parcae/Parcae/cuda/`. Full local notes (CMake flags, Catch2 tags, skip behavior): [`docs/architecture/cuda-build.md`](docs/architecture/cuda-build.md).

CI (`.github/workflows/ci.yml`) stays **CPU-default** — it does not enable `PARCAE_BUILD_CUDA`.

On multi-config generators (Visual Studio CMake), binaries land in
`build/tools/Release/`. On single-config (Ninja/Make), they are in
`build/tools/`. Examples below use `BIN=build/tools/Release` — adjust if needed.

Style: [`.clang-format`](.clang-format) (LLVM-ish, 4-space) and a light
[`.clang-tidy`](.clang-tidy) baseline. CI runs on Ubuntu and Windows
(`.github/workflows/ci.yml`).

## CLI tools

All tools accept `--data-dir <path>` (or `PARCAE_DATA_DIR`) pointing at the repo
`data/` root. Normative contracts: [docs/spec/tools.md](docs/spec/tools.md).

### Tokenize

```bash
$BIN/parcae-tokenize --data-dir data --json \
  data/fixtures/solved/a-warning/ciphertext.txt
```

### Decode

Manifest mode (method + skips from the fixture):

```bash
$BIN/parcae-decode --data-dir data \
  --manifest data/fixtures/solved/a-warning
```

Flag mode (Atbash decrypt of the same page):

```bash
$BIN/parcae-decode --data-dir data \
  --input data/fixtures/solved/a-warning/ciphertext.txt \
  --transform-id atbash --direction decrypt
```

Keyed example (Welcome / DIVINITY + skip indices):

```bash
$BIN/parcae-decode --data-dir data \
  --manifest data/fixtures/solved/welcome
```

### Score

```bash
$BIN/parcae-score --data-dir data --list
$BIN/parcae-score --data-dir data --score-id ic_mod29 --indices \
  --input data/fixtures/cli/score-indices.txt
$BIN/parcae-score --data-dir data --score-id chi2_english_gp_v0 --latin \
  --input data/fixtures/solved/a-warning/plaintext.txt --json
```

### Validate

```bash
$BIN/parcae-validate --data-dir data --id a-warning --require-locked
$BIN/parcae-validate --data-dir data --all --require-locked --json
```

Exit codes for validate: `0` pass, `1` fixture failure, `2` usage/I/O error.
