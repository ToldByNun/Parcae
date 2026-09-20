# CPU reference architecture

How the C++20 Liber Primus toolkit is laid out today: modules, ownership,
and the data flow from UTF-8 transcript to scored / validated Index29 streams.

Normative contracts live in [`docs/spec/`](../spec/README.md). Research background
is in [`docs/research/`](../research/README.md). CUDA twin expectations are in
[`cuda-handoff.md`](cuda-handoff.md); device ABI in [`cuda-abi.md`](cuda-abi.md);
roadmap in [`cuda-roadmap.md`](cuda-roadmap.md).

CPU reference is the source of truth. CUDA twin architecture:
[`cuda-reference.md`](cuda-reference.md). Sources land under
[`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/) (Visual Studio project), not
under a parallel `include/parcae/cuda/` tree.

## Design rules

1. **Hot path = pure Index29** — transforms take `span<const Index29>` and write
   `span<Index29>` (`apply_into` / static `kernel`). No I/O, no globals.
2. **Corpus owns structure** — separators, numbers, and literals stay in
   `TokenStream`; kernels never see them.
3. **Interrupts are explicit** — `InterruptPolicy` skip indices on the consumable
   rune stream; never inferred inside a kernel.
4. **Tools are thin** — CLIs and `parcae::tool` wrap the library; no search / LLM
   logic in the crypto core.
5. **Determinism** — same inputs ⇒ same Index29 bits and score IEEE bits
   (see [`parity.md`](../spec/parity.md)).

## Module map

```text
include/parcae/
  core/          Index29, Z29, Status, SHA-256, Version
  dsl/           Theory DSL compiler (ingest + semantic gate; IR later)
  gematria/      Profile load, RuneCodec, LatinCodec / labels
  corpus/        Tokenizer, TokenStream, Fixture + loader, masks
  interrupt/     InterruptPolicy (explicit skip indices)
  transform/     Families, ApplyTransform, span kernels
  math/          Primes sieve, TotientKeystream
  score/         Tier-A scores + ScoreRegistry
  generate/      Bounded candidate generators
  batch/         Top-k BatchRunner (serial / ordered parallel)
  validate/      FixtureValidator, PlaintextNormalizer, reports
  parity/        ParityRecord (params/input/output/interrupt digests)
  tool/          Context, TransformEnvelope, agent-facing API

tools/
  parcae-tokenize | parcae-decode | parcae-score | parcae-validate
  parcae-catalog | parcae-compile | parcae-parity | parcae-parity-gen | parcae-search-run

Parcae/Parcae/                 # Visual Studio app + CUDA twins
  main.cpp
  cuda/                        # device twins (see cuda-roadmap.md)
    emitted/                   # generated DSL twins (gitignored except README)

data/
  profiles/      gematria, separators, score tables
  fixtures/      solved oracles (+ synth drafts, CLI smoke inputs)
  theories/      parcae-compile artifacts (gitignored except README)
  parity/        (planned) ParityRecord goldens for CPU↔CUDA
```

| Module | Responsibility | CUDA twin? |
|--------|----------------|------------|
| `core` / `gematria` / `corpus` | Host setup, UTF-8, fixtures | No (CPU setup) |
| `dsl` | Theory DSL compile (ingest + semantic gate; IR later) | Emitted twins under `cuda/emitted/` |
| `interrupt` | Skip-set policy | Policy bytes mirrored on device |
| `transform` + `math` | Index29 kernels / keystream | **Yes — primary** |
| `score` | Pure scores on Index29 | **Yes — reductions** |
| `generate` / `batch` | Candidate expand + top-k | **Yes — batch ABI** |
| `validate` / `parity` / `tool` / CLIs | Oracles, digests, UX | No (call CPU or CUDA backends) |

## Data flow

### Solved-page validation (oracle path)

```text
fixture dir (manifest + ciphertext + plaintext)
        │
        ▼
 FixtureLoader ──► Fixture (method, skips, hashes)
        │
        ▼
 Tokenizer (+ gematria + separator grammar)
        │
        ▼
 TokenStream.consumable_indices()  ── span<Index29>
        │
        ▼
 ApplyTransform / family::apply_into  (+ InterruptPolicy)
        │
        ▼
 PlaintextNormalizer.from_indices  ↔  normalize(expected plaintext)
        │
        ▼
 compare Latin + optional SHA-256 locks  ──► ValidationReport
```

CLI: `parcae-validate --id <fixture> --require-locked`.

### Decode / score (interactive path)

```text
UTF-8 source
   │
   ├─► tokenize ──► TokenStream ──► consumable Index29
   │                                      │
   │                         TransformEnvelope / flags
   │                                      │
   │                                      ▼
   │                              apply_into / kernel
   │                                      │
   │                      ┌───────────────┼───────────────┐
   │                      ▼               ▼               ▼
   │                 to_latin          score          ParityRecord
   │                                                      │
   └──────────────────────────────────────────────────────┘
```

CLIs: `parcae-tokenize`, `parcae-decode`, `parcae-score`.

### Batch search sketch (CPU today)

```text
generators ──► N TransformCandidate
                    │
                    ▼
            BatchRunner (serial or ordered-parallel)
                    │
                    ▼
            top-k BatchHit by ScoreRegistry id
```

Ordering for parallel runs is documented in `batch/` (`BatchOrdering`);
single-threaded CPU remains the source of truth for parity.

## Transform hot path

```text
JSON / fixture params          InterruptPolicy
        │                              │
        ▼                              ▼
  parse (cold)                 skip_indices span
        │                              │
        └──────────► kernel(in, out, POD / spans) ──► out[i]
```

- `Transform::apply_into` — caller-owned output buffer
- `Transform::apply` — allocates once, delegates to `apply_into`
- `Family::kernel(...)` — no JSON; CUDA-shaped signature
- `compose` — ≤2 length-N scratch buffers (ping-pong)

## Parity transcript

`ParityRecord` (`parcae/parity/parity_record.hpp`) hashes:

| Field | Source |
|-------|--------|
| `params_hash_sha256` | compact `params.dump()` |
| `input_sha256` / `output_sha256` | raw `uint8_t` Index29 stream |
| `interrupt_sha256` | compact `interrupt.to_json().dump()` |

Use `ParityRecord::apply_and_capture` after a CPU (later: CUDA) run. Procedure
for golden digests: [`cuda-handoff.md`](cuda-handoff.md).

## Tests & fixtures

| Tag / suite | Role |
|-------------|------|
| `[solved]` | Locked solved-page oracles (CI gate for the CPU reference tag) |
| `[transform]` / `[property]` | Hand vectors + seeded round-trips |
| `[cuda][property]` | CUDA `CudaBackend` encrypt↔decrypt + CPU byte match |
| `[score]` / `[batch]` / `[tool]` | Scores, generators, library API |
| `cli_*` ctests | Smoke the CLIs (incl. parity / search-run when built) |

Locked fixture ids: `a-warning`, `some-wisdom`, `loss-of-divinity`,
`an-instruction`, `koan-1`, `welcome`, `koan-2`, `an-end`, `lp2-57-identity`.

## Related docs

- [`docs/spec/tools.md`](../spec/tools.md) — library + CLI contracts
- [`docs/spec/transforms.md`](../spec/transforms.md) — family schemas
- [`docs/spec/parity.md`](../spec/parity.md) — bit-identity obligations
- [`cuda-handoff.md`](cuda-handoff.md) — exit criteria and CUDA twin list
- [`cuda-reference.md`](cuda-reference.md) — CUDA twin architecture + exit checklist
- [`cuda-abi.md`](cuda-abi.md) — device SoA / interrupt encoding
- [`cuda-roadmap.md`](cuda-roadmap.md) — frozen CUDA commit roadmap
