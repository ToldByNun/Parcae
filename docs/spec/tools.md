# Spec: Deterministic Tool & CLI Contracts

**Status:** Normative  
**Headers (planned):** `parcae/tool/api.hpp`  
**Binaries:** `parcae-tokenize`, `parcae-decode`, `parcae-score`, `parcae-validate`,
`parcae-catalog`, `parcae-generate`, `parcae-parity`, `parcae-parity-gen`, `parcae-search-run`

## Principles

1. Tools are thin, deterministic wrappers over the library.
2. **No LLM / agent logic** inside tools (future agents call these APIs only).
3. Same inputs + flags ⇒ same stdout bytes (except explicitly documented timing
   fields, which MUST be omitted by default).
4. Nonzero exit status on any hard error.
5. UTF-8 in / UTF-8 out.

## Library API (`parcae::tool`)

Headers: `parcae/tool/api.hpp`, `parcae/tool/context.hpp`,
`parcae/tool/transform_envelope.hpp`.

Callers construct a `parcae::tool::Context` with the Parcae `data/` root; every
API below takes that context (or uses only envelope/indices when no profile I/O
is required).

### `tokenize`

```text
tokenize(ctx, source_utf8, grammar_id="rtkd-separator-grammar-v0", strict=true)
  → TokenStream | Status
```

### `apply_transform`

```text
apply_to_indices(span<Index29> | TokenStream, TransformEnvelope, backend=cpu)
  → vector<Index29> | Status

apply_and_rebuild_text(ctx, TokenStream, TransformEnvelope, backend=cpu)
  → string | Status   // preserves non-rune separators
```

`backend` is `cpu` (default) or `cuda`. CUDA requires a build with
`PARCAE_BUILD_CUDA=ON` (`PARCAE_HAS_CUDA`); otherwise the call returns an error.

When given a `TokenStream`, apply only to consumable runes. Both index-only and
rebuild-text paths are provided as above.

### `to_latin`

```text
to_latin(ctx, span<Index29>, label_profile="gematria-primus-v0-preferred")
  → string | Status
```

Multi-letter labels concatenated without inserted spaces (spaces come from
separators when rebuilding from a full stream). Exact join rules MUST be locked
in tests.

### `score`

```text
score(ctx, span<Index29>, score_id, score_version="v0", params_json?, request?, backend=cpu)
  → float | Status
```

`chi2_english_gp_v0` auto-loads `profiles/scores/english-gp-expected-v0.json`
when `request.expected_frequencies` is null. `backend=cuda` dispatches to
`CudaScore` when CUDA is linked.

### `validate_fixture`

```text
validate_fixture(ctx, fixture_dir_or_id, require_locked=false)
  → ValidationReport
```

`fixture_dir_or_id` may be an absolute/relative fixture directory or a solved
fixture id resolved under `data/fixtures/solved/<id>`.

`ValidationReport` MUST include:

| Field | Meaning |
|-------|---------|
| `ok` | overall pass |
| `fixture_id` | |
| `checks[]` | named check + pass/fail + message |
| `diff_excerpt` | optional, truncated |

Checks SHOULD include: manifest schema, tokenize, interrupt validation, transform
apply, plaintext compare, literal region compare, hash compare when locked.

---

## CLI contracts

Global flags (all tools):

| Flag | Meaning |
|------|---------|
| `--json` | Machine-readable JSON on stdout |
| `--strict` | Strict tokenizer / validation (default true for validate) |
| `-h` / `--help` | Help |

### `parcae-tokenize`

```text
parcae-tokenize [--json] <file|->
```

JSON shape (`--json` → `parcae.tool_response.v0`, `result`):

```json
{
  "token_count": 3,
  "consumable_count": 1,
  "tokens": [
    {
      "kind": "Rune",
      "index29": 4,
      "consumable_index": 0,
      "byte_begin": 0,
      "byte_end": 3,
      "text": "ᚱ"
    }
  ]
}
```

### `parcae-decode`

```text
parcae-decode --manifest <fixture_dir|manifest.json> [--backend cpu|cuda] [--rebuild-text] [--json]
parcae-decode --transform-json <path> --input <file|-> [--backend cpu|cuda] [--rebuild-text] [--json]
parcae-decode --input <file|-> --transform-id <id>
              [--direction decrypt|encrypt]
              [--params-json <json> | --key-indices <list> --key-latin <text> --shift <n>]
              [--skip-indices <list>]
              [--backend cpu|cuda]
              [--rebuild-text]
              [--json]
```

Prints Latin plaintext (or JSON with `indices` + `latin`). With `--rebuild-text`, also
rebuilds UTF-8 preserving non-rune separators (`result.text` in JSON; human mode prints
that text instead of Latin). Method/key/skips come
from the fixture manifest, a transform envelope JSON file, or explicit flags.
`--backend cuda` requires a CUDA-linked build; otherwise exit status **2**.

### `parcae-score`

```text
parcae-score --score-id <id> --input <file|->
             [--latin|--runes|--indices] [--params-json <json>]
             [--backend cpu|cuda]
             [--json] [--data-dir <path>]
parcae-score --list [--json] [--data-dir <path>]
```

Default input mode is `--latin` (letters only → delatinize). `--runes` tokenizes
UTF-8 Liber Primus text; `--indices` parses `0..28` integers. JSON shape:

```json
{ "score_id": "ic_mod29", "score_version": "v0", "backend": "cpu", "value": 1.0 }
```

`parcae-validate` stays CPU-only (no `--backend`).

`--list --json` emits a rich catalog under `result.scores[]` (`score_id`,
`score_version`, `order`, `arity`) plus flat `result.score_ids` for compatibility.

### `parcae-catalog`

```text
parcae-catalog [--all] [--transforms] [--scores] [--generators] [--backends]
               [--json] [--data-dir <path>]
```

Lists agent-facing registries. With no section flags, all sections are included.
`--json` emits `parcae.tool_response.v0` with `backend: null` and a `result`
object containing the selected sections (`transforms`, `scores`/`score_ids`,
`generators`/`generator_ids`, `backends`).

### `parcae-generate`

```text
parcae-generate --generator-id <id> --input <file|->
                [--latin|--runes|--indices] [--direction decrypt|encrypt]
                [--params-json <json>] [--json] [--data-dir <path>]
parcae-generate --list [--json] [--data-dir <path>]
```

Runs `GenerateCandidates` / `GeneratorRegistry` and emits candidates. `--json`
wraps `result.candidates[]` (`TransformCandidate::to_json`) plus `count`,
`generator_id`, `direction`, `input_mode`. Default input mode is `--runes`.

### `parcae-parity`

```text
parcae-parity check [--all|--name <golden>] [--compare-cuda]
                    [--parity-dir <path>] [--data-dir <path>] [--json]
parcae-parity dump  --name <golden> [--backend cpu|cuda]
                    [--parity-dir <path>] [--data-dir <path>]
```

Replays committed goldens under `data/parity/` (see `parcae-parity-gen`).
`check` verifies CPU digests; `--compare-cuda` also runs `CudaBackend` and
requires matching digests except `backend` (`cpu` vs `cuda`). Exit **2** if
CUDA is requested but not built.

### `parcae-search-run`

```text
parcae-search-run [--backend cpu|cuda] [--family caesar]
                  [--seed <u32>] [--stream-length <n>] [--repeats <n>]
                  [--score-id <id>] [--no-compare] [--json] [--omit-timing]
                  [--data-dir <path>]
```

AI-style search dashboard: **throughput** (runes/s over transform+score),
**sweep scores** (parameter axis), and **locked-fixture eval** (scorer sanity).
`tok_per_sec` is intentionally non-deterministic; with `--json --omit-timing` it is
omitted so agent output is replayable. Each `steps[]` entry includes replayable
`params` (plus `param_hash`). `--json` uses `parcae.tool_response.v0`.
v0 families: `caesar|atbash|atbash_caesar|affine|vigenere` (CPU: caesar only).
With `--backend cuda`, also reports CPU↔CUDA score parity unless `--no-compare`.
Exit **1** if fixture eval is not all-pass or CUDA parity fails;
exit **2** if CUDA is requested but not built.

### `parcae-validate`

```text
parcae-validate --id <fixture_id|path> [--require-locked] [--json] [--data-dir <path>]
parcae-validate --all [--require-locked] [--json] [--data-dir <path>]
```

With `--all --require-locked`, only fixtures whose `verification.status` is
`locked` are selected (draft/synth fixtures are skipped).

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | all selected fixtures passed |
| 1 | validation failure |
| 2 | usage / I/O / schema error |

---

## Agent-facing stability

Library compute primitives agents SHOULD build on:

1. `tokenize`
2. `apply_transform`
3. `to_latin`
4. `score`
5. `validate_fixture`
6. `GenerateCandidates` (`from_indices` / `from_stream` / `from_source`) — `gen_*` dispatch
7. `RankCandidates` (`run` / `result_to_json`) — top-k with stable ties

Plus read-only listing of `transform_id` / `score_id` / `generator_id` registries
(`parcae-catalog`, `GenerateCandidates::list_generator_ids`).

**Normative agent CLI surface** (allow-list, deny-list, `parcae.tool_response.v0`
envelope, exit codes, `parcae-agent` loop): see [`agent-tools.md`](agent-tools.md).

Agents MUST NOT receive:

- Arbitrary shell
- Mutable fixture hash rewriting
- Hidden non-deterministic “best guess” interrupt inference in reference tools

## Logging

Default: quiet success. `--json` is the structured interface. Human stderr
diagnostics MUST NOT affect stdout JSON mode.
