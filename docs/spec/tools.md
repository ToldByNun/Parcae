# Spec: Deterministic Tool & CLI Contracts

**Status:** Normative  
**Headers (planned):** `parcae/tool/api.hpp`  
**Binaries:** `parcae-tokenize`, `parcae-decode`, `parcae-score`, `parcae-validate`

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
apply_to_indices(span<Index29> | TokenStream, TransformEnvelope)
  → vector<Index29> | Status

apply_and_rebuild_text(ctx, TokenStream, TransformEnvelope)
  → string | Status   // preserves non-rune separators
```

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
score(ctx, span<Index29>, score_id, score_version="v0", params_json?, request?)
  → float | Status
```

`chi2_english_gp_v0` auto-loads `profiles/scores/english-gp-expected-v0.json`
when `request.expected_frequencies` is null.

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

JSON shape:

```json
{
  "tokens": [
    { "kind": "Rune", "index29": 4, "consumable_index": 0, "text": "ᚱ" }
  ]
}
```

### `parcae-decode`

```text
parcae-decode --manifest <fixture_dir|manifest.json> [--json]
parcae-decode --transform-json <path> --input <file|-> [--json]
parcae-decode --input <file|-> --transform-id <id>
              [--direction decrypt|encrypt]
              [--params-json <json> | --key-indices <list> --key-latin <text> --shift <n>]
              [--skip-indices <list>]
              [--json]
```

Prints Latin plaintext (or JSON with `indices` + `latin`). Method/key/skips come
from the fixture manifest, a transform envelope JSON file, or explicit flags.

### `parcae-score`

```text
parcae-score --score-id <id> --input <file|->
             [--latin|--runes|--indices] [--params-json <json>]
             [--json] [--data-dir <path>]
parcae-score --list [--json] [--data-dir <path>]
```

Default input mode is `--latin` (letters only → delatinize). `--runes` tokenizes
UTF-8 Liber Primus text; `--indices` parses `0..28` integers. JSON shape:

```json
{ "score_id": "ic_mod29", "score_version": "v0", "value": 1.0 }
```

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

These five operations are the **only** compute primitives agents SHOULD get:

1. `tokenize`
2. `apply_transform`
3. `to_latin`
4. `score`
5. `validate_fixture`

Plus read-only listing of `transform_id` / `score_id` registries.

Agents MUST NOT receive:

- Arbitrary shell
- Mutable fixture hash rewriting
- Hidden non-deterministic “best guess” interrupt inference in reference tools

## Logging

Default: quiet success. `--json` is the structured interface. Human stderr
diagnostics MUST NOT affect stdout JSON mode.
