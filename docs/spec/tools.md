# Spec: Deterministic Tool & CLI Contracts

**Status:** Normative  
**Headers (planned):** `parcae/tool/api.hpp`  
**Binaries (planned):** `parcae-tokenize`, `parcae-decode`, `parcae-score`, `parcae-validate`

## Principles

1. Tools are thin, deterministic wrappers over the library.
2. **No LLM / agent logic** inside tools (future agents call these APIs only).
3. Same inputs + flags ⇒ same stdout bytes (except explicitly documented timing
   fields, which MUST be omitted by default).
4. Nonzero exit status on any hard error.
5. UTF-8 in / UTF-8 out.

## Library API (`parcae::tool`)

### `tokenize`

```text
tokenize(source_utf8, grammar_id="rtkd-separator-grammar-v0", strict=true)
  → TokenStream | Status
```

### `apply_transform`

```text
apply_transform(TokenStream | span<Index29>, TransformEnvelope)
  → vector<Index29> | Status
```

When given a `TokenStream`, apply only to consumable runes; return either indices
only or a rebuilt stream — implementations **MUST** provide **both**:

- `apply_to_indices(...)`
- `apply_and_rebuild_text(...)` (preserves separators)

### `to_latin`

```text
to_latin(span<Index29>, label_profile="gematria-primus-v0-preferred")
  → string | Status
```

Multi-letter labels concatenated without inserted spaces (spaces come from
separators when rebuilding from a full stream). Exact join rules MUST be locked
in tests.

### `score`

```text
score(span<Index29>, score_id, score_version="v0", params_json?)
  → float | Status
```

### `validate_fixture`

```text
validate_fixture(fixture_dir_or_id, require_locked=false)
  → ValidationReport
```

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
parcae-decode --manifest <path> [--json]
parcae-decode --transform-json <path> --input <file> [--json]
```

Prints Latin plaintext (or JSON with indices + latin).

### `parcae-score`

```text
parcae-score --score-id <id> --input <file> [--indices|--runes] [--json]
```

### `parcae-validate`

```text
parcae-validate [--id <fixture_id>|--all] [--require-locked] [--json]
```

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
