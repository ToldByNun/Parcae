# Spec: Fixture Manifest Format

**Status:** Normative  
**Path (planned):** `data/fixtures/solved/<id>/manifest.json`  
**Research:** [test-material.md](../research/test-material.md)

## Directory layout

```text
data/fixtures/solved/
  ATTRIBUTION.md
  <fixture_id>/
    manifest.json
    ciphertext.txt
    plaintext.txt
    literals/                    # optional
      <name>.txt
```

Encoding: UTF-8, LF preferred; loaders MUST accept CRLF by normalizing newlines
in text files before hashing **if** `hash_newline_policy` says so (default:
normalize to LF).

## Manifest schema (`fixture_manifest_v0`)

```json
{
  "schema": "parcae.fixture_manifest.v0",
  "id": "welcome",
  "method": {
    "transform_id": "vigenere_key",
    "direction": "decrypt",
    "params": {
      "key_latin": "DIVINITY",
      "key_indices": [23, 10, 1, 10, 9, 10, 16, 26]
    },
    "interrupt": {
      "policy_id": "explicit_skip_indices_v0",
      "rune_index_base": 0,
      "skip_indices": [48, 74, 84, 132, 159, 160, 250, 421, 443, 465, 514]
    }
  },
  "files": {
    "ciphertext": "ciphertext.txt",
    "plaintext": "plaintext.txt"
  },
  "non_rune_literal_regions": [],
  "hashes": {
    "ciphertext_sha256": null,
    "plaintext_sha256": null,
    "normalized_plaintext_sha256": null
  },
  "normalization": {
    "newline_policy": "lf",
    "latin_label_profile": "gematria-primus-v0-preferred",
    "collapse_spaces": false
  },
  "verification": {
    "status": "draft",
    "skip_indices_provenance": "uncovering-cicada-wiki",
    "recomputed_ok": false,
    "notes": "Hash lock forbidden until recomputed_ok=true"
  },
  "attribution": {
    "sources": [
      "https://uncovering-cicada.fandom.com/wiki/How_the_solved_pages_of_the_Liber_Primus_were_solved"
    ],
    "license_note": "Community research transcript; not a claim of ownership of Cicada artwork"
  }
}
```

### Required fields

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.fixture_manifest.v0` |
| `id` | MUST match directory name and [test-material](../research/test-material.md) ids |
| `method` | MUST be a valid transform envelope ([transforms.md](transforms.md)) |
| `files.ciphertext` / `files.plaintext` | Relative paths inside fixture dir |
| `verification.status` | `draft` \| `locked` |

### `verification` gate (normative process)

| Status | Meaning |
|--------|---------|
| `draft` | Transcript/method present; skip lists may be inherited; hashes MAY be null |
| `locked` | `recomputed_ok` MUST be true; all hash fields MUST be non-null; CI MUST fail if validate diverges |

Transition `draft → locked` ONLY after:

```text
apply(method, ciphertext) → latin_normalize → equals(plaintext)
```

on the committed files (not on a different wiki paste).

### `non_rune_literal_regions` (required schema support)

Even when empty, the field MUST be present as an array.

Object shape:

```json
{
  "kind": "hex_string",
  "role": "plaintext_embedded",
  "value_file": "literals/deep-web-hash.txt",
  "compare": "exact",
  "optional_anchor": {
    "after_plaintext_substring": "HASHES TO"
  }
}
```

| Field | Meaning |
|-------|---------|
| `kind` | `hex_string` \| `decimal_grid` \| `raw_text` |
| `role` | `plaintext_embedded` \| `ciphertext_embedded` \| `side_channel` |
| `value_file` | Path relative to fixture dir with the literal bytes/text |
| `compare` | `exact` \| `ignore_whitespace` |
| `optional_anchor` | Optional hint for human/debug; validators MUST still compare via `value_file` |

**`an-end` MUST** declare the deep-web hash as a `hex_string` /
`plaintext_embedded` region so:

1. Tokenizer/validators never treat it as runes.
2. Plaintext hash includes the literal stably.
3. Transform kernels never see hex digits as `Index29`.

Identity pages with number grids (`some-wisdom`, `an-instruction`) SHOULD declare
`decimal_grid` regions or rely on tokenizer `Number` tokens plus full plaintext
file compare — pick one approach and test it; declaring regions is preferred.

### Hashes

- Algorithm: SHA-256, lowercase hex encoding.
- `ciphertext_sha256`: hash of ciphertext file after newline normalization.
- `plaintext_sha256`: hash of plaintext file after newline normalization.
- `normalized_plaintext_sha256`: hash of canonical Latin projection from decrypted
  indices (preferred labels), used when comparing transform output rather than
  raw file text.

Null hashes are allowed only while `verification.status == draft`.

### Negative-control manifests

Negative controls MAY live under `data/fixtures/negative/` with the same schema
plus:

```json
"expectation": { "validate_must_fail": true }
```

## Loader requirements

A conforming loader MUST:

1. Reject unknown `schema` values.
2. Reject `id` ≠ directory name.
3. Enforce interrupt validation rules.
4. Refuse to treat `locked` fixtures as passing if recomputation fails.
5. Surface `non_rune_literal_regions` to the validation pipeline.

## Out of scope

- Embedding JPG binaries
- Auto-downloading transcripts
- Mutating locked hashes without a deliberate version bump (`v1` fixture schema later)
